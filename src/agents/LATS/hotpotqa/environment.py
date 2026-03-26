import copy
import re
import string
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.tools.hotpotqa_tools.wikipedia import DocstoreExplorer


def clean_str(text: str) -> str:
    try:
        return text.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except Exception:
        return text


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    zero_metric = (0.0, 0.0, 0.0)
    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return zero_metric
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


class HotpotWikiEnv:
    """LATS HotpotQA environment (Wiki search + lookup + finish)."""

    def __init__(self) -> None:
        # Reuse the same Wikipedia integration path used by ReAct tools.
        self.docstore = DocstoreExplorer()
        self.page: Optional[str] = None
        self.obs: Optional[str] = None
        self.lookup_keyword: Optional[str] = None
        self.lookup_list: Optional[List[str]] = None
        self.lookup_cnt: int = 0
        self.steps: int = 0
        self.answer: Optional[str] = None
        self.question: Optional[str] = None
        self.gt_answer: Optional[str] = None
        self.search_time: float = 0.0
        self.num_searches: int = 0

    def reset(self, question: str, answer: str) -> str:
        self.docstore.document = None
        self.docstore.lookup_str = ""
        self.docstore.lookup_index = 0
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = 0
        self.steps = 0
        self.answer = None
        self.question = question
        self.gt_answer = answer
        self.obs = f"Question: {question}"
        return self.obs

    def clone_state(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "obs": self.obs,
            "lookup_keyword": self.lookup_keyword,
            "lookup_list": list(self.lookup_list) if self.lookup_list else None,
            "lookup_cnt": self.lookup_cnt,
            "wiki_document": copy.deepcopy(self.docstore.document),
            "wiki_lookup_str": self.docstore.lookup_str,
            "wiki_lookup_index": self.docstore.lookup_index,
            "steps": self.steps,
            "answer": self.answer,
            "question": self.question,
            "gt_answer": self.gt_answer,
            "search_time": self.search_time,
            "num_searches": self.num_searches,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self.page = state.get("page")
        self.obs = state.get("obs")
        self.lookup_keyword = state.get("lookup_keyword")
        lookup_list = state.get("lookup_list")
        self.lookup_list = list(lookup_list) if lookup_list else None
        self.lookup_cnt = int(state.get("lookup_cnt", 0))
        self.docstore.document = copy.deepcopy(state.get("wiki_document"))
        self.docstore.lookup_str = str(state.get("wiki_lookup_str", "") or "")
        self.docstore.lookup_index = int(state.get("wiki_lookup_index", 0) or 0)
        self.steps = int(state.get("steps", 0))
        self.answer = state.get("answer")
        self.question = state.get("question")
        self.gt_answer = state.get("gt_answer")
        self.search_time = float(state.get("search_time", 0.0))
        self.num_searches = int(state.get("num_searches", 0))

    @staticmethod
    def _get_page_obs(page: str) -> str:
        paragraphs = [p.strip() for p in page.split("\n") if p.strip()]
        sentences: List[str] = []
        for paragraph in paragraphs:
            sentences.extend([s.strip() + "." for s in paragraph.split(". ") if s.strip()])
        return " ".join(sentences[:5])

    def _construct_lookup_list(self, keyword: str) -> List[str]:
        if self.page is None:
            return []
        paragraphs = [p.strip() for p in self.page.split("\n") if p.strip()]
        sentences: List[str] = []
        for paragraph in paragraphs:
            sentences.extend([s.strip() + "." for s in paragraph.split(". ") if s.strip()])
        return [sent for sent in sentences if keyword.lower() in sent.lower()]

    def _search(self, entity: str, is_retry: bool = False) -> str:
        start_t = time.time()
        result = self.docstore.search(entity)
        self.search_time += time.time() - start_t
        if not is_retry:
            self.num_searches += 1
        if self.docstore.document is not None:
            self.page = self.docstore.document.page_content
        else:
            self.page = None
        if isinstance(result, str):
            text = result.strip()
            return text if text else f"No search results found for {entity}."
        return str(result)

    def _finish_metrics(self, answer: str) -> Dict[str, Any]:
        assert self.gt_answer is not None
        em = normalize_answer(answer) == normalize_answer(self.gt_answer)
        f1, _, _ = f1_score(answer, self.gt_answer)
        return {
            "answer": answer,
            "gt_answer": self.gt_answer,
            "em": int(em),
            "f1": f1,
            "reward": int(em),
            "steps": self.steps,
            "question": self.question,
        }

    def step(self, action: str) -> Tuple[str, int, bool, Dict[str, Any]]:
        if self.question is None or self.gt_answer is None:
            raise RuntimeError("Environment is not reset. Call reset(question, answer) first.")

        reward = 0
        done = False
        action = action.strip()

        lowered = action.lower()
        if lowered.startswith("search[") and action.endswith("]"):
            entity = action[action.find("[") + 1 : -1]
            self.obs = self._search(entity)
            info = {"steps": self.steps + 1, "answer": self.answer, "question": self.question}
        elif lowered.startswith("lookup[") and action.endswith("]"):
            keyword = action[action.find("[") + 1 : -1]
            try:
                lookup_obs = self.docstore.lookup(keyword)
            except ValueError:
                lookup_obs = "No Results"
            if lookup_obs == "No Results":
                self.obs = "No results."
            elif lookup_obs == "No More Results":
                self.obs = "No more results."
            else:
                self.obs = lookup_obs
            info = {"steps": self.steps + 1, "answer": self.answer, "question": self.question}
        elif lowered.startswith("finish[") and action.endswith("]"):
            answer = action[action.find("[") + 1 : -1]
            self.answer = answer
            done = True
            metrics = self._finish_metrics(answer)
            reward = metrics["reward"]
            self.obs = f"Episode finished, reward = {reward}"
            info = metrics
        elif lowered.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."
            info = {"steps": self.steps + 1, "answer": self.answer, "question": self.question}
        else:
            self.obs = f"Invalid action: {action}"
            info = {"steps": self.steps + 1, "answer": self.answer, "question": self.question}

        self.steps += 1
        return self.obs, reward, done, info

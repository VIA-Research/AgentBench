import copy
from typing import Any, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment


ACTION_TO_TEMPLATE = {
    "Description": "description_page.html",
    "Features": "features_page.html",
    "Reviews": "review_page.html",
    "Attributes": "attributes_page.html",
}


def clean_str(text: str) -> str:
    return text.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element: Any) -> bool:
    ignore = {"style", "script", "head", "title", "meta", "[document]"}
    return element.parent.name not in ignore and not isinstance(element, Comment)


def evaluate_webshop_output(output: str) -> Tuple[bool, float]:
    if "Your score (min 0.0, max 1.0): " not in output:
        return False, 0.0
    score = float(output.split("Your score (min 0.0, max 1.0): ")[-1].split()[0])
    return score == 1.0, score


class WebshopEnv:
    def __init__(self, webshop_url: str = "http://localhost:3000") -> None:
        self.webshop_url = webshop_url
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def set_webshop_url(self, url: str) -> None:
        self.webshop_url = url

    def clone_state(self) -> Dict[str, Dict[str, Any]]:
        return copy.deepcopy(self.sessions)

    def restore_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        self.sessions = copy.deepcopy(state)

    def reset(self, session_id: str) -> str:
        obs, _, _, _ = self.step(session_id, "reset")
        return obs

    def webshop_text(
        self,
        session: str,
        page_type: str,
        query_string: str = "",
        page_num: int = 1,
        asin: str = "",
        options: Optional[Dict[str, str]] = None,
        subpage: str = "",
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        del kwargs
        if options is None:
            options = {}

        if page_type == "init":
            url = f"{self.webshop_url}/{session}"
        elif page_type == "search":
            url = f"{self.webshop_url}/search_results/{session}/{query_string}/{page_num}"
        elif page_type == "item":
            url = f"{self.webshop_url}/item_page/{session}/{asin}/{query_string}/{page_num}/{options}"
        elif page_type == "item_sub":
            url = (
                f"{self.webshop_url}/item_sub_page/"
                f"{session}/{asin}/{query_string}/{page_num}/{subpage}/{options}"
            )
        elif page_type == "end":
            url = f"{self.webshop_url}/done/{session}/{asin}/{options}"
        else:
            return "Invalid page type.", {}

        html = requests.get(url, timeout=10).text
        html_obj = BeautifulSoup(html, "html.parser")
        texts = html_obj.findAll(text=True)
        visible_texts = list(filter(tag_visible, texts))

        observation = ""
        option_type = ""
        parsed_options: Dict[str, str] = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0

        for text in visible_texts:
            if text == "\n":
                continue
            if text.replace("\n", "").replace("\\n", "").replace(" ", "") == "":
                continue
            if text.parent.name == "button":
                processed_t = f"\n[{text}] "
            elif text.parent.name == "label":
                if f"'{text}'" in url:
                    processed_t = f"[[{text}]]"
                else:
                    processed_t = f"[{text}]"
                parsed_options[str(text)] = option_type
            elif text.parent.get("class") == ["product-link"]:
                processed_t = f"\n[{text}] "
                if prod_cnt >= 10:
                    processed_t = ""
                prod_cnt += 1
                asins.append(str(text))
                just_prod = 0
            else:
                processed_t = "\n" + str(text) + " "
                if cnt < 2 and page_type != "init":
                    processed_t = ""
                if just_prod <= 2 and prod_cnt >= 4:
                    processed_t = ""
                option_type = str(text)
                cnt += 1
            just_prod += 1
            observation += processed_t

        info: Dict[str, Any] = {}
        if parsed_options:
            info["option_types"] = parsed_options
        if asins:
            info["asins"] = asins
        if "Your score (min 0.0, max 1.0)" in visible_texts:
            idx = visible_texts.index("Your score (min 0.0, max 1.0)")
            score_text = str(visible_texts[idx + 1])
            info["reward"] = float(score_text)
            observation = "Your score (min 0.0, max 1.0): " + score_text

        return clean_str(observation), info

    def step(self, session_id: str, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        done = False
        reward = 0.0
        observation_override = None
        if session_id not in self.sessions and action != "reset":
            self.sessions[session_id] = {"session": session_id, "page_type": "init"}

        if action == "reset":
            self.sessions[session_id] = {"session": session_id, "page_type": "init"}
        elif action.startswith("think["):
            observation_override = "OK."
        elif action.startswith("search[") and action.endswith("]"):
            assert self.sessions[session_id]["page_type"] == "init"
            query = action[7:-1]
            self.sessions[session_id] = {
                "session": session_id,
                "page_type": "search",
                "query_string": query,
                "page_num": 1,
            }
        elif action.startswith("click[") and action.endswith("]"):
            button = action[6:-1]
            if button == "Buy Now":
                assert self.sessions[session_id]["page_type"] == "item"
                self.sessions[session_id]["page_type"] = "end"
            elif button == "Back to Search":
                assert self.sessions[session_id]["page_type"] in ["search", "item_sub", "item"]
                self.sessions[session_id] = {"session": session_id, "page_type": "init"}
            elif button == "Next >":
                assert self.sessions[session_id]["page_type"] == "search"
                self.sessions[session_id]["page_num"] += 1
            elif button == "< Prev":
                assert self.sessions[session_id]["page_type"] in ["search", "item_sub", "item"]
                if self.sessions[session_id]["page_type"] == "search":
                    self.sessions[session_id]["page_num"] -= 1
                elif self.sessions[session_id]["page_type"] == "item_sub":
                    self.sessions[session_id]["page_type"] = "item"
                elif self.sessions[session_id]["page_type"] == "item":
                    self.sessions[session_id]["page_type"] = "search"
                    self.sessions[session_id]["options"] = {}
            elif button in ACTION_TO_TEMPLATE:
                assert self.sessions[session_id]["page_type"] == "item"
                self.sessions[session_id]["page_type"] = "item_sub"
                self.sessions[session_id]["subpage"] = button
            else:
                if self.sessions[session_id]["page_type"] == "search":
                    assert button in self.sessions[session_id].get("asins", [])
                    self.sessions[session_id]["page_type"] = "item"
                    self.sessions[session_id]["asin"] = button
                elif self.sessions[session_id]["page_type"] == "item":
                    assert "option_types" in self.sessions[session_id]
                    assert button in self.sessions[session_id]["option_types"], (
                        button,
                        self.sessions[session_id]["option_types"],
                    )
                    option_type = self.sessions[session_id]["option_types"][button]
                    if "options" not in self.sessions[session_id]:
                        self.sessions[session_id]["options"] = {}
                    self.sessions[session_id]["options"][option_type] = button
                    observation_override = f"You have clicked {button}."
                elif self.sessions[session_id]["page_type"] == "item_sub":
                    assert button in ["Back to Search", "< Prev"]
        else:
            observation_override = f"Invalid action: {action}"

        if observation_override and observation_override.startswith("Invalid action"):
            obs = observation_override
            info: Dict[str, Any] = {}
        elif action.startswith("think["):
            obs, info = self.webshop_text(**self.sessions[session_id])
            obs = observation_override
        else:
            obs, info = self.webshop_text(**self.sessions[session_id])
            if observation_override:
                obs = observation_override

        self.sessions[session_id].update(info)
        reward = float(info.get("reward", 0.0))
        if reward == 1.0:
            done = True
        info["action"] = action
        info["observation"] = obs
        return obs, reward, done, info


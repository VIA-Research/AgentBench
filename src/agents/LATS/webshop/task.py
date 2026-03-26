import random
import re
from typing import Dict, List, Sequence

from src.agents.LATS.configs.webshop.prompt import (
    WEBSHOP_ACTION_EXAMPLES,
    build_webshop_feedback_prompt,
    build_webshop_prompt,
    reflection_prompt,
    score_prompt,
    score_prompt_feedback,
)
from src.agents.LATS.model_client import OpenAIChatClient


class WebShopTask:
    ZERO_SHOT_REFLECTION_PROMPT = """You are an advanced reasoning agent that can improve based on self reflection.
You will be given a failed WebShop trial. Diagnose likely failure reasons and provide
a concise plan to avoid repeating the same mistakes.

Previous Trial
{trajectory}

Reflection:
"""

    ZERO_SHOT_SCORE_PROMPT = """Given an item to purchase and a trajectory, score the trajectory quality.
At the final line, output exactly:
Thus the correctness score is {s}

Trajectory:
{input}
"""

    ZERO_SHOT_SCORE_PROMPT_FEEDBACK = """Given an item to purchase and a trajectory, score the trajectory quality.
You are also given failed trajectories and reflections; use them to calibrate the score.
At the final line, output exactly:
Thus the correctness score is {s}

Failed trajectories and reflections:
{trajectories}

Trajectory:
{input}
"""

    def __init__(self, client: OpenAIChatClient, fewshot: int = 1) -> None:
        self.client = client
        self.steps = 7
        self.stops = ["\nObservation:\n", None]
        self.value_cache: Dict[str, float] = {}
        self.fewshot = max(int(fewshot), 0)
        max_examples = len(WEBSHOP_ACTION_EXAMPLES)
        if self.fewshot > max_examples:
            print(
                f"Max fewshot examples for webshop LATS is {max_examples}. "
                f"Running with {max_examples} fewshot examples."
            )
        capped = min(self.fewshot, max_examples)
        self._prompt1 = build_webshop_prompt(capped)
        self._prompt1_feedback = build_webshop_feedback_prompt(capped)
        use_examples = capped > 0
        self._reflection_prompt = (
            reflection_prompt if use_examples else self.ZERO_SHOT_REFLECTION_PROMPT
        )
        self._score_prompt = score_prompt if use_examples else self.ZERO_SHOT_SCORE_PROMPT
        self._score_prompt_feedback = (
            score_prompt_feedback if use_examples else self.ZERO_SHOT_SCORE_PROMPT_FEEDBACK
        )

    def generate_self_reflection(
        self, failed_trajectories: Sequence[Dict], question: str
    ) -> List[Dict[str, str]]:
        reflection_mapping: List[Dict[str, str]] = []
        sampled_items = random.sample(list(failed_trajectories), min(3, len(failed_trajectories)))
        if not sampled_items:
            return reflection_mapping

        batch_messages = []
        for item in sampled_items:
            trajectory = item["trajectory"] + f"\nReward: {item['r']}\n"
            batch_messages.append(
                [
                    {"role": "user", "content": self._reflection_prompt.format(trajectory=trajectory)},
                ]
            )

        try:
            reflections = self.client.chat_batch(batch_messages)
        except Exception as e:
            print(
                "[LATS][Webshop][Reflection] chat_batch failed: "
                f"{type(e).__name__}: {e} | "
                f"failed_trajectories={len(failed_trajectories)} sampled={len(sampled_items)}"
            )
            return reflection_mapping
        for i, item in enumerate(sampled_items):
            reflection = reflections[i][0] if reflections[i] else ""
            reflection_mapping.append(
                {
                    "question": question,
                    "trajectory": item["trajectory"],
                    "reflection": reflection,
                }
            )
        return reflection_mapping

    def cot_prompt_wrap(
        self, x: str, y: str = "", reflection_mapping_list: Sequence[Dict[str, str]] = ()
    ) -> str:
        input_text = x + y
        if reflection_mapping_list:
            trajectories = ""
            for reflection_mapping in reflection_mapping_list:
                trajectories += (
                    reflection_mapping["trajectory"]
                    + "Reflection: "
                    + reflection_mapping["reflection"]
                    + "\n"
                )
            return self._prompt1_feedback.format(trajectories=trajectories, input=input_text)
        return self._prompt1.format(input=input_text)

    def value_prompt_wrap(
        self,
        x: str,
        y: str,
        failed_trajectories: Sequence[Dict] = (),
        reflections: Sequence[Dict[str, str]] = (),
    ) -> str:
        del x
        if failed_trajectories:
            failed_text = ""
            for traj, ref in zip(failed_trajectories, reflections):
                try:
                    score = int(float(traj["r"]) * 10) / 2
                except Exception:
                    score = 0
                trajectory = traj["trajectory"]
                split_trajectory = trajectory.split("Action: ")
                first_part = split_trajectory[0]
                remaining_parts = split_trajectory[2:]
                new_trajectory = "Action: ".join([first_part] + remaining_parts)
                failed_text += (
                    f"{y}\n{new_trajectory}\nReflection: {ref['reflection']}\n"
                    f"Thus the correctness score is {score}\n"
                )
            inp = y + "\n\nReflection: "
            return self._score_prompt_feedback.format(s="", trajectories=failed_text, input=inp)

        inp = y + "\n\nReflection: "
        return self._score_prompt.format(s="", input=inp)

    @staticmethod
    def value_outputs_unwrap(evaluate_output: str) -> float:
        if "10" in evaluate_output:
            return 1.0
        if "9" in evaluate_output:
            return 0.9
        if "8" in evaluate_output:
            return 0.8
        if "7" in evaluate_output:
            return 0.7
        if "6" in evaluate_output:
            return 0.6
        if "5" in evaluate_output:
            return 0.5
        if "4" in evaluate_output:
            return 0.4
        if "3" in evaluate_output:
            return 0.3
        if "2" in evaluate_output:
            return 0.2
        if "1" in evaluate_output:
            return 0.1
        return -1.0

    @staticmethod
    def extract_action_line(text: str) -> str:
        def normalize_action(candidate: str) -> str:
            match = re.match(r"^(search|click|think)\s*\[(.*)\]$", candidate.strip(), re.IGNORECASE)
            if not match:
                return ""
            return f"{match.group(1).lower()}[{match.group(2).strip()}]"

        # First try the explicit "Action:" lines.
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped.startswith("Action"):
                continue
            match = re.match(r"^Action(?:\s*\d+)?\s*:\s*(.*)$", stripped, re.IGNORECASE)
            if not match:
                continue
            action = normalize_action(match.group(1))
            if action:
                return action

        # Qwen-style models may emit hidden reasoning tags before the real tool call:
        # "Action: <think> ... </think>\nclick[...]". Strip those tags and recover the
        # actual action from the remaining text.
        cleaned = re.sub(r"(?is)<think>.*?</think>", "\n", text)
        cleaned = re.sub(r"(?i)</?think>", "\n", cleaned)

        for line in cleaned.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            action = normalize_action(stripped)
            if action:
                return action
            match = re.match(r"^Action(?:\s*\d+)?\s*:\s*(.*)$", stripped, re.IGNORECASE)
            if match:
                action = normalize_action(match.group(1))
                if action:
                    return action

        fallback_match = re.findall(r"(?i)(search|click|think)\s*\[(.*?)\]", cleaned)
        if fallback_match:
            tool_name, tool_input = fallback_match[-1]
            return f"{tool_name.lower()}[{tool_input.strip()}]"
        return ""

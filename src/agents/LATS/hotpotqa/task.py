import random
from typing import Dict, List, Sequence

from src.agents.LATS.configs.hotpotqa.prompt import (
    HOTPOTQA_COT_EXAMPLES,
    HOTPOTQA_REFLECTION_EXAMPLES,
    build_hotpotqa_cot_feedback_prompt,
    build_hotpotqa_cot_prompt,
    build_hotpotqa_cot_short_prompt,
    build_hotpotqa_reflection_sys_prompt,
    reflection_usr_msg,
    value_prompt_reasoning,
    value_prompt_reasoning_feedback,
    value_prompt_reasoning_feedback_short,
)
from src.agents.LATS.model_client import OpenAIChatClient


class HotpotQATask:
    def __init__(self, client: OpenAIChatClient, fewshot: int = 1) -> None:
        self.client = client
        self.steps = 7
        self.stops = ["\nObservation:\n", None]
        self.value_cache: Dict[str, float] = {}
        self.fewshot = max(int(fewshot), 0)
        max_examples = len(HOTPOTQA_COT_EXAMPLES)
        if self.fewshot > max_examples:
            print(
                f"Max fewshot examples for hotpotqa LATS is {max_examples}. "
                f"Running with {max_examples} fewshot examples."
            )
        capped = min(self.fewshot, max_examples)
        self._cot_prompt = build_hotpotqa_cot_prompt(capped)
        self._cot_prompt_short = build_hotpotqa_cot_short_prompt(capped)
        self._cot_prompt_feedback = build_hotpotqa_cot_feedback_prompt(capped)

        max_reflection_examples = len(HOTPOTQA_REFLECTION_EXAMPLES)
        reflection_capped = min(self.fewshot, max_reflection_examples)
        self._reflection_sys_msg = build_hotpotqa_reflection_sys_prompt(reflection_capped)

    def generate_self_reflection(
        self, failed_trajectories: Sequence[str], question: str
    ) -> List[Dict[str, str]]:
        reflection_mapping: List[Dict[str, str]] = []
        sampled = random.sample(list(failed_trajectories), min(3, len(failed_trajectories)))
        if not sampled:
            return reflection_mapping

        batch_messages = []
        for traj in sampled:
            trajectory = f"Question: {question}\n{traj}\n"
            batch_messages.append(
                [
                    {"role": "system", "content": self._reflection_sys_msg},
                    {"role": "user", "content": reflection_usr_msg.format(trajectory=trajectory)},
                ]
            )
        try:
            reflections = self.client.chat_batch(batch_messages)
        except Exception as e:
            print(
                "[LATS][Hotpot][Reflection] chat_batch failed: "
                f"{type(e).__name__}: {e} | "
                f"failed_trajectories={len(failed_trajectories)} sampled={len(sampled)}"
            )
            return reflection_mapping
        for i, traj in enumerate(sampled):
            text = reflections[i][0] if reflections[i] else ""
            reflection_mapping.append(
                {
                    "question": question,
                    "trajectory": traj,
                    "reflection": text,
                }
            )
        return reflection_mapping

    def cot_prompt_wrap(
        self,
        messages: List[Dict[str, str]],
        y: str = "",
        reflection_mapping_list: Sequence[Dict[str, str]] = (),
    ) -> List[Dict[str, str]]:
        del y
        if reflection_mapping_list:
            trajectories = ""
            for mapping in reflection_mapping_list:
                trajectories += (
                    mapping["trajectory"]
                    + "FAILED TRAJECTORY\nReflection: "
                    + mapping["reflection"]
                    + "\n\n"
                )
            system_prompt = self._cot_prompt_feedback.format(trajectories=trajectories, input="")
        else:
            # Keep original prompt behavior; short prompt fallback if needed.
            system_prompt = self._cot_prompt
            if len(system_prompt) > 30000:
                system_prompt = self._cot_prompt_short
            system_prompt = system_prompt.format(input="")
        return [{"role": "system", "content": system_prompt}] + messages

    def value_prompt_wrap(
        self,
        x: str,
        y: List[Dict[str, str]],
        failed_trajectories: Sequence[str] = (),
        reflections: Sequence[Dict[str, str]] = (),
    ) -> List[Dict[str, str]]:
        question = x.split("\n")[0]
        if failed_trajectories and reflections:
            combined = ""
            for traj, ref in zip(failed_trajectories, reflections):
                combined += (
                    f"{question}\n{traj}\n"
                    f"This trajectory is incorrect as {ref['reflection']}\n"
                    "Thus the correctness score is 1\n"
                )
            current = ""
            for msg in y:
                current += msg["content"] + "\n"
            current += "This trajectory is "
            if len(current) > 30000:
                system = value_prompt_reasoning_feedback_short.format(
                    s="", trajectories=combined, input=""
                )
            else:
                system = value_prompt_reasoning_feedback.format(
                    s="", trajectories=combined, input=""
                )
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": current},
            ]

        trajectory = ""
        for msg in y:
            trajectory += msg["content"] + "\n"
        trajectory += "This trajectory is "
        return [
            {"role": "system", "content": value_prompt_reasoning.format(s="", input="")},
            {"role": "user", "content": trajectory},
        ]

    @staticmethod
    def value_outputs_unwrap(evaluate_output: str) -> float:
        if "Thus" in evaluate_output:
            evaluate_output = evaluate_output.split("Thus", 1)[1]
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

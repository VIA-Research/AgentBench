import random
from typing import Dict, List, Sequence

from src.agents.LATS.configs.math.prompt import (
    MATH_COT_EXAMPLES,
    build_math_cot_prompt,
    build_math_cot_short_prompt,
    cot_prompt_feedback,
    cot_prompt_feedback_short,
    reflection_sys_msg,
    reflection_usr_msg,
    value_prompt_reasoning,
    value_prompt_reasoning_feedback,
    value_prompt_reasoning_feedback_short,
)
from src.agents.LATS.model_client import OpenAIChatClient


def _fill_template(template: str, **replacements: str) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


class MathTask:
    def __init__(self, client: OpenAIChatClient, fewshot: int = 1) -> None:
        self.client = client
        self.steps = 7
        self.stops = ["\nObservation:\n", None]
        self.value_cache: Dict[str, float] = {}
        self.fewshot = max(int(fewshot), 0)
        max_examples = len(MATH_COT_EXAMPLES)
        if self.fewshot > max_examples:
            print(
                f"Max fewshot examples for math LATS is {max_examples}. "
                f"Running with {max_examples} fewshot examples."
            )
        capped = min(self.fewshot, max_examples)
        self._cot_prompt = build_math_cot_prompt(capped)
        self._cot_prompt_short = build_math_cot_short_prompt(capped)

    def generate_self_reflection(
        self, failed_trajectories: Sequence[str], problem: str
    ) -> List[Dict[str, str]]:
        reflection_mapping: List[Dict[str, str]] = []
        sampled = random.sample(list(failed_trajectories), min(3, len(failed_trajectories)))
        if not sampled:
            return reflection_mapping

        batch_messages = []
        for traj in sampled:
            trajectory = f"{problem}\n{traj}\n"
            batch_messages.append(
                [
                    {"role": "system", "content": reflection_sys_msg},
                    {"role": "user", "content": reflection_usr_msg.format(trajectory=trajectory)},
                ]
            )

        try:
            reflections = self.client.chat_batch(batch_messages)
        except Exception as e:
            print(
                "[LATS][Math][Reflection] chat_batch failed: "
                f"{type(e).__name__}: {e} | "
                f"failed_trajectories={len(failed_trajectories)} sampled={len(sampled)}"
            )
            return reflection_mapping

        for i, traj in enumerate(sampled):
            reflection_text = reflections[i][0] if reflections[i] else ""
            reflection_mapping.append(
                {
                    "problem": problem,
                    "trajectory": traj,
                    "reflection": reflection_text,
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
                    + "\nReflection: "
                    + mapping["reflection"]
                    + "\n\n"
                )
            system_prompt = _fill_template(
                cot_prompt_feedback,
                trajectories=trajectories,
                input="",
            )
            if len(system_prompt) > 30000:
                system_prompt = _fill_template(
                    cot_prompt_feedback_short,
                    trajectories=trajectories,
                    input="",
                )
        else:
            system_prompt = _fill_template(self._cot_prompt, input="")
            if len(system_prompt) > 30000:
                system_prompt = _fill_template(self._cot_prompt_short, input="")
        return [{"role": "system", "content": system_prompt}] + messages

    def value_prompt_wrap(
        self,
        x: str,
        y: List[Dict[str, str]],
        failed_trajectories: Sequence[str] = (),
        reflections: Sequence[Dict[str, str]] = (),
    ) -> List[Dict[str, str]]:
        if failed_trajectories and reflections:
            combined = ""
            for traj, ref in zip(failed_trajectories, reflections):
                combined += (
                    f"{x}\n{traj}\n"
                    f"This trajectory is likely incorrect because: {ref['reflection']}\n"
                    "Thus the correctness score is 1\n\n"
                )
            current = ""
            for msg in y:
                current += msg["content"] + "\n"
            current += "Thus the correctness score is "
            if len(current) > 30000:
                system = _fill_template(
                    value_prompt_reasoning_feedback_short,
                    s="",
                    trajectories=combined,
                    input="",
                )
            else:
                system = _fill_template(
                    value_prompt_reasoning_feedback,
                    s="",
                    trajectories=combined,
                    input="",
                )
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": current},
            ]

        trajectory = ""
        for msg in y:
            trajectory += msg["content"] + "\n"
        trajectory += "Thus the correctness score is "
        return [
            {
                "role": "system",
                "content": _fill_template(value_prompt_reasoning, s="", input=""),
            },
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

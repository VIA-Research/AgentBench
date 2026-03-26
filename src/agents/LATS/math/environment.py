from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.tools.math_tools.math_equivalence import evaluate_math
from src.tools.math_tools.math_tools import CalculatorTool, FinishTool, WolframAlphaTool


class MathEnv:
    """LATS math environment (WolframAlpha + Calculator + Finish)."""

    def __init__(self) -> None:
        self.wolfram_alpha_tool = WolframAlphaTool(name="WolframAlpha")
        self.calc_tool = CalculatorTool(name="simplecalc")
        self.finish_tool = FinishTool(name="finish")
        self.problem: Optional[str] = None
        self.label: Optional[str] = None
        self.obs: Optional[str] = None
        self.steps: int = 0
        self.answer: Optional[str] = None
        self.done: bool = False

    def reset(self, problem: str, answer: str) -> str:
        self.problem = problem
        self.label = answer
        self.obs = f"Problem: {problem}"
        self.steps = 0
        self.answer = None
        self.done = False
        return self.obs

    def clone_state(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "label": self.label,
            "obs": self.obs,
            "steps": self.steps,
            "answer": self.answer,
            "done": self.done,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        self.problem = state.get("problem")
        self.label = state.get("label")
        self.obs = state.get("obs")
        self.steps = int(state.get("steps", 0))
        self.answer = state.get("answer")
        self.done = bool(state.get("done", False))

    def _default_info(self) -> Dict[str, Any]:
        return {
            "steps": self.steps + 1,
            "answer": self.answer,
            "problem": self.problem,
        }

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self.problem is None or self.label is None:
            raise RuntimeError("Environment is not reset. Call reset(problem, answer) first.")

        action = action.strip()
        lowered = action.lower()
        reward = 0.0
        done = False

        if (lowered.startswith("wolframalpha[") or lowered.startswith("search[")) and action.endswith("]"):
            query = action[action.find("[") + 1 : -1]
            try:
                self.obs = str(self.wolfram_alpha_tool.invoke(query))
            except Exception as e:
                self.obs = f"WolframAlpha error: {type(e).__name__}: {e}"
            info = self._default_info()
        elif (lowered.startswith("simplecalc[") or lowered.startswith("calculator[")) and action.endswith("]"):
            expression = action[action.find("[") + 1 : -1]
            try:
                self.obs = str(self.calc_tool.invoke(expression))
            except Exception as e:
                self.obs = f"Calculator error: {type(e).__name__}: {e}"
            info = self._default_info()
        elif lowered.startswith("finish[") and action.endswith("]"):
            answer = action[action.find("[") + 1 : -1]
            self.answer = answer
            done = True
            self.done = True
            try:
                _ = self.finish_tool.invoke(answer)
            except Exception:
                pass
            is_correct, _ = evaluate_math(answer, self.label)
            reward = 1.0 if is_correct else 0.0
            self.obs = f"Episode finished, reward = {int(reward)}"
            info = {
                "answer": answer,
                "gt_answer": self.label,
                "em": int(is_correct),
                "reward": reward,
                "steps": self.steps + 1,
                "problem": self.problem,
            }
        elif lowered.startswith("think[") and action.endswith("]"):
            self.obs = "OK."
            info = self._default_info()
        else:
            self.obs = f"Invalid action: {action}"
            reward = -1.0
            info = self._default_info()

        self.steps += 1
        return self.obs, reward, done, info

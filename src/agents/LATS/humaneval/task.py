from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

from langchain_core.messages import AIMessage
from langchain_core.tools import Tool

from src.agents.LATS.configs.humaneval.prompt import HUMANEVAL_ACTION_EXAMPLES
from src.agents.LATS.model_client import OpenAIChatClient
from src.tools.humaneval_tools.coding_tools import parse_code_block
from src.tools.humaneval_tools.executors import executor_factory
from src.tools.humaneval_tools.executors.executor_types import Executor
from src.tools.humaneval_tools.generators import generator_factory
from src.tools.humaneval_tools.generators.generator_types import Generator
from src.tools.humaneval_tools.generators.model import Message as HEMessage


class HumanevalModelAdapter:
    """Adapter bridging humaneval generator interfaces to OpenAIChatClient."""

    def __init__(self, client: OpenAIChatClient, default_temperature: float = 0.0) -> None:
        self.client = client
        self.is_chat = True
        self.default_temperature = default_temperature

    def _to_openai_messages(self, messages: Sequence[HEMessage]) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def generate_chat(
        self,
        messages: List[HEMessage],
        temperature: float = 0.0,
        num_comps: int = 1,
    ) -> Union[str, List[str]]:
        outputs = self.client.chat(
            messages=self._to_openai_messages(messages),
            n=num_comps,
            temperature=temperature if temperature is not None else self.default_temperature,
        )
        if num_comps == 1:
            return outputs[0] if outputs else ""
        return outputs

    def stream(self, messages: Sequence[Any]):
        converted: List[Dict[str, str]] = []
        for m in messages:
            role = "user"
            mtype = getattr(m, "type", "")
            if mtype in ("system", "human", "ai"):
                role = {"system": "system", "human": "user", "ai": "assistant"}[mtype]
            content = str(getattr(m, "content", ""))
            converted.append({"role": role, "content": content})

        outputs = self.client.chat(
            messages=converted,
            n=1,
            temperature=self.default_temperature,
        )
        text = outputs[0] if outputs else ""
        yield AIMessage(content=text)


@dataclass
class HumanevalTask:
    client: OpenAIChatClient
    print_log: bool = False
    fewshot: int = 0

    def __post_init__(self) -> None:
        self.generator: Generator = generator_factory("python")
        self.executor: Executor = executor_factory("python", is_leet=False)
        self.model_adapter = HumanevalModelAdapter(self.client, default_temperature=0.0)
        self.sampling_model_adapter = HumanevalModelAdapter(self.client, default_temperature=0.8)
        requested = max(int(self.fewshot), 0)
        max_examples = len(HUMANEVAL_ACTION_EXAMPLES)
        if requested > max_examples:
            self._log(
                f"Max humaneval fewshot examples is {max_examples}. "
                f"Running with {max_examples} examples."
            )
        self._fewshot_examples: List[str] = HUMANEVAL_ACTION_EXAMPLES[: min(requested, max_examples)]
        if self._fewshot_examples:
            self._log(f"Loaded {len(self._fewshot_examples)} humaneval fewshot example(s).")

    def _log(self, msg: str) -> None:
        if self.print_log:
            print(f"[LATS][HumanEval][Task] {msg}")

    def _apply_fewshot(self, func_sig: str) -> str:
        if not self._fewshot_examples:
            return func_sig
        examples_text = "\n\n".join(self._fewshot_examples)
        return (
            "Use the following HumanEval few-shot examples for guidance.\n\n"
            "[Few-shot examples]\n"
            f"{examples_text}\n\n"
            "[Target task]\n"
            f"{func_sig}"
        )

    def generate_internal_tests(self, func_sig: str, max_num_tests: int = 3) -> List[str]:
        try:
            runner = Tool(
                name="generate_internal_tests",
                description="Generate internal unit tests for HumanEval candidate.",
                func=lambda sig: self.generator.internal_tests(
                    func_sig=sig,
                    model=self.model_adapter,
                    max_num_tests=max_num_tests,
                ),
            )
            tests = runner.invoke(func_sig)
            return [t for t in tests if t.strip()]
        except Exception as e:
            self._log(f"internal_tests failed: {type(e).__name__}: {e}")
            return []

    def generate_solution_simple(
        self,
        func_sig: str,
        temperature: float = 0.8,
    ) -> str:
        prompt_func_sig = self._apply_fewshot(func_sig)
        output = self.generator.func_impl(
            func_sig=prompt_func_sig,
            model=self.sampling_model_adapter,
            strategy="simple",
            num_comps=1,
            temperature=temperature,
        )
        assert isinstance(output, str)
        parsed = parse_code_block(output)
        return parsed or output

    def generate_solution_mcts(
        self,
        func_sig: str,
        prev_solutions: Sequence[str],
        acc_feedback: Sequence[str],
        acc_reflection: Sequence[str],
        sampling_temperature: float,
    ) -> str:
        prompt_func_sig = self._apply_fewshot(func_sig)
        output = self.generator.func_impl(
            func_sig=prompt_func_sig,
            model=self.sampling_model_adapter,
            strategy="mcts",
            prev_func_impl=list(prev_solutions),
            num_comps=1,
            temperature=sampling_temperature,
            acc_feedback=list(acc_feedback),
            acc_reflection=list(acc_reflection),
        )
        assert isinstance(output, str)
        parsed = parse_code_block(output)
        return parsed or output

    def generate_self_reflection(self, func_impl: str, feedback: str) -> str:
        try:
            return self.generator.self_reflection(
                func=func_impl,
                feedback=feedback,
                model=self.model_adapter,
            )
        except Exception as e:
            self._log(f"self_reflection failed: {type(e).__name__}: {e}")
            return ""

    def execute_internal(self, func_impl: str, tests: Sequence[str]) -> Tuple[bool, str, float]:
        if not tests:
            return False, "No internal tests generated.", 0.0
        try:
            runner = Tool(
                name="execute_internal_tests",
                description="Execute candidate implementation on generated internal tests.",
                func=lambda impl: self.executor.execute(impl, list(tests), timeout=10),
            )
            result = runner.invoke(func_impl)
            passed = int(sum(1 for s in result.state if s))
            rate = passed / max(len(result.state), 1)
            return bool(result.is_passing), str(result.feedback), float(rate)
        except Exception as e:
            return False, f"Internal execution failed: {type(e).__name__}: {e}", 0.0

    def evaluate_final(self, entry_point: str, func_impl: str, test: str) -> bool:
        try:
            runner = Tool(
                name="evaluate_final_solution",
                description="Run final HumanEval correctness test for candidate.",
                func=lambda impl: self.executor.evaluate(entry_point, impl, test, timeout=10),
            )
            return bool(runner.invoke(func_impl))
        except Exception:
            return False

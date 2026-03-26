from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.agents.LATS.humaneval.task import HumanevalTask


@dataclass
class HumanevalLATSConfig:
    iterations: int = 20
    n_generate_sample: int = 5
    n_evaluate_sample: int = 3
    max_depth: int = 8
    sampling_temperature: float = 1.0
    print_log: bool = False


class Node:
    def __init__(self, solution: str, parent: Optional["Node"] = None, depth: int = 0) -> None:
        self.solution = solution
        self.parent = parent
        self.depth = depth
        self.children: List["Node"] = []
        self.value = 0.0
        self.visits = 0
        self.reflection = ""
        self.test_feedback = ""
        self.reward = 0.0
        self.is_terminal = False
        self.is_solved = False

    def uct(self, exploration_weight: float = 1.0) -> float:
        if self.visits == 0:
            return self.value
        if self.parent is None:
            return self.value
        parent_visits = max(self.parent.visits, 1)
        return (self.value / self.visits) + exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits
        )

    def update(self, reward: float) -> None:
        self.visits += 1
        self.value += reward


def gather_context_from_tree(node: Node) -> Tuple[List[str], List[str], List[str]]:
    accumulated_solutions: List[str] = []
    accumulated_feedback: List[str] = []
    accumulated_reflection: List[str] = []

    cur: Optional[Node] = node
    while cur:
        if cur.solution:
            accumulated_solutions.append(cur.solution)
        if cur.test_feedback:
            accumulated_feedback.append(cur.test_feedback)
        if cur.reflection:
            accumulated_reflection.append(cur.reflection)
        cur = cur.parent

    return (
        accumulated_solutions[::-1],
        accumulated_feedback[::-1],
        accumulated_reflection[::-1],
    )


class HumanevalLATSRunner:
    def __init__(self, task: HumanevalTask, config: HumanevalLATSConfig) -> None:
        self.task = task
        self.config = config

    @staticmethod
    def _colorize_log(msg: str) -> str:
        if os.getenv("NO_COLOR"):
            return f"[LATS][HumanEval] {msg}"

        reset = "\033[0m"
        bold = "\033[1m"
        cyan = "\033[36m"
        blue = "\033[34m"

        prefix = f"{bold}{cyan}[LATS][HumanEval]{reset}"
        colored = msg
        colored = colored.replace("Reward:", f"{cyan}Reward:{reset}")
        colored = colored.replace("Terminal:", f"{blue}Terminal:{reset}")
        if msg.startswith("==== New MCTS iteration"):
            colored = f"{blue}{colored}{reset}"
        elif msg.startswith("Selected node"):
            colored = f"{blue}{colored}{reset}"
        elif msg.startswith("Expanded -> children added:"):
            colored = f"{blue}{colored}{reset}"
        elif msg.startswith("Generated internal tests:"):
            colored = f"{blue}{colored}{reset}"
        return f"{prefix} {colored}"

    def _log(self, msg: str) -> None:
        if self.config.print_log:
            print(self._colorize_log(msg))

    def _select_leaf(self, root: Node) -> Node:
        node = root
        while node.children:
            node = max(node.children, key=lambda child: child.uct())
        return node

    @staticmethod
    def _backpropagate(child: Node, reward: float) -> None:
        child.update(reward)
        curr = child.parent
        while curr is not None:
            curr.update(reward)
            curr = curr.parent

    async def _aget_sampled_solutions(
        self,
        func_sig: str,
        prev_solutions: Sequence[str],
        acc_feedback: Sequence[str],
        acc_reflection: Sequence[str],
    ) -> List[str]:
        tasks = [
            asyncio.create_task(
                asyncio.to_thread(
                    self.task.generate_solution_mcts,
                    func_sig=func_sig,
                    prev_solutions=prev_solutions,
                    acc_feedback=acc_feedback,
                    acc_reflection=acc_reflection,
                    sampling_temperature=self.config.sampling_temperature,
                )
            )
            for _ in range(self.config.n_generate_sample)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        samples: List[str] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._log(f"Skip failed expansion candidate task idx={i}: {type(result).__name__}: {result}")
                continue
            text = str(result).strip()
            if not text:
                self._log(f"Skip empty expansion candidate task idx={i}")
                continue
            samples.append(text)
        return samples

    async def _aevaluate_one_child(
        self,
        child: Node,
        tests_i: Sequence[str],
        entry_point: str,
        final_test: str,
        is_last_iteration: bool,
    ) -> Dict[str, object]:
        is_passing_internal, feedback_internal, reward_internal = await asyncio.to_thread(
            self.task.execute_internal,
            child.solution,
            list(tests_i),
        )
        child.reward = reward_internal
        child.test_feedback = feedback_internal

        reward_real = 0.0
        if not is_passing_internal:
            child.reflection = await asyncio.to_thread(
                self.task.generate_self_reflection,
                child.solution,
                feedback_internal,
            )
        else:
            child.is_terminal = True

        solved = False
        if is_passing_internal or is_last_iteration:
            passed_final = await asyncio.to_thread(
                self.task.evaluate_final,
                entry_point,
                child.solution,
                final_test,
            )
            if passed_final:
                child.is_solved = True
                child.is_terminal = True
                reward_real = 1.0
                solved = True

        reward = reward_internal + reward_real
        return {"child": child, "reward": reward, "solved": solved}

    async def _aevaluate_children(
        self,
        children: Sequence[Node],
        tests_i: Sequence[str],
        entry_point: str,
        final_test: str,
        is_last_iteration: bool,
    ) -> List[Dict[str, object]]:
        tasks = [
            asyncio.create_task(
                self._aevaluate_one_child(
                    child=child,
                    tests_i=tests_i,
                    entry_point=entry_point,
                    final_test=final_test,
                    is_last_iteration=is_last_iteration,
                )
            )
            for child in children
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: List[Dict[str, object]] = []
        for idx, (child, result) in enumerate(zip(children, results), start=1):
            if isinstance(result, Exception):
                self._log(
                    f"Child eval failed {idx}/{len(children)}: "
                    f"{type(result).__name__}: {result}"
                )
                child.test_feedback = f"Child eval failed: {type(result).__name__}: {result}"
                child.reward = 0.0
                merged.append({"child": child, "reward": 0.0, "solved": False})
                continue
            merged.append(result)
        return merged

    def _expand(
        self,
        node: Node,
        func_sig: str,
    ) -> List[Node]:
        if node.depth >= self.config.max_depth:
            node.is_terminal = True
            return []

        prev_solutions, acc_feedback, acc_reflection = gather_context_from_tree(node)
        new_children: List[Node] = []
        seen = set()
        sampled_solutions = asyncio.run(
            self._aget_sampled_solutions(
                func_sig=func_sig,
                prev_solutions=prev_solutions,
                acc_feedback=acc_feedback,
                acc_reflection=acc_reflection,
            )
        )
        for new_solution in sampled_solutions:

            key = new_solution.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            child = Node(solution=new_solution, parent=node, depth=node.depth + 1)
            node.children.append(child)
            new_children.append(child)
        return new_children

    def run(self, item: Dict[str, str]) -> Dict:
        func_sig = item["prompt"]
        entry_point = item["entry_point"]
        final_test = item["test"]

        internal_tests_count = 0
        iterations_used = 0
        is_solved = False
        solved_solution = ""

        tests_i = self.task.generate_internal_tests(
            func_sig=func_sig,
            max_num_tests=max(self.config.n_evaluate_sample, 1),
        )
        internal_tests_count += len(tests_i)
        self._log(f"Generated internal tests: {len(tests_i)}")

        root_solution = self.task.generate_solution_simple(func_sig=func_sig)
        root = Node(root_solution)

        is_passing_internal, feedback, reward_internal = self.task.execute_internal(root.solution, tests_i)
        root.test_feedback = feedback
        root.reward = reward_internal
        root.is_terminal = is_passing_internal

        if is_passing_internal:
            passed_final = self.task.evaluate_final(entry_point, root.solution, final_test)
            if passed_final:
                is_solved = True
                solved_solution = root.solution
        else:
            root.reflection = self.task.generate_self_reflection(root.solution, feedback)

        if is_solved:
            return {
                "output": "Answer: True",
                "ispass": True,
                "score": None,
                "reward": 1.0,
                "solution": solved_solution,
                "iterations_used": iterations_used,
                "internal_tests_count": internal_tests_count,
            }

        for it in range(self.config.iterations):
            iterations_used = it + 1
            self._log(f"==== New MCTS iteration ====")

            tests_i = self.task.generate_internal_tests(
                func_sig=func_sig,
                max_num_tests=max(self.config.n_evaluate_sample, 1),
            )
            internal_tests_count += len(tests_i)

            node = self._select_leaf(root)
            self._log(
                f"Selected node depth={node.depth}, visits={node.visits}, value={node.value:.3f}, terminal={node.is_terminal}"
            )

            new_children = self._expand(node, func_sig)
            self._log(f"Expanded -> children added: {len(new_children)}")
            if not new_children:
                self._backpropagate(node, node.reward)
                continue

            self._log(f"Evaluating {len(new_children)} children in parallel")
            evaluated = asyncio.run(
                self._aevaluate_children(
                    children=new_children,
                    tests_i=tests_i,
                    entry_point=entry_point,
                    final_test=final_test,
                    is_last_iteration=(it == self.config.iterations - 1),
                )
            )

            for idx, result in enumerate(evaluated, start=1):
                child = result["child"]
                reward = float(result["reward"])
                self._backpropagate(child, reward)
                self._log(
                    f"Child {idx}/{len(new_children)} depth={child.depth} | "
                    f"Reward: {reward:.3f} | Terminal: {child.is_terminal}"
                )

                if bool(result["solved"]) and not is_solved:
                    is_solved = True
                    solved_solution = child.solution

            if is_solved:
                break

        if is_solved:
            return {
                "output": "Answer: True",
                "ispass": True,
                "score": None,
                "reward": 1.0,
                "solution": solved_solution,
                "iterations_used": iterations_used,
                "internal_tests_count": internal_tests_count,
            }

        best_solution = root.solution
        if root.children:
            best_solution = max(root.children, key=lambda child: child.value).solution
        final_pass = self.task.evaluate_final(entry_point, best_solution, final_test)
        return {
            "output": f"Answer: {final_pass}",
            "ispass": bool(final_pass),
            "score": None,
            "reward": 0.0,
            "solution": best_solution,
            "iterations_used": iterations_used,
            "internal_tests_count": internal_tests_count,
        }

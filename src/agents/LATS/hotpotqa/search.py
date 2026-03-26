import asyncio
import copy
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.tools import Tool

from src.agents.LATS.hotpotqa.environment import HotpotWikiEnv
from src.agents.LATS.hotpotqa.task import HotpotQATask
from src.agents.LATS.model_client import OpenAIChatClient


@dataclass
class HotpotLATSConfig:
    iterations: int = 30
    n_generate_sample: int = 5
    max_depth: int = 7
    sampling_temperature: float = 1.0
    print_log: bool = False


class Node:
    def __init__(
        self,
        state: Optional[Dict[str, str]],
        question: str,
        parent: Optional["Node"] = None,
        env_state: Optional[Dict] = None,
    ) -> None:
        self.state = (
            {"thought": "", "action": "", "observation": ""} if state is None else state
        )
        self.parent = parent
        self.question = question
        self.children: List["Node"] = []
        self.visits = 0
        self.value = 0.0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.em = 0
        self.answer = ""
        self.info: Dict = {}
        self.env_state = env_state

    def uct(self) -> float:
        if self.visits == 0:
            return self.value
        parent_visits = max(self.parent.visits, 1) if self.parent else 1
        return self.value / self.visits + np.sqrt(2 * np.log(parent_visits) / self.visits)

    def __str__(self) -> str:
        return (
            f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, "
            f"thought={self.state['thought']}, action={self.state['action']}, "
            f"observation={self.state['observation']})"
        )


def node_trajectory_to_text(node_string: str) -> str:
    lines = node_string.split("\n")
    formatted_lines = []
    for line in lines:
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            thought = line.split(", thought=")[1].split(", action=")[0].strip()
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split(")")[0].strip()
        except Exception:
            continue

        if depth != 0:
            if thought:
                formatted_lines.append(f"Thought {depth}: {thought}")
            if action:
                formatted_lines.append(f"Action {depth}: {action}")
            if observation:
                formatted_lines.append(f"Observation {depth}: {observation}")
    return "\n".join(formatted_lines)


def collect_all_nodes(node: Node) -> List[Node]:
    nodes = [node]
    for child in node.children:
        nodes.extend(collect_all_nodes(child))
    return nodes


def collect_trajectory(node: Node) -> str:
    trajectory = []
    curr: Optional[Node] = node
    while curr:
        trajectory.append(str(curr))
        curr = curr.parent
    return "\n".join(reversed(trajectory))


def generate_prompt(node: Node) -> List[Dict[str, str]]:
    trajectory: List[Dict[str, str]] = []
    question = [{"role": "user", "content": node.question}]
    curr: Optional[Node] = node
    while curr:
        segment: List[Dict[str, str]] = []
        react_content = ""
        if curr.state.get("observation") and curr.depth != 0:
            segment.append(
                {"role": "user", "content": f"Observation {curr.depth}: {curr.state['observation']}"}
            )
        if curr.state.get("thought"):
            react_content = f"Thought {curr.depth}: {curr.state['thought']}\n"
        if curr.state.get("action"):
            react_content += f"Action {curr.depth}: {curr.state['action']}"
        if react_content:
            segment.append({"role": "assistant", "content": react_content})
        trajectory += segment
        curr = curr.parent
    trajectory.reverse()
    return question + trajectory


def get_unique_trajectories(failed_trajectories: Sequence[Dict], num: int = 5) -> List[str]:
    unique_trajectories: List[str] = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get("final_answer")
        if final_answer not in seen_final_answers:
            unique_trajectories.append(node_trajectory_to_text(traj["trajectory"]))
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories


class HotpotLATSRunner:
    def __init__(
        self,
        env: HotpotWikiEnv,
        task: HotpotQATask,
        client: OpenAIChatClient,
        config: HotpotLATSConfig,
    ) -> None:
        self.env = env
        self.task = task
        self.client = client
        self.config = config
        self.reflection_map: List[Dict[str, str]] = []
        self.failed_trajectories: List[Dict] = []

    @staticmethod
    def _colorize_log(msg: str, agent_name: str) -> str:
        if os.getenv("NO_COLOR"):
            return f"[LATS][{agent_name}] {msg}"

        reset = "\033[0m"
        bold = "\033[1m"
        cyan = "\033[36m"
        blue = "\033[34m"

        prefix = f"{bold}{cyan}[LATS][{agent_name}]{reset}"
        colored = msg
        colored = colored.replace("Thought:", f"{cyan}Thought:{reset}")
        colored = colored.replace("Action:", f"{cyan}Action:{reset}")
        colored = colored.replace("Observation:", f"{cyan}Observation:{reset}")
        colored = colored.replace("Reward:", f"{cyan}Reward:{reset}")
        colored = colored.replace("Terminal:", f"{blue}Terminal:{reset}")
        if msg.startswith("==== New MCTS iteration"):
            colored = f"{blue}{colored}{reset}"
        elif msg.startswith("Selected node"):
            colored = f"{blue}{colored}{reset}"
        elif msg.startswith("Expand node") or msg.startswith("Rollout"):
            colored = f"{cyan}{colored}{reset}"
        elif msg.startswith("Evaluate"):
            colored = f"{blue}{colored}{reset}"
        elif msg.startswith("Sampled "):
            colored = f"{blue}{colored}{reset}"
        elif msg.startswith("Child "):
            colored = f"{blue}{colored}{reset}"
        elif msg.startswith("Expanded -> children added:"):
            colored = f"{blue}{colored}{reset}"
        return f"{prefix} {colored}"

    def _log(self, msg: str) -> None:
        if self.config.print_log:
            print(self._colorize_log(msg, "Hotpot"))

    def _extract_thought_and_action(self, text: str, depth: int) -> Tuple[str, Optional[str]]:
        thought_line = ""
        action_line = None
        thought_pattern = re.compile(r"^Thought(?:\s*\d+)?\s*[:：]\s*(.*)$", re.IGNORECASE)
        action_pattern = re.compile(r"^Action(?:\s*\d+)?\s*[:：]?\s*(.*)$", re.IGNORECASE)
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            m_thought = thought_pattern.match(line)
            if m_thought:
                thought_line = m_thought.group(1).strip()
                continue
            m_action = action_pattern.match(line)
            if m_action:
                action_line = m_action.group(1).strip()
        if action_line:
            return thought_line, action_line

        # Fallback: allow direct tool-call style outputs, e.g. "Search[foo]".
        tool_match = re.search(r"(search|lookup|finish|think)\s*\[[^\]]+\]", text, re.IGNORECASE)
        if tool_match:
            return thought_line, tool_match.group(0).strip()
        return thought_line, None

    def _to_env_action(self, action_line: str) -> str:
        if "[" not in action_line or not action_line.endswith("]"):
            return action_line
        action_type = action_line.split("[", 1)[0].strip().lower()
        action_param = action_line.split("[", 1)[1][:-1]
        return f"{action_type}[{action_param}]"

    @staticmethod
    def _tool_name_from_action(action: str) -> str:
        action = action.strip()
        if not action:
            return "unknown_tool"
        if "[" in action:
            tool_name = action.split("[", 1)[0].strip().lower()
            return tool_name or "unknown_tool"
        tool_name = action.split(None, 1)[0].strip().lower()
        return tool_name or "unknown_tool"

    def _maybe_update_reflection_map(self, prompt_messages: List[Dict[str, str]]) -> None:
        unique_trajectories = get_unique_trajectories(self.failed_trajectories)
        if len(unique_trajectories) > len(self.reflection_map) and len(unique_trajectories) < 4:
            self.reflection_map = self.task.generate_self_reflection(
                unique_trajectories, prompt_messages[0]["content"]
            )

    async def _aget_samples_core(
        self,
        prompt_messages: List[Dict[str, str]],
        n_generate_sample: int,
        reflection_snapshot: Sequence[Dict[str, str]],
    ) -> List[str]:
        messages = self.task.cot_prompt_wrap(
            prompt_messages, "", reflection_snapshot
        )

        tasks = [
            asyncio.create_task(
                self.client.achat(
                    messages=messages,
                    n=1,
                    stop="Observation",
                    temperature=self.config.sampling_temperature,
                )
            )
            for _ in range(n_generate_sample)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        samples: List[str] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._log(f"Drop failed sample task idx={i}: {result}")
                continue
            if not result:
                self._log(f"Drop empty sample task idx={i}")
                continue
            samples.append(result[0])
        return samples

    def _get_samples(
        self, prompt_messages: List[Dict[str, str]], prefix: str, n_generate_sample: int
    ) -> List[str]:
        del prefix
        self._maybe_update_reflection_map(prompt_messages)
        reflection_snapshot = list(self.reflection_map)
        return asyncio.run(
            self._aget_samples_core(
                prompt_messages=prompt_messages,
                n_generate_sample=n_generate_sample,
                reflection_snapshot=reflection_snapshot,
            )
        )

    async def _aget_values_core(
        self,
        x: str,
        prompts: Sequence[List[Dict[str, str]]],
        failed_trajectories_snapshot: Sequence[str],
        reflections_snapshot: Sequence[Dict[str, str]],
    ) -> List[float]:
        if not prompts:
            return []
        value_prompts = [
            self.task.value_prompt_wrap(
                x=x,
                y=prompt,
                failed_trajectories=failed_trajectories_snapshot,
                reflections=reflections_snapshot,
            )
            for prompt in prompts
        ]

        tasks = [
            asyncio.create_task(
                self.client.achat(
                    messages=prompt_messages,
                    stop=None,
                )
            )
            for prompt_messages in value_prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        values: List[float] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._log(f"Drop failed evaluate task idx={i}: {result}")
                values.append(0.0)
                continue
            parsed = [self.task.value_outputs_unwrap(output) for output in result]
            parsed = [p for p in parsed if p >= 0]
            values.append(float(sum(parsed) / len(parsed)) if parsed else 0.0)
        return values

    def _get_values(self, x: str, prompts: Sequence[List[Dict[str, str]]]) -> List[float]:
        failed_trajectories_snapshot = get_unique_trajectories(self.failed_trajectories)
        reflections_snapshot = list(self.reflection_map)
        return asyncio.run(
            self._aget_values_core(
                x=x,
                prompts=prompts,
                failed_trajectories_snapshot=failed_trajectories_snapshot,
                reflections_snapshot=reflections_snapshot,
            )
        )

    def _run_action_with_worker(
        self,
        node_env_state: Optional[Dict],
        parent_state: Dict[str, str],
        action_line: str,
        thought_line: str,
    ) -> Dict[str, Any]:
        worker_env = HotpotWikiEnv()
        if node_env_state is not None:
            worker_env.restore_state(copy.deepcopy(node_env_state))

        env_action = self._to_env_action(action_line)
        tool_name = self._tool_name_from_action(env_action)
        tool_input = ""
        if "[" in env_action and env_action.endswith("]"):
            tool_input = env_action.split("[", 1)[1][:-1]

        def run_search(text: str) -> Dict[str, Any]:
            obs_, reward_, done_, info_ = worker_env.step(f"search[{text}]")
            return {"obs": obs_, "reward": reward_, "done": done_, "info": info_}

        def run_lookup(text: str) -> Dict[str, Any]:
            obs_, reward_, done_, info_ = worker_env.step(f"lookup[{text}]")
            return {"obs": obs_, "reward": reward_, "done": done_, "info": info_}

        def run_finish(text: str) -> Dict[str, Any]:
            obs_, reward_, done_, info_ = worker_env.step(f"finish[{text}]")
            return {"obs": obs_, "reward": reward_, "done": done_, "info": info_}

        def run_think(text: str) -> Dict[str, Any]:
            obs_, reward_, done_, info_ = worker_env.step(f"think[{text}]")
            return {"obs": obs_, "reward": reward_, "done": done_, "info": info_}

        def run_raw(_: str) -> Dict[str, Any]:
            obs_, reward_, done_, info_ = worker_env.step(env_action)
            return {"obs": obs_, "reward": reward_, "done": done_, "info": info_}

        tool_funcs = {
            "search": run_search,
            "lookup": run_lookup,
            "finish": run_finish,
            "think": run_think,
        }
        runner = Tool(
            name=tool_name,
            description=f"LATS hotpot tool call: {tool_name}",
            func=tool_funcs.get(tool_name, run_raw),
        )
        try:
            result = runner.invoke(tool_input)
            obs = str(result.get("obs", ""))
            reward = float(result.get("reward", 0))
            done = bool(result.get("done", False))
            info = result.get("info", {}) or {}
        except AssertionError:
            obs = "Invalid action!"
            reward = -1.0
            done = False
            info = {}
            
        state = copy.deepcopy(parent_state)
        state["thought"] = thought_line
        state["action"] = action_line
        state["observation"] = obs

        return {
            "state": state,
            "env_state": worker_env.clone_state(),
            "reward": reward,
            "done": done,
            "em": int(info.get("em", 0)),
            "answer": str(info.get("answer", "")),
            "info": info,
            "obs": obs,
            "env_action": env_action,
        }

    async def _agenerate_new_states(
        self,
        node: Node,
        n_generate_sample: int,
        prompt_messages: List[Dict[str, str]],
        reflection_snapshot: Sequence[Dict[str, str]],
    ) -> List[Node]:
        sampled_actions = await self._aget_samples_core(
            prompt_messages=prompt_messages,
            n_generate_sample=n_generate_sample,
            reflection_snapshot=reflection_snapshot,
        )
        self._log(f"Sampled {len(sampled_actions)} candidate actions at depth={node.depth}")

        candidates: List[Tuple[str, str]] = []
        unique_keys = set()
        for sampled_action in sampled_actions:
            thought_line, action_line = self._extract_thought_and_action(
                sampled_action, depth=node.depth + 1
            )
            if not action_line:
                compact = sampled_action.replace("\n", "\\n")
                preview = compact[:180]
                tail = compact[-120:] if len(compact) > 180 else ""
                msg = f"Skip sample len={len(compact)} (no parsable action): {preview}"
                if tail:
                    msg += f" ... {tail}"
                self._log(msg)
                continue
            unique_key = f"{thought_line}::{action_line}"
            if unique_key in unique_keys:
                continue
            unique_keys.add(unique_key)
            candidates.append((thought_line, action_line))

        tasks = [
            asyncio.create_task(
                asyncio.to_thread(
                    self._run_action_with_worker,
                    node.env_state,
                    node.state,
                    action_line,
                    thought_line,
                )
            )
            for thought_line, action_line in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_nodes: List[Node] = []
        for i, ((thought_line, action_line), result) in enumerate(zip(candidates, results)):
            if isinstance(result, Exception):
                self._log(f"Drop failed tool task idx={i}, action={action_line}: {result}")
                continue
            child = Node(
                state=result["state"],
                question=node.question,
                parent=node,
                env_state=result["env_state"],
            )
            child.is_terminal = bool(result["done"] or result["reward"] == 1)
            child.reward = result["reward"]
            child.em = result["em"]
            child.answer = result["answer"]
            child.info = result["info"]
            new_nodes.append(child)
            self._log(
                f"Child {i + 1}/{len(candidates)} depth={child.depth} | "
                f"Thought: {thought_line} | Action: {action_line} | "
                f"Reward: {child.reward} | Terminal: {child.is_terminal}\n"
                f"Observation: {result['obs']}"
            )

            if child.is_terminal and child.reward == 0:
                trajectory = collect_trajectory(child)
                self.failed_trajectories.append(
                    {"trajectory": trajectory, "final_answer": result["env_action"]}
                )
        return new_nodes

    def _generate_new_states(self, node: Node, n_generate_sample: int) -> List[Node]:
        prompt_messages = generate_prompt(node)
        self._maybe_update_reflection_map(prompt_messages)
        reflection_snapshot = list(self.reflection_map)
        return asyncio.run(
            self._agenerate_new_states(
                node=node,
                n_generate_sample=n_generate_sample,
                prompt_messages=prompt_messages,
                reflection_snapshot=reflection_snapshot,
            )
        )

    def _select_node(self, node: Node) -> Optional[Node]:
        while node and node.children:
            terminal_children = [child for child in node.children if child.is_terminal]
            if len(terminal_children) == len(node.children):
                if node.parent:
                    node.parent.children.remove(node)
                node = node.parent
                continue

            node_with_reward_1 = next(
                (child for child in terminal_children if child.reward == 1), None
            )
            if node_with_reward_1:
                return node_with_reward_1

            node = max(
                (child for child in node.children if not child.is_terminal),
                key=lambda child: child.uct(),
                default=None,
            )
            if node is None:
                break

            while node.is_terminal and node.reward != 1:
                if node.parent is None:
                    return None
                node = max(
                    (child for child in node.parent.children if not child.is_terminal),
                    key=lambda child: child.uct(),
                    default=None,
                )
                if node is None:
                    return None
        return node

    def _expand_node(self, node: Node) -> None:
        if node.depth >= self.config.max_depth:
            node.is_terminal = True
            return
        self._log(f"Expand node depth={node.depth}, n={self.config.n_generate_sample}")
        new_nodes = self._generate_new_states(node, self.config.n_generate_sample)
        node.children.extend(new_nodes)
        self._log(f"Expanded -> children added: {len(new_nodes)}")

    def _evaluate_node(self, node: Node) -> float:
        child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]
        votes = self._get_values(node.question, child_prompts)
        votes = votes + [0.0] * (len(node.children) - len(votes))
        for i, child in enumerate(node.children):
            child.value = votes[i]
        self._log(f"Evaluate depth={node.depth}, votes={votes}")
        return float(sum(votes) / len(votes)) if votes else 0.0

    def _rollout(self, node: Node, max_depth: int = 4) -> Tuple[float, Node]:
        depth = node.depth
        rewards = [0.0]
        self._log(f"Rollout start depth={depth}, max_depth={max_depth}")
        while not node.is_terminal and depth < max_depth:
            new_states: List[Node] = []
            values: List[float] = []
            while not new_states:
                new_states = self._generate_new_states(node, 5)
                if node.depth >= self.config.max_depth:
                    break
            if not new_states:
                break

            for state in new_states:
                if state.is_terminal:
                    return float(state.reward), state

            child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal]
            while not values:
                values = self._get_values(node.question, child_prompts)
                if not child_prompts:
                    break
            if not values:
                break
            max_value_index = values.index(max(values))
            rewards.append(float(max(values)))
            node = new_states[max_value_index]
            self._log(
                f"Rollout step -> picked child depth={node.depth}, value={max(values):.3f}, "
                f"reward={node.reward}, terminal={node.is_terminal}"
            )
            depth += 1
            if depth == max_depth:
                rewards = [-1.0]
        return float(sum(rewards) / len(rewards)), node

    def _backpropagate(self, node: Node, value: float) -> None:
        curr: Optional[Node] = node
        while curr:
            curr.visits += 1
            if curr.is_terminal and curr.reward == 0:
                curr.value = (curr.value * (curr.visits - 1) + (-1.0)) / curr.visits
            else:
                curr.value = (curr.value * (curr.visits - 1) + value) / curr.visits
            curr = curr.parent

    def _build_output(self, node: Node) -> str:
        if node.answer:
            return node.answer
        action = node.state.get("action", "")
        if action.lower().startswith("finish[") and action.endswith("]"):
            return action[action.find("[") + 1 : -1]
        return ""

    def run(self, question: str, answer: str) -> Dict:
        x = self.env.reset(question=question, answer=answer)
        root = Node(state=None, question=x, env_state=self.env.clone_state())
        self.failed_trajectories = []
        self.reflection_map = []
        all_nodes: List[Tuple[Node, float]] = []
        terminal_nodes: List[Node] = []

        for _ in range(self.config.iterations):
            self._log("==== New MCTS iteration ====")
            node = self._select_node(root)
            if node is not None:
                self._log(
                    f"Selected node depth={node.depth}, visits={node.visits}, "
                    f"value={node.value:.3f}, terminal={node.is_terminal}"
                )
            while node is None or (node.is_terminal and node.reward != 1):
                node = self._select_node(root)
                if node is None:
                    break
            if node is None:
                break
            if node.is_terminal and node.reward == 1:
                output = self._build_output(node)
                return {
                    "output": output,
                    "ispass": True,
                    "score": None,
                    "reward": node.reward,
                    "em": node.em,
                    "best_node": node,
                }

            self._expand_node(node)
            while node.is_terminal or not node.children:
                node = self._select_node(root)
                if node is None:
                    break
                self._expand_node(node)
            if node is None:
                break

            self._evaluate_node(node)
            reward, terminal_node = self._rollout(
                max(node.children, key=lambda child: child.value),
                max_depth=min(4, self.config.max_depth),
            )
            terminal_nodes.append(terminal_node)

            if terminal_node.reward == 1:
                output = self._build_output(terminal_node)
                return {
                    "output": output,
                    "ispass": True,
                    "score": None,
                    "reward": terminal_node.reward,
                    "em": terminal_node.em,
                    "best_node": terminal_node,
                }

            self._backpropagate(terminal_node, reward)
            all_nodes = [(n, n.value) for n in collect_all_nodes(root)]
            terminal_nodes_with_reward_1 = [
                n for n in collect_all_nodes(root) if n.is_terminal and n.reward == 1
            ]
            if terminal_nodes_with_reward_1:
                best_node = max(terminal_nodes_with_reward_1, key=lambda x: x.value)
                output = self._build_output(best_node)
                return {
                    "output": output,
                    "ispass": True,
                    "score": None,
                    "reward": best_node.reward,
                    "em": best_node.em,
                    "best_node": best_node,
                }

        all_nodes_list = collect_all_nodes(root)
        all_nodes_list.extend(terminal_nodes)
        best_child = max(all_nodes_list, key=lambda x: (x.reward, x.value)) if all_nodes_list else root
        output = self._build_output(best_child)
        return {
            "output": output,
            "ispass": bool(best_child.reward == 1),
            "score": None,
            "reward": best_child.reward,
            "em": best_child.em,
            "best_node": best_child,
            "all_nodes": all_nodes,
        }

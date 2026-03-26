import asyncio
import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.tools import Tool

from src.agents.LATS.model_client import OpenAIChatClient
from src.agents.LATS.webshop.environment import WebshopEnv, evaluate_webshop_output
from src.agents.LATS.webshop.task import WebShopTask


@dataclass
class WebshopLATSConfig:
    iterations: int = 30
    n_generate_sample: int = 5
    max_depth: int = 15
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
        self.state = {"action": "", "observation": ""} if state is None else state
        self.parent = parent
        self.question = question
        self.children: List["Node"] = []
        self.visits = 0
        self.value = 0.0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0.0
        self.em = 0
        self.env_state = env_state
        self.info: Dict = {}

    def uct(self) -> float:
        if self.visits == 0 and self.value >= 0:
            return float("inf")
        if self.visits == 0 and self.value < 0:
            return self.value
        parent_visits = max(self.parent.visits, 1) if self.parent else 1
        return self.value / self.visits + np.sqrt(2 * np.log(parent_visits) / self.visits)

    def __str__(self) -> str:
        return (
            f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, "
            f"action={self.state['action']}, observation={self.state['observation']})"
        )


def node_trajectory_to_text(node_string: str) -> str:
    lines = node_string.split("\n")
    formatted_lines = []
    for line in lines:
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split(")")[0].strip()
        except Exception:
            continue
        if depth != 0:
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
    trajectory = [node.question]
    curr: Optional[Node] = node
    while curr:
        if curr.parent:
            if curr.state.get("action"):
                trajectory.append(f"Action: {curr.state['action']}")
            if curr.state.get("observation"):
                trajectory.append(f"Observation: {curr.state['observation']}\n")
        curr = curr.parent
    return "\n".join(trajectory)


def generate_prompt(node: Node) -> str:
    trajectory = []
    question = node.question
    curr: Optional[Node] = node
    while curr:
        segment = []
        if curr.state.get("action"):
            segment.append(f"Action: {curr.state['action']}")
        if curr.state.get("observation") and curr.depth != 0:
            segment.append(f"Observation: {curr.state['observation']}")
        trajectory.append("\n".join(segment))
        curr = curr.parent
    return question + "\n\n".join(reversed(trajectory))


def get_unique_trajectories(failed_trajectories: Sequence[Dict], num: int = 3) -> List[str]:
    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get("final_answer")
        if final_answer not in seen_final_answers:
            unique_trajectories.append(node_trajectory_to_text(traj["trajectory"]))
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories


class WebshopLATSRunner:
    def __init__(
        self,
        env: WebshopEnv,
        task: WebShopTask,
        client: OpenAIChatClient,
        config: WebshopLATSConfig,
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
            print(self._colorize_log(msg, "Webshop"))

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

    def _maybe_update_reflection_map(self, prompt_text: str) -> None:
        if (
            len(self.failed_trajectories) > len(self.reflection_map)
            and len(self.failed_trajectories) < 4
        ):
            self.reflection_map = self.task.generate_self_reflection(
                self.failed_trajectories, prompt_text
            )

    async def _aget_samples_core(
        self,
        prompt_text: str,
        y: str,
        n_generate_sample: int,
        reflection_snapshot: Sequence[Dict[str, str]],
    ) -> List[str]:
        prompt = self.task.cot_prompt_wrap(prompt_text, y, reflection_snapshot)

        messages = [{"role": "user", "content": prompt}]
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
            samples.append(y + result[0])
        return samples

    def _get_samples(
        self, prompt_text: str, y: str, n_generate_sample: int
    ) -> List[str]:
        self._maybe_update_reflection_map(prompt_text)
        reflection_snapshot = list(self.reflection_map)
        return asyncio.run(
            self._aget_samples_core(
                prompt_text=prompt_text,
                y=y,
                n_generate_sample=n_generate_sample,
                reflection_snapshot=reflection_snapshot,
            )
        )

    async def _aget_values_core(
        self,
        x: str,
        ys: Sequence[str],
        failed_trajectories_snapshot: Sequence[Dict],
        reflections_snapshot: Sequence[Dict[str, str]],
    ) -> List[float]:
        if not ys:
            return []
        prompts = [
            self.task.value_prompt_wrap(
                x=x,
                y=y,
                failed_trajectories=failed_trajectories_snapshot,
                reflections=reflections_snapshot,
            )
            for y in ys
        ]
        tasks = [
            asyncio.create_task(
                self.client.achat(
                    messages=[{"role": "user", "content": prompt}],
                    stop=None,
                )
            )
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        values: List[float] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._log(f"Drop failed evaluate task idx={i}: {result}")
                values.append(0.0)
                continue
            parsed = [self.task.value_outputs_unwrap(item) for item in result]
            parsed = [value for value in parsed if value >= 0]
            values.append(float(sum(parsed) / len(parsed)) if parsed else 0.0)
        return values

    def _get_values(self, x: str, ys: Sequence[str]) -> List[float]:
        failed_trajectories_snapshot = copy.deepcopy(self.failed_trajectories)
        reflections_snapshot = list(self.reflection_map)
        return asyncio.run(
            self._aget_values_core(
                x=x,
                ys=ys,
                failed_trajectories_snapshot=failed_trajectories_snapshot,
                reflections_snapshot=reflections_snapshot,
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
                (child for child in terminal_children if child.reward == 1.0), None
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
            while node.is_terminal and node.reward != 1.0:
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

    def _run_action_with_worker(
        self,
        node_env_state: Optional[Dict],
        parent_state: Dict[str, str],
        session_id: str,
        action_line: str,
    ) -> Dict[str, Any]:
        worker_env = WebshopEnv(webshop_url=self.env.webshop_url)
        worker_env.restore_state(copy.deepcopy(node_env_state or {}))
        tool_name = self._tool_name_from_action(action_line)
        tool_input = ""
        if "[" in action_line and action_line.endswith("]"):
            tool_input = action_line.split("[", 1)[1][:-1]

        def run_search(text: str) -> Dict[str, Any]:
            obs_, reward_, done_, info_ = worker_env.step(session_id, f"search[{text}]")
            return {"obs": obs_, "reward": reward_, "done": done_, "info": info_}

        def run_click(text: str) -> Dict[str, Any]:
            obs_, reward_, done_, info_ = worker_env.step(session_id, f"click[{text}]")
            return {"obs": obs_, "reward": reward_, "done": done_, "info": info_}

        def run_think(text: str) -> Dict[str, Any]:
            obs_, reward_, done_, info_ = worker_env.step(session_id, f"think[{text}]")
            return {"obs": obs_, "reward": reward_, "done": done_, "info": info_}

        def run_raw(_: str) -> Dict[str, Any]:
            obs_, reward_, done_, info_ = worker_env.step(session_id, action_line)
            return {"obs": obs_, "reward": reward_, "done": done_, "info": info_}

        tool_funcs = {
            "search": run_search,
            "click": run_click,
            "think": run_think,
        }
        runner = Tool(
            name=tool_name,
            description=f"LATS webshop tool call: {tool_name}",
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
        state["action"] = action_line
        state["observation"] = obs
        return {
            "state": state,
            "env_state": worker_env.clone_state(),
            "reward": float(reward),
            "done": bool(done),
            "info": info,
            "obs": obs,
            "action_line": action_line,
        }

    async def _agenerate_new_states(
        self,
        node: Node,
        session_id: str,
        n_generate_sample: int,
        reflection_snapshot: Sequence[Dict[str, str]],
    ) -> List[Node]:
        prompt = generate_prompt(node)
        sampled_actions = await self._aget_samples_core(
            prompt_text=prompt,
            y="\nAction: ",
            n_generate_sample=n_generate_sample,
            reflection_snapshot=reflection_snapshot,
        )

        candidates: List[str] = []
        seen_actions = set()
        for sampled_action in sampled_actions:
            action_line = self.task.extract_action_line(sampled_action)
            if not action_line:
                continue
            if action_line in seen_actions:
                continue
            seen_actions.add(action_line)
            candidates.append(action_line)

        tasks = [
            asyncio.create_task(
                asyncio.to_thread(
                    self._run_action_with_worker,
                    node.env_state,
                    node.state,
                    session_id,
                    action_line,
                )
            )
            for action_line in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_nodes: List[Node] = []
        added = False
        for i, (action_line, result) in enumerate(zip(candidates, results)):
            if isinstance(result, Exception):
                self._log(f"Drop failed tool task idx={i}, action={action_line}: {result}")
                continue
            child = Node(
                state=result["state"],
                question=node.question,
                parent=node,
                env_state=result["env_state"],
            )
            child.reward = result["reward"]
            child.value = result["reward"]
            child.is_terminal = bool(result["done"] or result["reward"] > 0.0)
            child.info = result["info"]
            new_nodes.append(child)
            self._log(
                f"Child {i + 1}/{len(candidates)} depth={child.depth} | Action: {action_line} | "
                f"Reward: {child.reward} | "
                f"Terminal: {child.is_terminal}\nObservation: {result['obs']}"
            )

            if child.is_terminal and 0.0 < child.reward < 1.0 and not added:
                trajectory = collect_trajectory(child)
                existing_rewards = [traj["r"] for traj in self.failed_trajectories]
                if child.reward not in existing_rewards:
                    added = True
                    self.failed_trajectories.append(
                        {"trajectory": trajectory, "final_answer": action_line, "r": child.reward}
                    )

        return new_nodes

    def _generate_new_states(
        self, node: Node, session_id: str, n_generate_sample: int
    ) -> List[Node]:
        prompt = generate_prompt(node)
        self._maybe_update_reflection_map(prompt)
        reflection_snapshot = list(self.reflection_map)
        return asyncio.run(
            self._agenerate_new_states(
                node=node,
                session_id=session_id,
                n_generate_sample=n_generate_sample,
                reflection_snapshot=reflection_snapshot,
            )
        )

    def _expand_node(self, node: Node, session_id: str) -> None:
        n = self.config.n_generate_sample
        if node.depth >= self.config.max_depth:
            node.is_terminal = True
            return
        if node.depth == 0:
            n *= 2
        self._log(f"Expand node depth={node.depth}, n={n}, session={session_id}")
        new_nodes = self._generate_new_states(node, session_id=session_id, n_generate_sample=n)
        node.children.extend(new_nodes)
        self._log(f"Expanded -> children added: {len(new_nodes)}")

    def _evaluate_node(self, node: Node) -> float:
        child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]
        votes = self._get_values(node.question, child_prompts)
        votes = votes + [0.0] * (len(node.children) - len(votes))
        max_vote = max(votes) if votes else 1.0
        if max_vote == 0:
            max_vote = 1.0
        terminal_conditions = [1 if child.is_terminal else 0 for child in node.children]
        for i, condition in enumerate(terminal_conditions):
            if condition == 1:
                votes[i] = max_vote + 1
        for i, child in enumerate(node.children):
            child.value = votes[i] / max_vote
        self._log(f"Evaluate depth={node.depth}, normalized_votes={[child.value for child in node.children]}")
        return float(sum(votes) / len(votes)) if votes else 0.0

    def _rollout(self, node: Node, session_id: str, max_depth: int = 15) -> Node:
        depth = 0
        self._log(f"Rollout start depth={node.depth}, max_depth={max_depth}, session={session_id}")
        while not node.is_terminal and depth < max_depth:
            new_states: List[Node] = []
            values: List[float] = []
            attempts = 0
            while len(new_states) == 0 and attempts < 2:
                new_states = self._generate_new_states(
                    node, session_id=session_id, n_generate_sample=5
                )
                attempts += 1
            if not new_states:
                self._log(
                    f"Rollout expansion produced no children at depth={node.depth}; stop rollout"
                )
                break
            for state in new_states:
                if state.is_terminal:
                    return state

            child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal]
            while len(values) == 0:
                values = self._get_values(node.question, child_prompts)
                if not child_prompts:
                    break
            if not values:
                break
            max_value_index = values.index(max(values))
            node = new_states[max_value_index]
            self._log(
                f"Rollout step -> picked child depth={node.depth}, value={max(values):.3f}, "
                f"reward={node.reward}, terminal={node.is_terminal}"
            )
            depth += 1
            if depth == max_depth:
                node.reward = -0.5
        return node

    def _backpropagate(self, node: Node, value: float) -> None:
        curr: Optional[Node] = node
        while curr:
            curr.visits += 1
            curr.value = (curr.value * (curr.visits - 1) + value) / curr.visits
            curr = curr.parent

    def _build_output(self, node: Node) -> str:
        observation = node.state.get("observation", "")
        if "Your score (min 0.0, max 1.0): " in observation:
            return observation
        if node.reward > 0:
            return f"Your score (min 0.0, max 1.0): {node.reward}"
        return observation

    def run(self, session_id: str) -> Dict:
        question = self.env.reset(session_id)
        root = Node(state=None, question=question)
        root.env_state = self.env.clone_state()
        self.failed_trajectories = []
        self.reflection_map = []
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
                ispass, score = evaluate_webshop_output(output)
                return {
                    "output": output,
                    "ispass": ispass,
                    "score": score,
                    "reward": node.reward,
                    "best_node": node,
                }

            self._expand_node(node, session_id=session_id)
            while node.is_terminal:
                node = self._select_node(root)
                if node is None:
                    break
                self._expand_node(node, session_id=session_id)
            if node is None:
                break
            if not node.children:
                self._log("Expand produced no children; skip iteration")
                self._backpropagate(node, -1.0)
                continue

            self._evaluate_node(node)
            terminal_node = self._rollout(
                max(node.children, key=lambda child: child.value),
                session_id=session_id,
                max_depth=self.config.max_depth,
            )
            terminal_nodes.append(terminal_node)

            if terminal_node.reward == 1:
                output = self._build_output(terminal_node)
                ispass, score = evaluate_webshop_output(output)
                return {
                    "output": output,
                    "ispass": ispass,
                    "score": score,
                    "reward": terminal_node.reward,
                    "best_node": terminal_node,
                }
            self._backpropagate(terminal_node, terminal_node.reward)

            all_nodes = [(candidate, candidate.reward) for candidate in collect_all_nodes(root)]
            terminal_nodes_with_reward_1 = [
                candidate for candidate, reward in all_nodes if candidate.is_terminal and reward == 1
            ]
            if terminal_nodes_with_reward_1:
                best_node = max(terminal_nodes_with_reward_1, key=lambda item: item.reward)
                output = self._build_output(best_node)
                ispass, score = evaluate_webshop_output(output)
                return {
                    "output": output,
                    "ispass": ispass,
                    "score": score,
                    "reward": best_node.reward,
                    "best_node": best_node,
                }

        all_nodes_list = collect_all_nodes(root)
        all_nodes_list.extend(terminal_nodes)
        best_child = (
            max(all_nodes_list, key=lambda x: (x.reward, x.value))
            if all_nodes_list
            else root
        )
        output = self._build_output(best_child)
        ispass, score = evaluate_webshop_output(output)
        return {
            "output": output,
            "ispass": ispass,
            "score": score,
            "reward": best_child.reward,
            "best_node": best_child,
        }

import asyncio
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast

from colorama import Fore, Style

from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.LLMCompiler.chain import Chain
from src.agents.LLMCompiler.constants import END_OF_RESPONSE, JOINNER_REPLAN, JOINNER_FINISH
from src.agents.LLMCompiler.planner import Planner
from src.agents.LLMCompiler.task_fetching_unit import Task, TaskFetchingUnit
from src.agents.LLMCompiler.tools.base import StructuredTool, Tool
from langsmith import traceable


class LLMCompilerAgent:
    """Self defined agent for LLM Compiler."""

    def __init__(self, llm) -> None:
        self.llm = llm
        
    async def arun(self, messages, callbacks=None) -> str:
        self.llm.callbacks = callbacks
        response = None
        async for chunk in self.llm.astream(messages, stop=[END_OF_RESPONSE],):
            if not response:
                response = chunk
            else:
                response += chunk
        return response.content

class LLMCompiler(Chain, extra="allow"):
    """LLMCompiler Engine."""

    """The step container to use."""
    name: str = "LLMCompiler"
    input_key: str = "input"
    output_key: str = "output"

    def __init__(
        self,
        tools: Sequence[Union[Tool, StructuredTool]],
        max_replans: int,
        max_chat_history: int,
        planner_llm: BaseChatModel,
        planner_example_prompt: str,
        planner_example_prompt_replan: Optional[str],
        planner_stop: Optional[list[str]],
        planner_stream: bool,
        agent_llm: BaseChatModel,
        joinner_prompt: str,
        joinner_prompt_final: Optional[str],
        end_condition=None,
        print_log: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            tools: List of tools to use.
            max_replans: Maximum number of replans to do.
            max_chat_history: Maximum number of previous chat history to use.
            end_condition: A function that takes in the current agent scratchpad and returns
                a tuple of (end: bool, answer: str, reward: float). If end is True, the agent will stop replanning and return the answer. 
                (for webshop workload)
            
        Planner Args:
            planner_llm: LLM to use for planning.
            planner_example_prompt: Example prompt for planning.
            planner_example_prompt_replan: Example prompt for replanning.
                Assign this if you want to use different example prompt for replanning.
                If not assigned, default to `planner_example_prompt`.
            planner_stop: Stop tokens for planning.
            planner_stream: Whether to stream the planning.

        Agent Args:
            agent_llm: LLM to use for agent.
            joinner_prompt: Prompt to use for joinner.
            joinner_prompt_final: Prompt to use for joinner at the final replanning iter.
                If not assigned, default to `joinner_prompt`.
            
        """
        super().__init__(**kwargs)
        self.print_log = bool(print_log)

        if not planner_example_prompt_replan:
            if self.print_log:
                print(
                    "Replan example prompt not specified, using the same prompt as the planner."
                )
            planner_example_prompt_replan = planner_example_prompt

        self.planner = Planner(
            llm=planner_llm,
            example_prompt=planner_example_prompt,
            example_prompt_replan=planner_example_prompt_replan,
            tools=tools,
            stop=planner_stop,
            print_log=self.print_log,
        )

        self.agent = LLMCompilerAgent(agent_llm)
        self.joinner_prompt = joinner_prompt
        self.joinner_prompt_final = joinner_prompt_final or joinner_prompt
        self.planner_stream = planner_stream
        self.max_replans = max_replans
        self.end_condition = end_condition
        self.max_chat_history = max_chat_history
        self.current_iter = 0

    def _log_execution(self, message: str) -> None:
        if not self.print_log or not message:
            return
        print(
            Fore.CYAN
            + Style.BRIGHT
            + "Executed task:"
            + Style.RESET_ALL
            + f"\n{message}"
        )
        print("-" * 40)

    # def reset_all_stats(self):
    #     if self.planner_callback:
    #         self.planner_callback.reset()
    #     if self.executor_callback:
    #         self.executor_callback.reset()

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    @traceable
    def _parse_joinner_output(self, raw_answer: str):
        """We expect the joinner output format to be:
        ```
        Thought: xxx
        Action: Finish/Replan(yyy)
        ```
        Returns:
            thought (xxx)
            answer (yyy)
            is_replan (True/False)
        """
        thought, answer, is_replan = "", "", False  # default values
        raw_answers = raw_answer.split("\n")
        for ans in raw_answers:
            ans = ans.strip()
            if ans.startswith("Action:") or ans.startswith(JOINNER_REPLAN) or ans.startswith(JOINNER_FINISH):
                answer = ans[ans.find("(") + 1 : ans.find(")")]
                is_replan = JOINNER_REPLAN in ans
            elif ans.startswith("Thought:"):
                thought = ans.split("Thought:")[1].strip()
        return thought, answer, is_replan

    @traceable
    def _generate_context_for_replanner(
        self, tasks: Mapping[int, Task], joinner_thought: str
    ) -> str:
        """Formatted like this:
        ```
        1. action 1
        Observation: xxx
        2. action 2
        Observation: yyy
        ...
        Thought: joinner_thought
        ```
        """
        previous_plan_and_observations = "\n".join(
            [
                task.get_though_action_observation(
                    include_action=True, include_action_idx=True
                )
                for task in tasks.values()
                if not task.is_join
            ]
        )
        joinner_thought = f"Thought: {joinner_thought}"
        context = "\n\n".join([previous_plan_and_observations, joinner_thought])
        return context

    @traceable
    def _format_contexts(self, contexts: Sequence[str]) -> List[Any]:
        """contexts is a list of context
        each context is formatted as the description of _generate_context_for_replanner
        """
        formatted_contexts = []
        for i, context in enumerate(contexts):
            formatted_contexts.append(("system", f"Previous Plan {i}:\n\n{context}\n\n"))
        return formatted_contexts

    @traceable
    async def join(
        self, input_query: str, agent_scratchpad: List[str], is_final: bool
    ):
        if is_final:
            joinner_prompt = self.joinner_prompt_final
        else:
            joinner_prompt = self.joinner_prompt
            

            
        messages = [("system", joinner_prompt), ("human", input_query)]+ [("assistant", scratchpad.strip()) for scratchpad in agent_scratchpad]
        response = await self.agent.arun(messages)
        raw_answer = cast(str, response)
        if self.print_log:
            print(Fore.CYAN+Style.BRIGHT+f"Joinner response ({self.current_iter+1}/{self.max_replans}): \n"+Style.RESET_ALL, response)
            print("-"*40)
        end = False
        if self.end_condition:
            end, answer, reward = self.end_condition('\n'.join(agent_scratchpad))
            if end:
                return answer, answer, False
        thought, answer, is_replan = self._parse_joinner_output(raw_answer)
        if is_final:
            # If final, we don't need to replan
            is_replan = False
        return thought, answer, is_replan

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        raise NotImplementedError("LLMCompiler is async only.")

    @traceable
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        contexts = []
        joinner_thought = ""
        agent_scratchpad_list = []
        for i in range(self.max_replans):
            is_first_iter = i == 0
            is_final_iter = i == self.max_replans - 1
            self.current_iter = i
            try:
                task_fetching_unit = TaskFetchingUnit(
                    log_fn=self._log_execution if self.print_log else None
                )
                if self.planner_stream:
                    task_queue = asyncio.Queue()
                    asyncio.create_task(
                        self.planner.aplan(
                            inputs=inputs,
                            task_queue=task_queue,
                            is_replan=not is_first_iter,
                        )
                    )
                    await task_fetching_unit.aschedule(
                        task_queue=task_queue, func=lambda x: None
                    )
                else:
                    tasks = await self.planner.plan(
                        inputs=inputs,
                        is_replan=not is_first_iter,
                        callbacks=(
                            [self.planner_callback] if self.planner_callback else None
                        ),
                    )
                    task_fetching_unit.set_tasks(tasks)
                    await task_fetching_unit.schedule()
                tasks = task_fetching_unit.tasks
            except TypeError as e:  
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)

            agent_scratchpad_list.append(
                "".join(
                    [
                        task.get_though_action_observation(
                            include_action=True, include_thought=True
                        )
                        for task in tasks.values()
                        if not task.is_join
                    ]
                )
            )
            agent_scratchpad = agent_scratchpad_list[-self.max_chat_history:]
            joinner_thought, answer, is_replan = await self.join(
                inputs["input"],
                agent_scratchpad=agent_scratchpad,
                is_final=is_final_iter,
            )
            if not is_replan:
                break

            # Collect contexts for the subsequent replanner
            context = self._generate_context_for_replanner(
                tasks=tasks, joinner_thought=joinner_thought+"\n"+answer
            )
            contexts.append(context)
            formatted_contexts = self._format_contexts(contexts[-self.max_chat_history:])
            inputs["context"] = formatted_contexts

        if is_final_iter and self.print_log:
            print(Fore.RED+"Reached max replan limit."+Style.RESET_ALL)

        return {self.output_key: answer}

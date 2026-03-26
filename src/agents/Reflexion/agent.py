from src.agents.Reflexion.prompt import REFLECTION_HEADER
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langsmith import traceable
import re
from colorama import Fore, Style

def parse_action(input_str, print_log: bool = False):
    pattern = re.compile(r"Action\s+(\d+):\s*(\w+)\s*([\[\(])", re.MULTILINE)
    match = pattern.search(input_str)
    if not match:
        if print_log:
            print("Parse failed!!")
        return None, None

    action_type = match.group(2)
    open_bracket = match.group(3)
    close_bracket = ']' if open_bracket == '[' else ')'
    start_idx = match.end()
    argument = input_str[start_idx:].strip().strip(close_bracket)
    
    return action_type, argument

def parse_thought_action(string):
    t = re.search(r"Thought.*?:.*", string)
    if not t:
        return None, None
    thought = t.group(0).strip()
    a = re.search(r"Action.*?:", string[t.end():])
    if not a:
        return thought, None
    action_start = t.end() + a.start()
    content_start = t.end() + a.end()
    end = re.search(r"Action|Thought|Observation|\Z", string[content_start:])
    action_end = content_start + end.start() if end else len(string)
    action = string[action_start:action_end].strip()
    return thought, action


def parse_reflection(text) -> str:
    m = re.search(r"Reflection:.*", text, re.S)
    if not m:
        return ""
    return m.group(0).strip()


class ReflexionAgent:
    def __init__(self,
                 actor_llm,
                 actor_prompt,
                 actor_examples,
                 reflect_llm,
                 reflect_prompt,
                 reflect_examples,
                 tools,
                 workload: str = "hotpotqa",
                 context_limit: int = 2000,
                 max_steps: int = 6,
                 evaluator = lambda ans, key: ans == key,
                 print_log: bool = False,
                 ) -> None:
        
        self.actor_llm = actor_llm
        self.actor_llm.stop = "\nObservation"
        self.actor_prompt = actor_prompt
        self.actor_examples = actor_examples
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = reflect_examples
        self.reflections = []
        self.scratchpad_list = []
        self.last_scratchpad = []

        self.tools = tools 
        self.workload = workload
        self.tools_dict = {}
        self.set_tools()
        self.max_steps = max_steps
        self.context_limit = context_limit
        self.evaluator = evaluator
        self.print_log = bool(print_log)
                
        self.step_n = 0
        self.finished = False

    def _log(self, text: str) -> None:
        if self.print_log:
            print(text)

    def __reset_agent(self):
        self.scratchpad_list = []
        self.reflections = []
        self.last_scratchpad = []
        self.step_n = 0
        self.finished = False
        return
    
    def __new_trial(self):
        self.scratchpad_list = []
        # leave reflections as is
        self.last_scratchpad = self.scratchpad_list
        self.step_n = 0
        self.finished = False
        return
        
    @traceable
    def _build_agent_prompt(self):
        messages = [SystemMessage(role="system", content=self.actor_prompt.format(examples=self.actor_examples))]
        messages += [HumanMessage(role="user", content=f"Question: {self.query}")]
        if self.reflections:
            messages += [HumanMessage(role="user", content=f"[Previous reflections]: ")]
            messages += [AIMessage(role="assistant", content=msg) for msg in self.reflections]
        if self.last_scratchpad:
            messages += [HumanMessage(role="user", content=f"[Last trial]:")]
            messages += [HumanMessage(role="user", content=msg) if msg[0] == "O" else AIMessage(role="assistant", content=msg) for msg in self.last_scratchpad]
        if self.scratchpad_list:
            messages += [HumanMessage(role="user", content=f"[Current trial]:")]
            messages += [HumanMessage(role="user", content=msg) if msg[0] == "O" else AIMessage(role="assistant", content=msg) for msg in self.scratchpad_list]
        messages += [HumanMessage(role="user", content=f"Answer with\nThought {self.step_n}:\nAction {self.step_n}:")]
        return messages
    
    @traceable
    def _build_reflection_prompt(self):
        self.truncate_scratchpad()
        messages = [SystemMessage(role="system", content=self.reflect_prompt.format(examples=self.reflect_examples))] 
        messages += [HumanMessage(role="user", content=f"[question]: {self.query}",)] 
        if self.reflections:
            messages += [HumanMessage(role="user", content=f"[Previous reflections]: ",)]
            messages += [HumanMessage(role="user", content=msg) for msg in self.reflections]
        if self.scratchpad_list:
            messages += [HumanMessage(role="user", content=f"[Current trial]: ")]
            messages += [HumanMessage(role="user", content=msg) if msg[0] == "O" else AIMessage(role="assistant", content=msg) for msg in self.scratchpad_list]
        messages += [HumanMessage(role="user", content="Answer with\nReflection: ")]
        return messages
    
    def set_qa(self, query: str) -> None:
        self.query = query
        self.__reset_agent()
        self.set_tools()
        return
    
    def set_tools(self) -> None:
        self.tools_dict = {}
        for tool in self.tools:
            self.tools_dict[tool.name] = tool
            self.tools_dict[tool.name.lower()] = tool
            if self.workload == "math":
                if tool.name == "WolframAlpha":
                    self.tools_dict["search"] = tool
                if tool.name == "simplecalc":
                    self.tools_dict["calculator"] = tool

    @traceable
    def truncate_scratchpad(self):
        observations = filter(lambda x: x.startswith('observation'), self.scratchpad_list)
        observations_by_tokens = sorted(observations, key=lambda x: len(x.split()))
        while sum(len(x.split()) for x in observations) > self.context_limit:
            largest_observation = observations_by_tokens.pop(-1)
            ind = self.scratchpad_list.index(largest_observation)
            self.scratchpad_list[ind] = largest_observation.split(':')[0] + ': [truncated excerpt]'
    
    @traceable("tool")
    def run_tool(self, tool, argument):
        assert isinstance(tool, BaseTool)
        return tool._run(argument)
    
    @traceable
    def call_actor(self) -> str:
        result = ''
        prompt = self._build_agent_prompt()
        for chunk in self.actor_llm.stream(prompt):
            result += chunk.content
        response = result.strip()
        return response

    @traceable
    def run(self) -> str:
        if self.is_finished() or self.is_halted(): # After first trial, reflection is needed.
            self.reflect()
            self.__new_trial()
        output = self._run() # Run ReAct iterations
        return output

    @traceable
    def _run(self) -> str:
        output = ""
        while not self.is_halted() and not self.is_finished():
            output = self.step()
        return output
    
    @traceable
    def step(self) -> str:
        self.step_n += 1
        thought_and_action = self.call_actor()
        
        try:
            thought, action = parse_thought_action(thought_and_action)
            action_type, argument = parse_action(action, print_log=self.print_log)
            self.scratchpad_list.append(f"{thought}\n{action}")
            self._log(f"{thought}\n{action}")
        except:
            self.scratchpad_list.append(f"{thought_and_action}\nWrong Action format. Follow the instruction about 'Action' carefully.")
            return "Agent result is not successfull."
        observation = f'Observation {self.step_n}: '
        try:
            tool_key = action_type if action_type in self.tools_dict else action_type.lower()
            tool_output = self.run_tool(self.tools_dict[tool_key], argument)
            if type(tool_output) is tuple:
                (tool_output, artifact) = tool_output
            else:
                artifact = None
        except Exception as e:
            self._log(f"Error: {e}")
            self._log(f"Tool {action_type} failed.")
            lowered_action = (action_type or "").lower()
            if lowered_action in {"search", "wolframalpha"}:
                observation += "WolframAlpha failed. Please try again."
            elif action_type == "click":
                observation += "You clicked an invalid object"
            elif lowered_action == "lookup":
                observation += f"The last page searched was not found, so you cannot lookup a keyword in it. Please try one of the similar pages given."
            elif lowered_action in {"simplecalc", "calculator"}:
                observation += "simplecalc failed. Please try again."
            self._log(observation)
            self._log(Fore.CYAN+Style.BRIGHT+f"{'-'*30}"+Style.RESET_ALL)
            self.scratchpad_list.append(observation)
            return "Agent result is not successfull."
        if action_type != "finish":
            if artifact and "done" in artifact and artifact["done"]: # For Webshop click[Buy Now]
                observation += tool_output
                self._log(observation)
                self._log(Fore.CYAN+Style.BRIGHT+f"{'-'*30}"+Style.RESET_ALL)
                self.scratchpad_list.append(observation)
                self.finished = True
                return observation
            observation += tool_output
            answer = "Agent result is not successfull."
        elif action_type == "finish":
            answer = tool_output.replace("Answer: ", "")
            observation += answer
            self.finished = True
        else:
            observation += f"Invalid action type. Follow the tool description carefully."
            answer = "Agent result is not successfull."
            
        self.scratchpad_list.append(observation)
        self._log(observation)
        self._log(Fore.CYAN+Style.BRIGHT+f"{'-'*30}"+Style.RESET_ALL)
        return answer

    def is_finished(self) -> bool:
        return self.finished
    
    def is_halted(self) -> bool:
        return self.step_n > self.max_steps and not self.finished


    @traceable()
    def reflect(self):
        self.truncate_scratchpad()
        result = ""
        for chunk in self.reflect_llm.stream(self._build_reflection_prompt()):
            result += chunk.content
        reflection = parse_reflection(result)
        if not self.reflections:
            reflection = REFLECTION_HEADER + reflection
        else:
            reflection = reflection
        self.reflections.append(reflection)
        self._log("\n".join(self.reflections))
        self._log(Fore.CYAN+Style.BRIGHT+f"{'-'*30}"+Style.RESET_ALL)

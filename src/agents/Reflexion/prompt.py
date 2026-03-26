from langchain_core.prompts import PromptTemplate

# HotpotQA
HOTPOTQA_PROMPT = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) search(keyword), which searches the keyword on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search. Search keyword must be a simple and concise.
(2) lookup(keyword), which returns the next sentence containing keyword in the last passage successfully found by search.
(3) finish(answer), which returns the answer and finishes the task.
You may take as many steps as necessary
IMPORTANT:
- Plan your Thought and only ONE Action AT ONCE
- For yes or no questions, answer with "finish(yes)" or "finish(no)". 

Here are some examples:
{examples}
(END OF EXAMPLES)"""

# Math
MATH_PROMPT = """Solve a problem answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) WolframAlpha(query), Performs a precise and difficult mathematical calculation or query using WolframAlpha. The query must be complete, mathematically precise, and written in correct syntax (e.g., LaTeX for mathematical expressions or symbolic representations). WolframAlpha will return either an exact result or related entities if the query cannot be resolved.
    Examples:
        WolframAlpha(Factor x**2-3x+2)
        WolframAlpha(Solve x**3+x**2-3x-1=0)
(2) simplecalc(expression), Performs a simple mathematical calculation using Python's numexpr library. Expression should be a single line mathematical expression that solves the problem.
    Examples:
        simplecalc(37593 * 67) for "37593 times 67"
        simplecalc(37593**(1/5))  for "37593^(1/5)" (Always Use '**' for exponentiation instead of '^')
(3) finish(answer), which returns the answer and finishes the task.
You may take as many steps as necessary and solve the math problem step by step
IMPORTANT:
- DO NOT solve the problem by sending direct query to WolframAlpha, without breaking down the problem into manageable mathematical steps.
- DO NOT think too much for one Thought. Think step by step.
- Always be mathematically precise when constructing the query for WolframAlpha and the expression for simple calculation.
- Include all relevant details in the query (e.g., formulas, known values, and specific terms) to avoid ambiguity.
- expression for simple calculator should be like "2 + 4, sqrt(2), sin(pi/2), 3*13.5", **without any variables**
- Use LaTeX or formal mathematical notation whenever possible to improve result accuracy.
- Always follow the format "WolframAlpha(<query>)" or "simplecalc(<expression>) or "finish(<answer>)" during "Action #:" step.
- Never use "search(...)" for math; use "WolframAlpha(...)" instead.
- Answer with "finish(<value>)" where <value> is the answer to the question.
- Never forget to use "\\" to escape special characters properly before submitting the query to WolframAlpha. 
- Plan your Thought and plan Action after your Thought

Here are some examples:
{examples}
(END OF EXAMPLES)"""

# Webshop
WEBSHOP_PROMPT = """Solve a shopping task with interleaving Thought, Action, Observation steps.
You will be given a shopping instruction and the current webshop observation.

At each turn, respond with exactly one Thought line and exactly one Action line.
Use the step number provided by the user.

Required format:
Thought <step>: <brief reasoning>
Action <step>: <tool call>

Available actions:
(1) search(query), which searches for a product query in the webshop environment.
    You may use this only when the observation explicitly shows [Search].
(2) click(button), which clicks a visible button, item id, option, or navigation control.
    You may only click text that appears inside brackets in the current observation.

IMPORTANT:
- Plan only the next move for the current state.
- Use only ONE action at a time.
- Do NOT output numbered plans, bullet lists, multiple actions, or Observation lines.
- Thought should be concise, about 1-2 sentences.
- You MUST NEVER say that you do not know the answer.
- Do NOT use finish() for webshop. The task ends only after clicking Buy Now on the correct product page.

Navigation rules:
- After clicking Description, Features, Reviews, or Attributes, click(< Prev) to return to the item page.
- If you want to inspect a different product after opening an item page, click(< Prev) first to return to the search results.
- If an invalid action or error occurs, recover using the current page. Usually click(Back to Search) or click(< Prev) is the right next step.
- Click all relevant product options before click(Buy Now).

End condition:
- The phrase "Your score (min 0.0, max 1.0): ..." appears only after click(Buy Now).
- Do not stop early before clicking Buy Now.

Here are some example trajectories:
{examples}
(END OF EXAMPLES)

Common reflections to keep in mind:
- click ALL relevant option buttons before click(Buy Now)
- when "Error: Wrong tool use." happens, interact with buttons visible on the current page
"""

HUMANEVAL_PROMPT = """You are an AI that only responds with Thought and Action. 
You will be given a function signature and its docstring by the user. 
First, write your thought on how to answer the given programming query after 'Thought "step": ' within one sentence
Then, write your full python 'function implementation' (restate the function signature) only in the bracket of 'Action "step": execute(), NOT English' 
When the Observation of 'execute()' is 'True', finish with the tool 'Action: finish()'

**Use a Python code block to write your function implementation.**
For example:\n```python\nprint('Hello world!')\n```

### Tools Available:
- **execute('function implementation')**: Executes 'function implementation' you wrote to validate its functionality.
Examples:
    Action #: execute(```python\nfrom typing import List
    def sum_elements(numbers: List[float]) -> float:
        sum = 0
        for i in range(len(numbers)):
            sum += numbers[i]
        return sum\n```
    )
- **finish('function implmentation')**: use finish tool to finish the task with your function implementation if your implementation in 'execute()' succeeded to be 'Test passed'.

Here is the example
{examples}
(END OF EXAMPLES)
"""

## For Generating reflections
REFLECTION_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'

LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n'

REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish(<answer>), or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
"""

REFLECT_INSTRUCTION_WEBSHOP = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Webshop API environment and an instruction for a product to follow to buy it correctly. You were unsuccessful in buying the product correctly either because you missed some criteria of the instruction, or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)"""

REFLECT_INSTRUCTION_HUMANEVAL = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to a programming environment and a programming task to complete. You were unsuccessful in completing the task either because your implementation failed the test cases when executed, or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)"""

### Actor prompt
hotpotqa_action_prompt = PromptTemplate(
                        input_variables=["examples"],
                        template = HOTPOTQA_PROMPT,
                        )


math_action_prompt = PromptTemplate(
                        input_variables=["examples"],
                        template = MATH_PROMPT,
                        )

webshop_action_prompt = PromptTemplate(
                        input_variables=["examples"],
                        template = WEBSHOP_PROMPT,
                        )

humaneval_action_prompt = PromptTemplate(
                        input_variables=["examples"],
                        template = HUMANEVAL_PROMPT,
                        )

### Reflection prompt 
hotpotqa_reflection_prompt = PromptTemplate(
                        input_variables=["examples"],
                        template = REFLECT_INSTRUCTION,
                        )

math_reflection_prompt = PromptTemplate(
                        input_variables=["examples"],
                        template = REFLECT_INSTRUCTION,
                        )

webshop_reflection_prompt = PromptTemplate(
                        input_variables=["examples"],
                        template = REFLECT_INSTRUCTION_WEBSHOP,
                        )

humaneval_reflection_prompt = PromptTemplate(
                        input_variables=["examples"],
                        template = REFLECT_INSTRUCTION_HUMANEVAL,
                        )

def get_action_prompt(workload: str) -> PromptTemplate:
    if workload == "hotpotqa":
        return hotpotqa_action_prompt
    elif workload == "math":
        return math_action_prompt
    elif workload == "webshop":
        return webshop_action_prompt
    elif workload == "humaneval":
        return humaneval_action_prompt
    else:
        raise ValueError(f"Unknown workload: {workload}")
    
def get_reflection_prompt(workload: str) -> PromptTemplate:
    if workload == "hotpotqa":
        return hotpotqa_reflection_prompt
    elif workload == "math":
        return math_reflection_prompt
    elif workload == "webshop":
        return webshop_reflection_prompt
    elif workload == "humaneval":
        return humaneval_reflection_prompt
    else:
        raise ValueError(f"Unknown workload: {workload}")

import re
from typing import List, Sequence, Tuple

standard_prompt = """Solve the given math problem with interleaving Thought, Action, Observation.
Use exactly one action at a time.

Available actions:
1. WolframAlpha[query]
2. simplecalc[expression]
3. finish[answer]

Rules:
- Use `WolframAlpha[...]` for symbolic or variable-based math queries such as factoring,
  solving equations, simplifying expressions, sequence formulas, and calculus operations.
- Use `simplecalc[...]` for straightforward numeric expressions.
- Use `finish[...]` only when you are confident in the final answer.
- Never use `search[...]` for math; use `WolframAlpha[...]` instead.
- Keep answers concise and use LaTeX-style expressions when appropriate.

{input}
"""

cot_prompt = """You are a math-solving assistant specialized in solving complex math problems using tools.
Your task is to answer with Thought and Action lines only.

Use exactly these action formats:
- Action N: WolframAlpha[...]
- Action N: simplecalc[...]
- Action N: finish[...]

Example:
Problem: Find the largest value of $c$ such that $\\frac{c^2 + 6c - 27}{c-3} + 2c = 23$.
Thought 1: I should factor the numerator first before reducing the rational expression.
Action 1: WolframAlpha[Factor c^2 + 6c - 27]
Observation 1: (c - 3)(c + 9)
Thought 2: The fraction simplifies to c+9, so I now solve c + 9 + 2c = 23 by simple arithmetic.
Action 2: simplecalc[(23 - 9)/3]
Observation 2: 4.666666666666667
Thought 3: Therefore c = 14/3.
Action 3: finish[\\frac{14}{3}]

Problem: Let f(x) = {{ ax+3 if x>2, x-5 if -2<=x<=2, 2x-b if x<-2 }}. Find a+b if f is continuous.
Thought 1: Continuity at x=2 gives 2a+3 = -3, so a = (-6)/2.
Action 1: simplecalc[(-6)/2]
Observation 1: -3.0
Thought 2: Continuity at x=-2 gives -7 = -4 - b, so b = 3.
Action 2: simplecalc[3]
Observation 2: 3
Thought 3: a+b = -3 + 3 = 0.
Action 3: finish[0]

{input}
"""

cot_prompt_short = cot_prompt

cot_prompt_feedback = """You are a math-solving assistant specialized in solving complex math problems using tools.
You also receive reflections from previous failed trajectories.
Use the reflections to avoid repeating errors.

Use exactly these action formats:
- Action N: WolframAlpha[...]
- Action N: simplecalc[...]
- Action N: finish[...]

Never use `search[...]` for math; use `WolframAlpha[...]` instead.

Previous failed trajectories and reflections:
{trajectories}

Now solve:
{input}
"""

cot_prompt_feedback_short = cot_prompt_feedback

reflection_sys_msg = """You are an advanced reasoning agent that can improve based on self reflection.
You will be given a previous failed math-solving trajectory.
Diagnose likely failure reasons and provide a concise high-level plan to avoid repeating them.
Use 2-4 complete sentences.
"""

reflection_usr_msg = """Previous trial:
{trajectory}

Reflection:
"""

value_prompt_reasoning = """You are evaluating a math-solving trajectory.
Score the trajectory from 1 to 10 where:
- 10 means the final answer is very likely correct and well-supported.
- 1 means the trajectory is very likely incorrect or incoherent.

Focus on mathematical correctness, consistency with observations, and whether the final answer follows from the steps.
At the final line, output exactly:
Thus the correctness score is <score>
"""

value_prompt_reasoning_feedback = """You are evaluating a math-solving trajectory.
You are given previous failed trajectories and reflections.
Use them to calibrate your score and penalize repeated mistakes.

Failed trajectories and reflections:
{trajectories}

At the final line, output exactly:
Thus the correctness score is <score>
"""

value_prompt_reasoning_feedback_short = value_prompt_reasoning_feedback


def _split_examples(
    prompt: str,
    entry_prefix: str,
    tail_marker: str,
) -> Tuple[str, List[str], str]:
    marker = "Example:"
    if marker not in prompt:
        return prompt, [], ""
    prefix, rest = prompt.split(marker, 1)
    tail_idx = rest.find(tail_marker)
    if tail_idx == -1:
        tail_idx = len(rest)
    examples_block = rest[:tail_idx].strip()
    suffix = rest[tail_idx:].lstrip()
    if not examples_block:
        return prefix, [], suffix
    split_pattern = rf"\n(?={re.escape(entry_prefix)})"
    examples = [
        part.strip()
        for part in re.split(split_pattern, examples_block)
        if part.strip().startswith(entry_prefix)
    ]
    return prefix, examples, suffix


def _render_prompt_with_examples(
    prefix: str,
    examples: Sequence[str],
    suffix: str,
    fewshot: int,
) -> str:
    selected = list(examples[: min(max(int(fewshot), 0), len(examples))])
    if selected:
        examples_text = "\n\n".join(selected) + "\n\n"
        return f"{prefix}Example:\n{examples_text}{suffix}"
    return f"{prefix}{suffix}"


_COT_PREFIX, MATH_COT_EXAMPLES, _COT_SUFFIX = _split_examples(
    cot_prompt, entry_prefix="Problem:", tail_marker="{input}"
)
_COT_SHORT_PREFIX, MATH_COT_SHORT_EXAMPLES, _COT_SHORT_SUFFIX = _split_examples(
    cot_prompt_short, entry_prefix="Problem:", tail_marker="{input}"
)


def build_math_cot_prompt(fewshot: int) -> str:
    return _render_prompt_with_examples(
        _COT_PREFIX,
        MATH_COT_EXAMPLES,
        _COT_SUFFIX,
        fewshot=fewshot,
    )


def build_math_cot_short_prompt(fewshot: int) -> str:
    return _render_prompt_with_examples(
        _COT_SHORT_PREFIX,
        MATH_COT_SHORT_EXAMPLES,
        _COT_SHORT_SUFFIX,
        fewshot=fewshot,
    )

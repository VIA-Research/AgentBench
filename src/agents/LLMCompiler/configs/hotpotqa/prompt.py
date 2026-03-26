from src.agents.LLMCompiler.constants import (
    END_OF_PLAN,
    END_OF_RESPONSE,
    JOINNER_FINISH,
    JOINNER_REPLAN,
)

PLANNER_PROMPT = f"""You are planning tool calls for a HotpotQA-style Wikipedia question answering task.
Create a short numbered plan that the executor can run.

### Tools Available:
- **search({{"text": "entity or topic"}})**:
  Search Wikipedia for an entity or topic and load that page into memory.
- **lookup({{"page": "searched page title", "keyword": "keyword to scan for"}}, ["$N"])**:
  Scan a previously searched page for the next passage containing `keyword`.
  The first argument is the actual tool input.
  The second argument is only a dependency list and must contain the search step(s) that must finish first.
  Example:
  1. search({{"text": "Pam Veasey"}})
  2. lookup({{"page": "Pam Veasey", "keyword": "American"}}, ["$1"])
- **join()**:
  Finalize the current plan so the next agent can answer or request a replan.

### Rules:
- Output only numbered actions. Do not add explanations, comments, or prose.
- Always search a page before looking it up.
- In `lookup`, `page` must be the literal page title string, not `$1` or another placeholder.
- Use the dependency list to enforce execution order for `lookup`.
- If the search result already directly states the needed fact, do not add an unnecessary `lookup`; go to `join()`.
- For nationality questions, prefer clues that actually appear in Wikipedia text, such as demonyms like `American`, `British`, or `French`, or use `born` / `birthplace` if needed.
- If one page is not enough, plan follow-up searches for the related person or entity, then call `join()`.
- Always end the plan with `join(){END_OF_PLAN}`.

Here are some example plans:
"""

PLANNER_FEWSHOT_LIST = [
    f"""Question: Are Pam Veasey and Jon Jost both American?
1. search({{"text": "Pam Veasey"}})
2. lookup({{"page": "Pam Veasey", "keyword": "American"}}, ["$1"])
3. search({{"text": "Jon Jost"}})
4. lookup({{"page": "Jon Jost", "keyword": "American"}}, ["$3"])
5. join(){END_OF_PLAN}
###
""",
    f"""Question: What is the profession of Bill Gates' mother and father?
1. search({{"text": "Bill Gates"}})
2. lookup({{"page": "Bill Gates", "keyword": "mother"}}, ["$1"])
3. lookup({{"page": "Bill Gates", "keyword": "father"}}, ["$1"])
4. search({{"text": "Mary Maxwell Gates"}})
5. lookup({{"page": "Mary Maxwell Gates", "keyword": "profession"}}, ["$4"])
6. search({{"text": "William H. Gates Sr."}})
7. lookup({{"page": "William H. Gates Sr.", "keyword": "attorney"}}, ["$6"])
8. join(){END_OF_PLAN}
###
""",
    f"""Question: What was the birthplace of the mother of Marie Curie?
1. search({{"text": "Marie Curie"}})
2. lookup({{"page": "Marie Curie", "keyword": "mother"}}, ["$1"])
3. search({{"text": "Bronisława Skłodowska"}})
4. lookup({{"page": "Bronisława Skłodowska", "keyword": "born"}}, ["$3"])
5. join(){END_OF_PLAN}
###
""",
    f"""Question: Were Scott Derrickson and Ed Wood of the same nationality?
1. search({{"text": "Scott Derrickson"}})
2. search({{"text": "Ed Wood"}})
3. join(){END_OF_PLAN}
###
""",
]

OUTPUT_PROMPT = (
    "You must solve the Question using only the given observations.\n"
    "Respond with exactly one Thought and one Action.\n"
    f"Action must be either {JOINNER_FINISH}(answer) or {JOINNER_REPLAN}(reflection).\n"
    f"End your response with {END_OF_RESPONSE}.\n"
    "Use short answers. Do not use outside knowledge.\n"
    "If the observations are insufficient, explain what information is missing and request a replan.\n"
    "\n"
    "Here are some examples:\n\n"
)

OUTPUT_FEWSHOT_LIST = [
    f"""Question: Which magazine was started first Arthur's Magazine or First for Women?
search(Arthur's Magazine)
Observation: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
search(First for Women)
Observation: First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989.
Thought: Arthur's Magazine started in 1844, while First for Women started in 1989. Arthur's Magazine started earlier.
Action: {JOINNER_FINISH}(Arthur's Magazine)
{END_OF_RESPONSE}
""",
    f"""Question: What is the birth date of the father of the founder of Microsoft?
search(Bill Gates)
Observation: Bill Gates (born October 28, 1955) is an American business magnate, software developer, and co-founder of Microsoft Corporation.
search(William H. Gates Sr.)
Observation: William H. Gates Sr. was an American attorney, philanthropist, and the father of Bill Gates.
lookup(William H. Gates Sr., born)
Observation: William H. Gates Sr. was born on November 30, 1925.
Thought: The father of the founder of Microsoft is William H. Gates Sr., and the observation gives his birth date directly.
Action: {JOINNER_FINISH}(November 30, 1925)
{END_OF_RESPONSE}
""",
    f"""Question: What was the birthplace of the mother of Marie Curie?
search(Marie Curie)
Observation: Marie Curie was the daughter of Władysław Skłodowski and Bronisława Skłodowska.
search(Bronisława Skłodowska)
Observation: Bronisława Skłodowska was born in Warsaw, Poland, and was a teacher and educator.
Thought: The second observation directly states Bronisława Skłodowska's birthplace.
Action: {JOINNER_FINISH}(Warsaw, Poland)
{END_OF_RESPONSE}
""",
    f"""Question: Were Scott Derrickson and Ed Wood of the same nationality?
search(Scott Derrickson)
Observation: Scott Derrickson is an American filmmaker.
search(Ed Wood)
Observation: Ed Wood was an American filmmaker, actor, writer, producer, and director.
Thought: Both observations identify Scott Derrickson and Ed Wood as American, so they share the same nationality.
Action: {JOINNER_FINISH}(yes)
{END_OF_RESPONSE}
""",
    f"""Question: Were Scott Derrickson and Ed Wood of the same nationality?
search(Scott Derrickson)
Observation: Scott Derrickson is an American filmmaker.
search(Ed Wood)
Observation: Ed Wood was a filmmaker and screenwriter.
Thought: Scott Derrickson's nationality is available, but Ed Wood's nationality is still missing from the observations.
Action: {JOINNER_REPLAN}(I need more evidence about Ed Wood's nationality, such as whether he was American or where he was born.)
{END_OF_RESPONSE}
""",
]


def get_planner_prompt(fewshot: int):
    if fewshot > len(PLANNER_FEWSHOT_LIST):
        fewshot = len(PLANNER_FEWSHOT_LIST)
        print(
            f"Max fewshot examples for hotpotqa planner prompt is {len(PLANNER_FEWSHOT_LIST)}. Running with {fewshot} fewshot examples."
        )
    return PLANNER_PROMPT + "\n".join(PLANNER_FEWSHOT_LIST[:fewshot])


def get_output_prompt(fewshot: int):
    if fewshot > len(OUTPUT_FEWSHOT_LIST):
        fewshot = len(OUTPUT_FEWSHOT_LIST)
        print(
            f"Max fewshot examples for hotpotqa output prompt is {len(OUTPUT_FEWSHOT_LIST)}. Running with {fewshot} fewshot examples."
        )
    return OUTPUT_PROMPT + "\n".join(OUTPUT_FEWSHOT_LIST[:fewshot])

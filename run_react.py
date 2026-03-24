import time
import numpy as np
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from colorama import Fore, Style
from src.agents.ReAct.react import create_react_agent
from src.utils import parse_answer

from dotenv import load_dotenv
from langsmith import traceable, trace

load_dotenv()

def main(args):
    if args.host:
        host_url = f"http://{args.host}:{args.port}/v1"
    else:
        host_url = None

    score_sum = 0
    pass_count = 0
    # Load dataset
    from src.utils import load_dataset, get_evaluation_function
    print(f"Loading dataset for workload: {args.workload}")
    dataset = load_dataset(args.workload)
    evaluator = get_evaluation_function(args.workload)
    samples = min(len(dataset), args.samples) if args.samples else len(dataset)
    latencies = []

    def pretty_output(i):
        print(Fore.YELLOW+"=" * 30)
        print(f"Sample {i + 1}/{samples}")
        if args.workload == "webshop":
            print(f"Average score so far: {round(score_sum / (i + 1), 2)}")
        print(f"Accuracy so far: {round(pass_count / (i + 1), 2)}")
        if latencies:
            print(f"Avg. latency: {round(sum(latencies) / len(latencies), 2)} sec")
            print(f"p50 latency: {round(np.percentile(latencies, 50), 2)} sec")
            print(f"p90 latency: {round(np.percentile(latencies, 90), 2)} sec")
            print(f"p95 latency: {round(np.percentile(latencies, 95), 2)} sec") 
            print(f"p99 latency: {round(np.percentile(latencies, 99), 2)} sec")
        print("=" * 30+Style.RESET_ALL)
        print("\n")

    # Load model
    model = ChatOpenAI(model=args.model, base_url=host_url, stream_usage=True, stop="\nObservation:", temperature=args.temperature)

    system_prompt = None
    count = 0
    pass_count = 0
    if args.workload == "hotpotqa":
        from src.tools.hotpotqa_tools.wikipedia import WikipediaTool, LookupTool, FinishTool
        from src.agents.ReAct.prompt.hotpotqa import get_system_prompt
        if args.fewshot > 5:
            print(f"Max fewshot examples for {args.workload} is 5. Running with 5 fewshot examples.")
        system_prompt = get_system_prompt(fewshots=min(args.fewshot, 5))
        search = WikipediaTool(name="search")
        lookup = LookupTool(name="lookup")
        finish = FinishTool(name="finish")
        tools = [search, lookup, finish]
        langgraph_agent_executor = create_react_agent(model, tools=tools)
        
        for i in range(samples):
            query = dataset[i]["question"]
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)

            if system_prompt:
                messages = [("system", system_prompt), ("human", query)]
            else:
                messages = [("human", query)]

            count += 1
            start_time = time.time()
            try:
                with trace("ReAct_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit)]):
                    output_dict = run_agent(args=args, agent=langgraph_agent_executor, messages=messages, label=dataset[i]['answer'], evaluator=evaluator, query=query) # query is just for tracing.
                if output_dict["ispass"]:
                    pass_count += 1
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n"+Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            print(f"Latency: {round(end_time-start_time, 2)} sec")
            pretty_output(i)

    elif args.workload == "webshop":
        from src.tools.webshop_tools.webshop_tools import SearchTool, ClickTool, ResetTool, set_webshop_url
        from src.agents.ReAct.prompt.webshop import get_system_prompt
        set_webshop_url(args.webshop_url)
        reset = ResetTool()
        search = SearchTool()
        click = ClickTool()
        tools = [search, click]
        if args.fewshot > 5:
            print(f"Max fewshot examples for {args.workload} is 5. Running with 5 fewshot examples.")
        system_prompt = get_system_prompt(fewshots=min(args.fewshot, 5))
        langgraph_agent_executor = create_react_agent(model, tools=tools)
        
        for i in range(samples):
            session_id = dataset[i]
            query = reset._run(session_id=session_id)
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)
            if system_prompt:
                messages = [("system", system_prompt), ("human", query)]
            else:
                messages = [("human", query)]
                
            count += 1
            start_time = time.time()
            try:
                with trace("ReAct_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Index:"+str(i)]):
                    output_dict = run_agent(args=args, agent=langgraph_agent_executor, messages=messages, label=None, evaluator=evaluator, query=query)
                if output_dict["ispass"]:
                    pass_count += 1
                
                score_sum += float(output_dict["score"])
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n" + Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            print(f"Latency: {round(end_time-start_time, 2)} sec\n")
            pretty_output(i)
        
    elif args.workload == "math":
        from src.tools.math_tools.math_tools import WolframAlphaTool, CalculatorTool, FinishTool
        from src.agents.ReAct.prompt.math import get_system_prompt
        
        tools = [WolframAlphaTool(), CalculatorTool(), FinishTool()]
        langgraph_agent_executor = create_react_agent(model, tools=tools)
        if args.fewshot > 2:
            print(f"Max fewshot examples for {args.workload} is 2. Running with 2 fewshot examples.")
        system_prompt = get_system_prompt(min(args.fewshot, 2))
        for i in range(samples):
            query = dataset[i]["problem"]
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)
            messages = [("system", system_prompt), ("human", query)]
            count += 1
            start_time = time.time()
            try:
                with trace("ReAct_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Index:"+str(i)]):
                    output_dict = run_agent(args=args, agent=langgraph_agent_executor, messages=messages, label=dataset[i]['solution'], evaluator=evaluator, query=query)
                if output_dict["ispass"]:
                    pass_count += 1
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n" + Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            print(f"Latency: {round(end_time-start_time, 2)} sec\n")
            pretty_output(i)

    elif args.workload == "humaneval":
        from src.tools.humaneval_tools.coding_tools import GeneratorTool, ExecutorTool, FinishTool
        from src.agents.ReAct.prompt.humaneval import HUMANEVAL_PROMPT
        language = "python"
        exe = ExecutorTool(language = language, is_leet = False)
        gen = GeneratorTool(name = "generate", llm=model)
        finish = FinishTool()
        tools = [exe, finish]
        langgraph_agent_executor = create_react_agent(model, tools=tools)
        if args.fewshot > 1:
            print(f"Max fewshot examples for {args.workload} is 1. Running with 1 fewshot example.")
        system_prompt = HUMANEVAL_PROMPT

        for i in range(samples):
            query = dataset[i]["prompt"]
            tests = dataset[i]["test"]
            entry_point = dataset[i]["entry_point"]
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)
            messages = [("system", system_prompt), ("human", query)]
            count += 1
            start_time = time.time()
            try:
                finish.tests = tests
                finish.entry_point = entry_point
                with trace("ReAct_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Index:"+str(i)]):
                    exe.tests_i = gen.invoke(query)
                    output_dict = run_agent(args=args, agent=langgraph_agent_executor, messages=messages, label=None, evaluator=evaluator, query=query)
                if output_dict["ispass"]:
                    pass_count += 1
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n" + Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            print(f"Latency: {round(end_time-start_time, 2)} sec\n")
            pretty_output(i)

@traceable()
def run_agent(args, agent, messages, label=None, evaluator=None, query=None):
    score_output = ""
    for num, chunk in enumerate(
        agent.stream(
            {"messages": messages},
            stream_mode="values",
            config={"recursion_limit": args.iteration_limit}
        )
    ):
        final_output = chunk
        if args.workload == "webshop":
            # Track the last purchase
            if "Your score (min 0.0, max 1.0): " in chunk['messages'][-1].content:
                score_output = chunk['messages'][-1].content
            
    
    output = parse_answer(final_output['messages'][-1].content)
    print(f'Output: {Fore.CYAN+Style.BRIGHT+output+Style.RESET_ALL}')

    score = 0.0      
    if args.workload == "webshop":
        ispass, score = evaluator(score_output)
        if ispass:
            output = score_output
            print(Fore.GREEN+f'Score: {str(score)}'+Style.RESET_ALL)
        else:
            print(Fore.RED+f'Score: {str(score)}'+Style.RESET_ALL)
    else:
        if args.workload != "humaneval":
            print(f'Label: {Fore.CYAN+Style.BRIGHT+label+Style.RESET_ALL}')
        ispass, _ = evaluator(output, label)

    if ispass:
        print(Fore.GREEN + "PASS" + Style.RESET_ALL)
    else:
        print(Fore.RED + "FAIL" + Style.RESET_ALL)
    return {"output": output, "ispass": ispass, "score": score}


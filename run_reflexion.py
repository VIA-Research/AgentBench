import time
import numpy as np
from colorama import Fore, Style

from src.agents.Reflexion.agent import ReflexionAgent
from src.agents.Reflexion.fewshots import get_action_examples, get_reflection_examples
from src.agents.Reflexion.prompt import get_action_prompt, get_reflection_prompt
from langchain_openai import ChatOpenAI
from langsmith import traceable, trace
from dotenv import load_dotenv

load_dotenv()

def get_tools(args):
    if args.workload == "hotpotqa":
        from src.tools.hotpotqa_tools.wikipedia import LookupTool, WikipediaTool, FinishTool
        tools = [WikipediaTool(name="search"), LookupTool(name="lookup"), FinishTool(name="finish")]
    elif args.workload == "math":
        from src.tools.math_tools.math_tools import CalculatorTool, WolframAlphaTool, FinishTool
        tools = [WolframAlphaTool(name="WolframAlpha"), CalculatorTool(name="simplecalc"), FinishTool(name="finish")]
    elif args.workload == "webshop":
        from src.tools.webshop_tools.webshop_tools import SearchTool, ClickTool, FinishTool, set_webshop_url
        set_webshop_url(args.webshop_url)
        tools = [SearchTool(name="search"), ClickTool(name="click"), FinishTool(name="finish")]
    elif args.workload == "humaneval":
        tools = []  # tools will be set in the main function for humaneval
    else:
        raise NotImplementedError(f"Not implmented error: {args.workload}")
    return tools

def main(args):
    ## Setting
    print_log = bool(getattr(args, "print_log", False))
    num_success = 0
    total_score = 0.0
    context_limit = args.context_limit 
    from src.utils import load_dataset, get_evaluation_function
    print(f"Loading dataset for workload: {args.workload}")
    dataset = load_dataset(args.workload, shuffle=args.shuffle)
    evaluator = get_evaluation_function(args.workload)
    iteration = min(len(dataset), args.samples) if args.samples else len(dataset)
    latencies = []

    def pretty_output(i):
        print(Fore.YELLOW+"=" * 30)
        print(f"Sample {i + 1}/{iteration}")
        if args.workload == "webshop":
            print(f"Average score so far: {round(total_score / (i + 1), 2)}")
        print(f"Accuracy so far: {round(num_success / (i + 1), 2)}")
        if latencies:
            print(f"Avg. latency: {round(sum(latencies) / len(latencies), 2)} sec")
            print(f"p50 latency: {round(np.percentile(latencies, 50), 2)} sec")
            print(f"p90 latency: {round(np.percentile(latencies, 90), 2)} sec")
            print(f"p95 latency: {round(np.percentile(latencies, 95), 2)} sec") 
            print(f"p99 latency: {round(np.percentile(latencies, 99), 2)} sec")
        print("=" * 30+Style.RESET_ALL)
        print("\n")

    if args.host:
        host_url = f"http://{args.host}:{args.port}/v1"
    else:
        host_url = None

    llm = ChatOpenAI(model=args.model, base_url=host_url, stream_usage=True, temperature=args.temperature)
    tools = get_tools(args)
    action_prompt = get_action_prompt(args.workload)
    reflection_prompt = get_reflection_prompt(args.workload)
    action_examples = get_action_examples(args.workload, args.fewshot)
    reflection_examples = get_reflection_examples(args.workload, args.fewshot)
    agent = ReflexionAgent(
        actor_llm=llm,
        actor_prompt=action_prompt,
        actor_examples=action_examples, # todo
        reflect_llm=llm,
        reflect_prompt=reflection_prompt,
        reflect_examples=reflection_examples, # todo
        tools=tools,
        context_limit=context_limit,
        workload=args.workload,
        max_steps=args.iteration_limit,
        evaluator=evaluator,
        print_log=print_log,
    )
    if args.workload == "hotpotqa":
        for i in range(iteration):
            data = dataset[i]
            query = data.get("question")
            answer = data.get("answer")
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{iteration}] {query}"+Style.RESET_ALL)
            agent.set_qa(query)
            start = time.time()
            with trace("Reflexion_trace", 
                       tags=[args.workload, 
                             args.model, 
                             "Iteration_limit:"+str(args.iteration_limit), 
                             "Reflection_limit:"+str(args.reflection_limit),
                             "Index:"+str(i)]):
                _, ispass = run_agent(
                    agent,
                    args.workload,
                    query=query,
                    max_reflextions=args.reflection_limit,
                    reset_func=None,
                    label=answer,
                    print_log=print_log,
                ) # query is just for tracing.
            end = time.time()
            latencies.append(end - start)
            print(f"Latency: {round(end - start, 2)} sec\n")
            if ispass:
                num_success += 1
            pretty_output(i)

    elif args.workload == "math":
        from src.tools.math_tools.math_equivalence import extract_boxed_value
        for i in range(iteration):
            data = dataset[i]
            problem = data.get("problem")
            solution = extract_boxed_value(data.get("solution"))
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{iteration}] {problem}"+Style.RESET_ALL)
            agent.set_qa(problem)
            start = time.time()
            with trace("Reflexion_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Reflection_limit:"+str(args.reflection_limit)]):
                _, ispass = run_agent(
                    agent,
                    args.workload,
                    query=problem,
                    max_reflextions=args.reflection_limit,
                    reset_func=None,
                    label=solution,
                    print_log=print_log,
                )
            end = time.time()
            latencies.append(end - start)
            print(f"Latency: {round(end - start, 2)} sec\n")
            if ispass:
                num_success += 1
            pretty_output(i)
        
    elif args.workload == "webshop":
        from src.tools.webshop_tools.webshop_tools import ResetTool
        reset = ResetTool()
        for i in range(iteration):
            session_id = dataset[i]
            reset.session_id=session_id
            query = reset._run()
            agent.set_qa(query)
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{iteration}] {query}"+Style.RESET_ALL)
            start = time.time()
            with trace("Reflexion_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Reflection_limit:"+str(args.reflection_limit)]):
                score, ispass = run_agent(
                    agent,
                    args.workload,
                    query=query,
                    max_reflextions=args.reflection_limit,
                    reset_func=reset._run,
                    label=None,
                    print_log=print_log,
                )
            end = time.time()
            latencies.append(end - start)
            print(f"Latency: {round(end - start, 2)} sec\n")
            total_score += float(score)
            if ispass:
                num_success += 1
            pretty_output(i)

    elif args.workload == "humaneval":
        from src.tools.humaneval_tools.coding_tools import GeneratorTool, ExecutorTool, FinishTool
        gen = GeneratorTool(llm=llm)
        exe = ExecutorTool(language="python", is_leet=False)
        finish = FinishTool()
        tools = [exe, finish]
        agent.tools = tools
        agent.set_tools()

        for i in range(iteration):
            data = dataset[i]
            query = data.get("prompt")  
            tests = data.get("test")   
            entry_point = data.get("entry_point")  
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{iteration}] {query}"+Style.RESET_ALL)
            agent.set_qa(query=query)
            # Generate test cases
            if print_log:
                print("Generating test cases...")
            start = time.time()
            with trace("Reflexion_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Reflection_limit:"+str(args.reflection_limit)]):
                exe.tests_i = gen.invoke(query)
                finish.tests = tests
                finish.entry_point = entry_point  
                _, ispass = run_agent(
                    agent=agent,
                    workload=args.workload,
                    query=query,
                    max_reflextions=args.reflection_limit,
                    label=None,
                    print_log=print_log,
                )
            end = time.time()
            latencies.append(end - start)
            if ispass:
                num_success += 1
            pretty_output(i)
    else:
        NotImplementedError(f"Not implemented error: {args.workload}")
    return

@traceable()
def run_agent(
    agent: ReflexionAgent,
    workload=None,
    query=None,
    max_reflextions=None,
    reset_func=None,
    label="",
    print_log: bool = False,
):
    output = ""
    max_score = 0
    for i in range(max_reflextions):
        try:
            ispass, score = agent.evaluator(output, label)
            if score and score > max_score:
                max_score = score
            if not ispass:
                if print_log:
                    print(Fore.CYAN+Style.BRIGHT+f'[Trial {i+1}/{max_reflextions}]'+Style.RESET_ALL)
                if workload == "webshop" and "Your score (min 0.0, max 1.0):" in output:
                    reset_func() # reset environment for next trial
                output = agent.run()
                if label:
                    print(f'Output: {Fore.CYAN+Style.BRIGHT+output+Style.RESET_ALL}\nLabel: {Fore.CYAN+Style.BRIGHT+label+Style.RESET_ALL}')
                else:
                    print(f'Output: {Fore.CYAN+Style.BRIGHT+output+Style.RESET_ALL}')
            else:
                break
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            output = "Error: {e}"
            print(f"Error: {e}")
    ispass, score = agent.evaluator(output, label)
    if score and score > max_score:
        max_score = score
    if score is not None:
        if score == 1.0:
            print(Fore.GREEN+f'Score: {str(max_score)}'+Style.RESET_ALL)
        else:
            print(Fore.RED+f'Score: {str(max_score)}'+Style.RESET_ALL)
        output = max_score
    if ispass:
        ispass_str = Fore.GREEN + "PASS" + Style.RESET_ALL
    else:
        ispass_str = Fore.RED + "FAIL" + Style.RESET_ALL
    print(ispass_str)
    return output, ispass
    

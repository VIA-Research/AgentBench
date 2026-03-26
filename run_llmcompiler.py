import argparse
import asyncio
import json
import os
import time
import shutil
import random
import string
import re
from colorama import Fore, Style

import numpy as np
from src.agents.LLMCompiler.constants import END_OF_PLAN, JOINNER_FINISH
from src.agents.LLMCompiler.agent import LLMCompiler
from src.agents.LLMCompiler.utils.model_utils import get_model
from src.utils import parse_answer, load_dataset, get_evaluation_function

from langsmith import trace, traceable
from dotenv import load_dotenv

load_dotenv()

# argparser = argparse.ArgumentParser()
# argparser.add_argument("--samples", type=int, default=100, help="number of samples")
# argparser.add_argument("--stream", action="store_true", help="stream plan")
# argparser.add_argument("--logging", action="store_true", help="logging")
# argparser.add_argument(
#     "--model", type=str, default=None, help="model name to override default"
# )
# argparser.add_argument(
#     "--workload",
#     type=str,
#     required=True,
#     choices=["movie", "hotpotqa", "webshop", "math", "humaneval"],
# )
# argparser.add_argument("--do_benchmark", action="store_true", help="do benchmark")
# argparser.add_argument("--max-replan", type=int, default=10, help="max replan count")
# argparser.add_argument("--max-chat-history", type=int, default=10, help="max chat history")
# argparser.add_argument("--fewshot", type=int, default=1, help="number of fewshot examples")

# # vllm-specific arguments
# argparser.add_argument("--port", type=int, default=None, help="port")
# argparser.add_argument("--host", type=str, default=None, help="host")

# args = argparser.parse_args()


# if args.logging:
#     enable_logging(True)
# else:
#     enable_logging(False)

def get_tools(args):
    if args.workload == "hotpotqa":
        from src.agents.LLMCompiler.configs.hotpotqa.tools import tools
        return tools
    elif args.workload == "webshop":
        from src.agents.LLMCompiler.configs.webshop.tools import tools
        return tools
    else:
        raise NotImplementedError(f"Not implmented error: {args.workload}")
    

def get_prompt(args):
    if args.workload == "hotpotqa":
        from src.agents.LLMCompiler.configs.hotpotqa.prompt import get_output_prompt, get_planner_prompt
        return get_output_prompt(args.fewshot), get_planner_prompt(args.fewshot)
    elif args.workload == "webshop":
        from src.agents.LLMCompiler.configs.webshop.prompt import get_output_prompt, get_planner_prompt
        return get_output_prompt(args.fewshot), get_planner_prompt(args.fewshot)
    else:
        raise NotImplementedError(f"Not implmented error: {args.workload}")

@traceable()
async def run_agent(args, agent, question, label=None, evaluator=None):
    def print_pass_fail(ispass):
        if ispass:
            print(Fore.GREEN + "PASS" + Style.RESET_ALL)
        else:
            print(Fore.RED + "FAIL" + Style.RESET_ALL)
    start = time.time()
    raw_answer = await agent.arun(question, callbacks=None)
    end = time.time()
    output = parse_answer(raw_answer)
    print(f'Output: {Fore.CYAN+Style.BRIGHT+output+Style.RESET_ALL}')
    if args.workload == "hotpotqa":
        ispass, _ = evaluator(output, label)
        print(f'Label: {Fore.CYAN+Style.BRIGHT+label+Style.RESET_ALL}')
        print_pass_fail(ispass)
        return {"output": output, "ispass": ispass, "score": None, "time": end - start}
    elif args.workload == "webshop":
        if "Your score (min 0.0, max 1.0): " in output:
            ispass, score = evaluator(output)
        else:
            ispass, score = False, 0.0
        print_pass_fail(ispass)
        return {"output": output, "ispass": ispass, "score": score, "time": end - start}
    else:
        raise NotImplementedError(f"Not implmented error: {args.workload}")


        



async def main(args):
    print_log = bool(getattr(args, "print_log", False))
    if args.workload not in ["hotpotqa", "webshop"]:
        raise NotImplementedError(f"Not implmented error: {args.workload}")
    
    # Load dataset
    print(f"Loading dataset for workload: {args.workload}")
    dataset = load_dataset(args.workload, shuffle=getattr(args, "shuffle", False))
    evaluator = get_evaluation_function(args.workload)
    samples = min(len(dataset), args.samples) if args.samples else len(dataset)

    tools = get_tools(args)
    output_prompt, planner_prompt = get_prompt(args)
    
    llm = get_model(
        model_name=args.model,
        host=args.host,
        port=args.port,
        temperature=args.temperature,
    )
    
    planner_llm = get_model(
        model_name=args.model,
        host=args.host,
        port=args.port,
        temperature=args.temperature,
    )
    
    pass_count = 0
    latencies = []
    score_sum = 0.0
    
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

    if args.workload == "webshop":
        from src.tools.webshop_tools import end_condition
        end_condition_ = end_condition
    else:
        end_condition_ = None
        
    agent = LLMCompiler(
        tools=tools,
        planner_llm=planner_llm,
        planner_example_prompt=planner_prompt,
        planner_example_prompt_replan=planner_prompt+"\nConsidering the previous plans and observations, please plan for the next steps.",
        planner_stop=[END_OF_PLAN],
        planner_stream=True,
        agent_llm=llm,
        joinner_prompt=output_prompt,
        joinner_prompt_final=output_prompt+f"\nThis is the last trial, please think carefully and answer with {JOINNER_FINISH}(answer).",
        max_replans=args.max_replan,
        end_condition=end_condition_,
        max_chat_history=args.max_chat_history,
        print_log=print_log,
    )
    
    all_results = {}
        
    if args.workload == 'webshop':
        from src.tools.webshop_tools import ResetTool
        reset = ResetTool()
        len_dataset = len(dataset)
        for i in range(samples):
            try:
                session_id = dataset[i%len_dataset]
                question = reset._run(session_id=session_id)
                print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {question}"+Style.RESET_ALL)
                with trace("LLMCompiler_trace", 
                            tags=[args.workload, 
                                    args.model, 
                                    "max_replan:"+str(args.max_replan), 
                                    "max_chat_history:"+str(args.max_chat_history),
                                    "index:"+str(i)]):
                    output_dict = await run_agent(args=args, agent=agent, question=question, label=None, evaluator=evaluator)
                
                if output_dict["ispass"]:
                    pass_count += 1
                    print(Fore.GREEN + f"{output_dict['output']}" + Style.RESET_ALL)
                else:
                    print(Fore.RED + f"{output_dict['output']}" + Style.RESET_ALL)
                    
                score_sum += output_dict["score"]
                latencies.append(output_dict["time"])
                print(f"Latency: {round(output_dict['time'], 2)} sec\n")
                pretty_output(i)
                
                all_results[session_id] = {
                    "question": question,
                    "answer": output_dict["output"],  # not normalized
                    "ispass": output_dict["ispass"],
                    "score": output_dict["score"],
                    "time": output_dict["time"],
                }
                    
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
                
    elif args.workload == "hotpotqa":
        from src.agents.LLMCompiler.configs.hotpotqa.tools import clear_pages
        
        for i in range(samples):
            try:
                example = dataset[i]
                id = example["_id"]
                question = example["question"]
                print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {question}"+Style.RESET_ALL)
                
                label = example["answer"]
                with trace("LLMCompiler_trace", 
                            tags=[args.workload, 
                                    args.model, 
                                    "max_replan:"+str(args.max_replan), 
                                    "max_chat_history:"+str(args.max_chat_history),
                                    "index:"+str(i)]):
                    output_dict = await run_agent(args=args, agent=agent, question=question, label=label, evaluator=evaluator)
                
                if output_dict["ispass"]:
                    pass_count += 1
                    
                latencies.append(output_dict["time"])
                print(f"Latency: {round(output_dict['time'], 2)} sec\n")
                pretty_output(i)
                clear_pages()
                
                all_results[id] = {
                    "question": question,
                    "label": "Purchase successfull",  # not normalized
                    "answer": output_dict["output"],  # not normalized
                    "ispass": output_dict["ispass"],
                    "time": output_dict["time"],
                }

            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
                
    # Save results  
    if args.save_trace:
        with open(args.trace_path, "w") as f:
            json.dump(all_results, f, indent=4)
            
if __name__ == "__main__":
    results = asyncio.get_event_loop().run_until_complete(main())

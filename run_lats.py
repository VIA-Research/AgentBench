import json
import os
import time
import traceback
from typing import Any, Dict

import numpy as np
from colorama import Fore, Style
from dotenv import load_dotenv
from langsmith import trace

from src.agents.LATS.hotpotqa.environment import HotpotWikiEnv
from src.agents.LATS.hotpotqa.search import HotpotLATSConfig, HotpotLATSRunner
from src.agents.LATS.hotpotqa.task import HotpotQATask
from src.agents.LATS.humaneval.search import HumanevalLATSConfig, HumanevalLATSRunner
from src.agents.LATS.humaneval.task import HumanevalTask
from src.agents.LATS.math.environment import MathEnv
from src.agents.LATS.math.search import MathLATSConfig, MathLATSRunner
from src.agents.LATS.math.task import MathTask
from src.agents.LATS.model_client import OpenAIChatClient
from src.agents.LATS.webshop.environment import WebshopEnv, evaluate_webshop_output
from src.agents.LATS.webshop.search import WebshopLATSConfig, WebshopLATSRunner
from src.agents.LATS.webshop.task import WebShopTask
from src.utils import get_evaluation_function, load_dataset

load_dotenv()


def _get_attr(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def main(args: Any) -> None:
    if args.workload not in ["hotpotqa", "webshop", "humaneval", "math"]:
        raise NotImplementedError(
            f"LATS supports only ['hotpotqa', 'webshop', 'humaneval', 'math'], got: {args.workload}"
        )

    if args.host:
        host_url = f"http://{args.host}:{args.port}/v1"
    else:
        host_url = None

    n_generate_sample = int(_get_attr(args, "n_generate_sample", 5))
    n_evaluate_sample = int(_get_attr(args, "n_evaluate_sample", 1))
    default_max_depth = 7 if args.workload in ("hotpotqa", "math") else 15
    if args.workload == "humaneval":
        default_max_depth = 8
    max_depth = int(_get_attr(args, "max_depth", default_max_depth))
    iteration_limit = int(_get_attr(args, "iteration_limit", 30))
    sampling_temperature = float(_get_attr(args, "sampling_temperature", 1.0))
    fewshot = int(_get_attr(args, "fewshot", 0))
    print_log = bool(_get_attr(args, "print_log", False))

    client = OpenAIChatClient(
        model=args.model,
        base_url=host_url,
        temperature=float(_get_attr(args, "temperature", 0.0)),
    )

    print(f"Loading dataset for workload: {args.workload}")
    dataset = load_dataset(args.workload, shuffle=_get_attr(args, "shuffle", False))
    evaluator = get_evaluation_function(args.workload)
    samples = min(len(dataset), args.samples) if args.samples else len(dataset)

    pass_count = 0
    score_sum = 0.0
    latencies = []
    all_results: Dict[str, Dict[str, Any]] = {}

    def pretty_output(i: int) -> None:
        print(Fore.YELLOW + "=" * 30)
        print(f"Sample {i + 1}/{samples}")
        if args.workload == "webshop":
            print(f"Average score so far: {round(score_sum / (i + 1), 4)}")
        print(f"Accuracy so far: {round(pass_count / (i + 1), 4)}")
        if latencies:
            print(f"Avg. latency: {round(sum(latencies) / len(latencies), 2)} sec")
            print(f"p50 latency: {round(np.percentile(latencies, 50), 2)} sec")
            print(f"p90 latency: {round(np.percentile(latencies, 90), 2)} sec")
            print(f"p95 latency: {round(np.percentile(latencies, 95), 2)} sec")
            print(f"p99 latency: {round(np.percentile(latencies, 99), 2)} sec")
        print("=" * 30 + Style.RESET_ALL)
        print("")

    if args.workload == "hotpotqa":
        env = HotpotWikiEnv()
        task = HotpotQATask(client=client, fewshot=fewshot)
        config = HotpotLATSConfig(
            iterations=iteration_limit,
            n_generate_sample=n_generate_sample,
            max_depth=max_depth,
            sampling_temperature=sampling_temperature,
            print_log=print_log,
        )
        runner = HotpotLATSRunner(env=env, task=task, client=client, config=config)

        for i in range(samples):
            example = dataset[i]
            sample_id = example["_id"]
            question = example["question"]
            label = example["answer"]
            print(Fore.CYAN + Style.BRIGHT + f"[Sample {i+1}/{samples}] {question}" + Style.RESET_ALL)
            start_t = time.time()
            try:
                with trace(
                    "LATS_trace",
                    tags=[
                        args.workload,
                        args.model,
                        f"iteration_limit:{iteration_limit}",
                        f"n_generate_sample:{n_generate_sample}",
                        f"index:{i}",
                    ],
                ):
                    result = runner.run(question=question, answer=label)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                err_type = type(e).__name__
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__, limit=8))
                print(Fore.RED + f"Error: {err_type}: {e}" + Style.RESET_ALL)
                print(Fore.RED + f"Traceback (hotpot sample idx={i}):\n{tb}" + Style.RESET_ALL)
                result = {
                    "output": "",
                    "ispass": False,
                    "score": None,
                    "reward": 0,
                    "meta": {"error": str(e), "error_type": err_type, "traceback": tb},
                }

            end_t = time.time()
            elapsed = end_t - start_t
            latencies.append(elapsed)
            output = result.get("output", "")
            ispass, _ = evaluator(output, label)
            if ispass:
                pass_count += 1
            print(f"Output: {Fore.CYAN + Style.BRIGHT}{output}{Style.RESET_ALL}")
            print(f"Label: {Fore.CYAN + Style.BRIGHT}{label}{Style.RESET_ALL}")
            print((Fore.GREEN + "PASS" + Style.RESET_ALL) if ispass else (Fore.RED + "FAIL" + Style.RESET_ALL))
            print(f"Latency: {round(elapsed, 2)} sec")
            pretty_output(i)

            all_results[sample_id] = {
                "question": question,
                "label": label,
                "answer": output,
                "ispass": ispass,
                "score": None,
                "time": elapsed,
                "meta": {
                    "reward": result.get("reward"),
                    "em": result.get("em"),
                },
            }

    elif args.workload == "webshop":
        env = WebshopEnv(webshop_url=args.webshop_url)
        task = WebShopTask(client=client, fewshot=fewshot)
        config = WebshopLATSConfig(
            iterations=iteration_limit,
            n_generate_sample=n_generate_sample,
            max_depth=max_depth,
            sampling_temperature=sampling_temperature,
            print_log=print_log,
        )
        runner = WebshopLATSRunner(env=env, task=task, client=client, config=config)

        for i in range(samples):
            session_id = dataset[i]
            print(Fore.CYAN + Style.BRIGHT + f"[Sample {i+1}/{samples}] Session={session_id}" + Style.RESET_ALL)
            start_t = time.time()
            try:
                with trace(
                    "LATS_trace",
                    tags=[
                        args.workload,
                        args.model,
                        f"iteration_limit:{iteration_limit}",
                        f"n_generate_sample:{n_generate_sample}",
                        f"index:{i}",
                    ],
                ):
                    result = runner.run(session_id=session_id)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                err_type = type(e).__name__
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__, limit=8))
                print(Fore.RED + f"Error: {err_type}: {e}" + Style.RESET_ALL)
                print(Fore.RED + f"Traceback (webshop sample idx={i}):\n{tb}" + Style.RESET_ALL)
                result = {
                    "output": "",
                    "ispass": False,
                    "score": 0.0,
                    "reward": 0.0,
                    "meta": {"error": str(e), "error_type": err_type, "traceback": tb},
                }

            end_t = time.time()
            elapsed = end_t - start_t
            latencies.append(elapsed)
            output = result.get("output", "")
            ispass, score = evaluate_webshop_output(output)
            if ispass:
                pass_count += 1
            score_sum += float(score)

            print(f"Output: {Fore.CYAN + Style.BRIGHT}{output}{Style.RESET_ALL}")
            print(f"Score: {round(score, 4)}")
            print((Fore.GREEN + "PASS" + Style.RESET_ALL) if ispass else (Fore.RED + "FAIL" + Style.RESET_ALL))
            print(f"Latency: {round(elapsed, 2)} sec")
            pretty_output(i)

            all_results[session_id] = {
                "question": f"session:{session_id}",
                "label": "Your score (min 0.0, max 1.0): 1.0",
                "answer": output,
                "ispass": ispass,
                "score": score,
                "time": elapsed,
                "meta": {
                    "reward": result.get("reward"),
                },
            }

    elif args.workload == "math":
        env = MathEnv()
        task = MathTask(client=client, fewshot=fewshot)
        config = MathLATSConfig(
            iterations=iteration_limit,
            n_generate_sample=n_generate_sample,
            max_depth=max_depth,
            sampling_temperature=sampling_temperature,
            print_log=print_log,
        )
        runner = MathLATSRunner(env=env, task=task, client=client, config=config)

        for i in range(samples):
            example = dataset[i]
            sample_id = example.get("unique_id", str(i))
            question = example["problem"]
            label = example["solution"]
            print(Fore.CYAN + Style.BRIGHT + f"[Sample {i+1}/{samples}] {question}" + Style.RESET_ALL)
            start_t = time.time()
            try:
                with trace(
                    "LATS_trace",
                    tags=[
                        args.workload,
                        args.model,
                        f"iteration_limit:{iteration_limit}",
                        f"n_generate_sample:{n_generate_sample}",
                        f"index:{i}",
                    ],
                ):
                    result = runner.run(problem=question, answer=label)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                err_type = type(e).__name__
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__, limit=8))
                print(Fore.RED + f"Error: {err_type}: {e}" + Style.RESET_ALL)
                print(Fore.RED + f"Traceback (math sample idx={i}):\n{tb}" + Style.RESET_ALL)
                result = {
                    "output": "",
                    "ispass": False,
                    "score": None,
                    "reward": 0.0,
                    "em": 0,
                    "meta": {"error": str(e), "error_type": err_type, "traceback": tb},
                }

            end_t = time.time()
            elapsed = end_t - start_t
            latencies.append(elapsed)
            output = result.get("output", "")
            ispass, _ = evaluator(output, label)
            if ispass:
                pass_count += 1
            print(f"Output: {Fore.CYAN + Style.BRIGHT}{output}{Style.RESET_ALL}")
            print((Fore.GREEN + "PASS" + Style.RESET_ALL) if ispass else (Fore.RED + "FAIL" + Style.RESET_ALL))
            print(f"Latency: {round(elapsed, 2)} sec")
            pretty_output(i)

            all_results[sample_id] = {
                "question": question,
                "label": label,
                "answer": output,
                "ispass": ispass,
                "score": None,
                "time": elapsed,
                "meta": {
                    "reward": result.get("reward"),
                    "em": result.get("em"),
                },
            }

    else:
        task = HumanevalTask(client=client, print_log=print_log, fewshot=fewshot)
        config = HumanevalLATSConfig(
            iterations=iteration_limit,
            n_generate_sample=n_generate_sample,
            n_evaluate_sample=n_evaluate_sample,
            max_depth=max_depth,
            sampling_temperature=sampling_temperature,
            print_log=print_log,
        )
        runner = HumanevalLATSRunner(task=task, config=config)

        for i in range(samples):
            example = dataset[i]
            sample_id = example.get("task_id", str(i))
            sample_label = sample_id.split("/", 1)[0] if isinstance(sample_id, str) else str(sample_id)
            query = example["prompt"]
            print(Fore.CYAN + Style.BRIGHT + f"[Sample {i+1}/{samples}] {sample_label}" + Style.RESET_ALL)
            start_t = time.time()
            try:
                with trace(
                    "LATS_trace",
                    tags=[
                        args.workload,
                        args.model,
                        f"iteration_limit:{iteration_limit}",
                        f"n_generate_sample:{n_generate_sample}",
                        f"n_evaluate_sample:{n_evaluate_sample}",
                        f"index:{i}",
                    ],
                ):
                    result = runner.run(item=example)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                err_type = type(e).__name__
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__, limit=8))
                print(Fore.RED + f"Error: {err_type}: {e}" + Style.RESET_ALL)
                print(Fore.RED + f"Traceback (humaneval sample idx={i}):\n{tb}" + Style.RESET_ALL)
                result = {
                    "output": "Answer: False",
                    "ispass": False,
                    "score": None,
                    "reward": 0.0,
                    "solution": "",
                    "meta": {"error": str(e), "error_type": err_type, "traceback": tb},
                }

            end_t = time.time()
            elapsed = end_t - start_t
            latencies.append(elapsed)
            output = result.get("output", "Answer: False")
            ispass, _ = evaluator(output, None)
            if ispass:
                pass_count += 1

            print(f"Output: {Fore.CYAN + Style.BRIGHT}{output}{Style.RESET_ALL}")
            print((Fore.GREEN + "PASS" + Style.RESET_ALL) if ispass else (Fore.RED + "FAIL" + Style.RESET_ALL))
            print(f"Latency: {round(elapsed, 2)} sec")
            pretty_output(i)

            all_results[sample_id] = {
                "question": query,
                "label": "Answer: True",
                "answer": output,
                "ispass": ispass,
                "score": None,
                "time": elapsed,
                "meta": {
                    "reward": result.get("reward"),
                    "solution": result.get("solution", ""),
                    "iterations_used": result.get("iterations_used"),
                    "internal_tests_count": result.get("internal_tests_count"),
                },
            }

    if _get_attr(args, "save_trace", False):
        trace_path = _get_attr(args, "trace_path", "./trace_lats.json")
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

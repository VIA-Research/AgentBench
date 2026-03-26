# VIA-AgentBench
This repository contains the AI agent implementations and benchmarking utilities used in our paper: Kim et al., "The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective," HPCA-2026 [[Paper](https://ieeexplore.ieee.org/abstract/document/11408569)].

## Setting
### Prerequisites
- Python 3.13.9
- OpenAI-compatible LLM server. We used vLLM for the LLM endpoint. Refer to the [vLLM OpenAI-Compatible Server documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/).

### Environment Setup
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Copy the template environment file to `.env`:
```bash
cp .env_tmp .env
```
Then edit `.env` to configure the following variables:
- OPENAI_API_KEY: Required by some modules in this project. Do not remove this entry, even if you are not using OpenAI models.
- LANGSMITH_TRACING: Enables LangSmith tracing. Supported values: `true`, `false`. Refer to [LangSmith Document](https://docs.langchain.com/langsmith/home) for more information.
- LANGSMITH_API_KEY: LangSmith API key.
- LANGCHAIN_PROJECT: The name of the LangSmith project, used only when tracing is enabled.
- WOLFRAM_ALPHA_APPID: API key for the Wolfram Alpha tool used in math benchmarks.
  - Create an account and App ID at the [Wolfram Alpha Developer Portal](https://developer.wolframalpha.com/).

## Usage
### Configure AI Agent Parameters
This project uses a configuration file (`config.yaml`) to control agent behavior and runtime settings.

#### Global Settings
The `global` section defines model and environment parameters shared across all agents. For example:
```yaml
global:
  model: "Qwen/Qwen3-32B"
  host: localhost
  port: 8000
  temperature: 0.0
  samples: 5
  shuffle: true
  save_trace: true
  trace_path: "./trace.txt"
  webshop_url: "http://localhost:3000"
```
- **model**: Name of the LLM to use for all agents.
- **host, port**: Address of the LLM server.
- **temperature**: Base LLM sampling temperature shared by all agents.
- **samples**: Number of evaluation samples to run.
- **shuffle**: Enables or disables shuffling of evaluation samples.
- **save_trace**: Saves per-sample outputs to a trace file when supported by the runner.
- **trace_path**: Output path for the saved trace file.
- **webshop_url**: URL endpoint required for the WebShop environment. Refer to [WebShop GitHub](https://github.com/princeton-nlp/WebShop).

#### Agent Definitions
Agents are defined under the `agents` section. Each entry corresponds to one runnable agent. For example:
```yaml
agents:
  my_react_agent: # Pass this name to "python agent_bench.py --agent [agent name]"
    type: "react"
    workload: "hotpotqa"
    iteration_limit: 30
    fewshot: 5

  my_reflexion_agent:
    type: "reflexion"
    workload: "hotpotqa"
    fewshot: 2
    context_limit: 2000
    iteration_limit: 10
    reflection_limit: 3

  my_llmcompiler_agent:
    type: "llmcompiler"
    workload: "hotpotqa"
    fewshot: 3
    max_replan: 20
    max_chat_history: 10

  my_lats_agent:
    type: "lats"
    workload: "hotpotqa"
    fewshot: 1
    iteration_limit: 20
    max_depth: 7
    n_generate_sample: 5
    n_evaluate_sample: 1
    sampling_temperature: 1.0

  my_llmcompiler_agent:
    type: "llmcompiler"
    workload: "hotpotqa"
    fewshot: 3
    max_replan: 20
    max_chat_history: 10
```
Each agent has the following parameter groups:
1. **Agent Type**
   - **type**: Specifies the agent architecture. Supported values: `react`, `reflexion`, `lats`, `llmcompiler`.
2. **Workload**
   - **workload**: Determines which benchmark or environment the agent will run. Valid workloads depend on agent type.
   - ReAct / Reflexion / LATS: `hotpotqa`, `webshop`, `math`, `humaneval`
   - LLMCompiler: `hotpotqa`, `webshop`
3. **Prompt**
   - **fewshot**: Number of few-shot examples used in the initial prompt.
   - **context_limit** (Reflexion only): Maximum number of words for stored conversation history.
4. **Iteration and Search Limits**
   - **iteration_limit**: Maximum number of ReAct steps or search iterations. In Reflexion, this limit applies to the iterations between reflection steps.
   - **reflection_limit** (Reflexion only): Maximum number of reflection cycles.
   - **max_depth** (LATS only): Maximum search depth for tree expansion and simulation.
   - **max_replan** (LLMCompiler only): Maximum number of replanning rounds.
5. **LATS Search Config**
   - **n_generate_sample**: Number of sampled actions generated per expansion.
   - **n_evaluate_sample**: Number of LLM samples used for value evaluation.
   - **sampling_temperature**: LATS-only sampling temperature used when generating child nodes during expansion. This is separate from the global `temperature` setting.
6. **LLMCompiler Config**
   - **max_chat_history** (LLMCompiler only): Maximum number of previous chat-history entries kept for replanning.


### Run Agent
```bash
python agent_bench.py --agent [agent name] --config [config file path] [--print-log]
# For example:
# python agent_bench.py --agent my_react_agent --config config.yaml --print-log
```

### 🛠 Agent Availability
- ✅ **ReAct** (Ready)
- ✅ **Reflexion** (Ready)
- ✅ **LATS** (Ready)
- ✅ **LLMCompiler** (Ready)

## Citation
```bibtex
@inproceedings{kim2026cost,
  title={The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective},
  author={Kim, Jiin and Shin, Byeongjun and Chung, Jinha and Rhu, Minsoo},
  booktitle={2026 IEEE International Symposium on High Performance Computer Architecture (HPCA)}, 
  year={2026},
}
```

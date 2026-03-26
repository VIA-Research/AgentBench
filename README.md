# VIA-AgentBench
The set of AI agent model implementations, benchmarks, and others used in our paper Kim et al., "The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective," HPCA-2026 [[arXiv](https://arxiv.org/abs/2506.04301)]

## Setting
### Prerequisites
- Python 3.13.9
- OpenAI-Compatible LLM Server (We used vLLM for LLM endpoint. Refer to [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/))

### Environment Setup
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Copy the template environment file to `.env`
```bash
cp .env_tmp .env
```
Then edit `.env` to configure the following variables:
- OPENAI_API_KEY: Required by some modules in this project. Do not remove this entry, even if you are not using OpenAI models.

- LANGSMITH_TRACING: Enables LangSmith tracing. Supported values: `true`, `false`. Refer to [LangSmith Document](https://docs.langchain.com/langsmith/home) for more information.

- LANGSMITH_API_KEY: LangSmith API key.

- LANGCHAIN_PROJECT: The name of the LangSmith project, used only when tracing is enabled.

- WOLFRAM_ALPHA_APPID: API key for the Wolfram Alpha tool used in math benchmarks.  
    - Create an account and App ID at [Wolfram Alpha Dev homepage](https://developer.wolframalpha.com/).

## Usage
### Configure AI agent paramters
This project uses a configuration file (`config.yaml`) to control agent behavior and runtime settings.

#### Global Settings
The `global` section defines model and environment parameters shared across all agents. For example, 
```yaml
global:
  model: "Qwen/Qwen3-14B-FP8"
  host: localhost                       
  port: 8000
  temperature: 0.0
  webshop_url: "http://localhost:3000"
```
- **model**: Name of the LLM to use for all agents.

- **host, port**: Address of the LLM server.

- **temperature**: LLM model sampling temperature.

- **webshop_url**: URL endpoint required for the WebShop environment. Refer to [WebShop GitHub](https://github.com/princeton-nlp/WebShop).

#### Agent Definitions
Agents are defined under the `agents` section. Each entry corresponds to one runnable agent. For example,
```yaml
agents:
  my_react_agent: # Name of agent, put this name to "python agent_bench.py --agent [agent name]""
    type: "react"
    workload: "math"
    iteration_limit: 30
    fewshot: 5
    samples: 5
    shuffle: true

  my_reflexion_agent:
    type: "reflexion"
    workload: "webshop"
    fewshot: 2
    context_limit: 2000
    iteration_limit: 5
    reflection_limit: 3
    samples: 5
    shuffle: true
```
Each agent has the following parameter groups:
1. **Agent Type**
    - **type**: Specifies the agent architecture. Supported values: `react`, `reflexion`, `lats`, `llmcompiler`.
2. **Workload**
    - **workload**: Determines which benchmark or environment the agent will run. Valid workloads depend on agent type.
    - ReAct / Reflexion: `hotpotqa`, `webshop`, `math`, `humaneval`
    - LATS: `hotpotqa`, `webshop`
    - LLMCompiler: `hotpotqa`, `webshop`
3. **Prompt**
    - **fewshot**: Number of few shot examples used in the initial prompt.
    - **context_limit** (Reflexion only): Maximum number of words for stored conversation history.
4. **Iteration Limits**
    - **iteration_limit**: Maximum number of ReAct steps (Thought, Action, Observation). In Reflexion, this limit applies to the iterations between reflection steps.
    - **reflection_limit** (Reflexion only): Maximum number of reflection cycles.
    - **max_depth** (LATS only): Maximum search depth for tree expansion/simulation.
5. **LATS Search Config**
    - **prompt_sample**: Prompt style used for action generation (`cot` or `standard`).
    - **n_generate_sample**: Number of sampled actions generated per expansion.
    - **n_evaluate_sample**: Number of LLM samples used for value evaluation.
6. **Evaluation**
    - **samples**: Number of tasks to evaluate.
    - **shuffle**: Enable or disable shuffling of evaluation samples.

#### LATS Example
```yaml
agents:
  my_lats_hotpot_agent:
    type: "lats"
    workload: "hotpotqa"
    fewshot: 1
    iteration_limit: 20
    max_depth: 7
    prompt_sample: "cot"
    n_generate_sample: 5
    n_evaluate_sample: 1
```

### Run Agent
```bash
python agent_bench.py --agent [agent name] --config [config file path]
# For example, 
# python agent_bench.py --agent my_react_agent --config config.yaml
```

### 🛠 Agent Availability
- ✅ **ReAct** (Ready)
- ✅ **Reflexion** (Ready)
- ✅ **LATS** (Ready)
- 🚧 **LLMCompiler** (Working on...)

## Citation
```bibtex
@inproceedings{kim2026cost,
  title={The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective},
  author={Kim, Jiin and Shin, Byeongjun and Chung, Jinha and Rhu, Minsoo},
  booktitle={2026 IEEE International Symposium on High Performance Computer Architecture (HPCA)}, 
  year={2026},
}
```

# Scripts
This directory contains scripts that were used in creating the benchmark.
They are

`create_study.py`: used to generate the OpenML benchmark suite: openml.org/s/218
`find_matching_datasets.py`: used to determine the datasets which are in this benchmark but also used for warm-starting `auto-sklearn`.

## Diploma agent-framework runners

`run_diploma_agent_frameworks.py` runs the agentic framework adapters through
the same AMLB path as the other frameworks:

```powershell
python scripts\run_diploma_agent_frameworks.py --framework AIDE --fold 0
python scripts\run_diploma_agent_frameworks.py --framework AutoGluonAssistant --part classification
```

The available framework definitions are `AutoGluonAssistant`, `AIDE`,
`AutoKaggle`, `AutoMLAgent`, and `DSAgent`. Each definition uses
`frameworks.AgentML`, which receives the OpenML folds from `runbenchmark.py`,
exports a temporary Kaggle-like task folder for the agent, then reads
`submission.csv` back into AMLB scoring.

On Linux, install and clone the external agent frameworks first:

```bash
bash scripts/setup_diploma_agent_frameworks.sh
source scripts/diploma_agent_frameworks.env
bash scripts/start_diploma_ollama.sh
source scripts/diploma_ollama.env
python scripts/run_diploma_agent_frameworks.py --framework all --setup skip --ollama --continue-on-error
```

The bootstrap writes isolated virtualenv paths into
`scripts/diploma_agent_frameworks.env`. AIDE is launched through `AIDE_PYTHON`
when the `aideml` package does not expose an `aide` command-line entry point.

`start_diploma_ollama.sh` starts an Ollama container and writes
`scripts/diploma_ollama.env`. The agent adapters then use the Ollama
OpenAI-compatible endpoint through `AGENT_LLM_BASE_URL`,
`AGENT_LLM_API_KEY`, and `AGENT_LLM_MODEL`.
The actual pulled Ollama model is `LLM_MODEL`; the framework-facing model is
an OpenAI-style alias such as `gpt-4o-mini`, created with `ollama cp`.

```bash
# CPU/default model:
bash scripts/start_diploma_ollama.sh

# GPU override and an explicit model:
OLLAMA_USE_GPU=true LLM_MODEL=qwen2.5-coder:32b bash scripts/start_diploma_ollama.sh

# Optional alias override:
LLM_MODEL=qwen2.5-coder:32b LLM_MODEL_ALIAS=gpt-4o-mini bash scripts/start_diploma_ollama.sh
```

Repository-based frameworks need checkout paths. Set environment variables or
pass normal AMLB framework parameter overrides:

```powershell
$env:AUTOKAGGLE_REPO="D:\Diploma\frameworks\AutoKaggle"
$env:AUTOML_AGENT_REPO="D:\Diploma\frameworks\automl-agent"
$env:DS_AGENT_REPO="D:\Diploma\frameworks\DS-Agent"

python scripts\run_diploma_agent_frameworks.py --framework AutoKaggle --fold 0
python scripts\run_diploma_agent_frameworks.py --framework AutoMLAgent --fold 0
python scripts\run_diploma_agent_frameworks.py --framework DSAgent --fold 0

python scripts\run_diploma_agent_frameworks.py --framework AutoKaggle --extra f._repo=D:\Diploma\frameworks\AutoKaggle
```

The PowerShell wrapper exposes the same common AMLB options:

```powershell
.\scripts\run_diploma_agent_frameworks.ps1 -Framework AIDE -Fold 0
.\scripts\run_diploma_agent_frameworks.ps1 -Framework all -Part classification -Ollama
```

## M4 agent-framework runner

`run_m4_agent_frameworks.py` prepares the M4 frequency-group classification
artifact with `run_m4_classification_frameworks.py`, generates AMLB CSV folds,
then runs the AgentML adapters on the generated benchmark:

```bash
python scripts/run_m4_agent_frameworks.py \
  --framework AutoMLAgent,DSAgent,AIDE \
  --fold 0 \
  --setup skip \
  --ollama \
  --continue-on-error
```

By default it builds two generated folds and runs fold 0. Use `--groups`,
`--n-per-group`, `--window-length`, and `--output-dir` to control the M4
artifact. The same repository environment variables from the diploma runner
are used: `AUTOML_AGENT_REPO`, `DS_AGENT_REPO`, `AIDE_PYTHON`, and friends.

## Fedot.Industrial runner

`run_fedot_industrial_benchmarks.py` runs Fedot.Industrial directly from a
checkout, defaulting to `~/Fedot.Industrial`, on the same diploma tasks and the
generated M4 frequency classification folds:

```bash
python scripts/run_fedot_industrial_benchmarks.py \
  --suite all \
  --fedot-root ~/Fedot.Industrial \
  --diploma-data-dir ~/industrial-learning-agent/data/datasets \
  --fold 0 \
  --continue-on-error
```

Use `--suite diploma` or `--suite m4` to run one side only. Diploma tasks use
prepared CSV folds from `--diploma-data-dir` when available, so the
Fedot.Industrial Poetry environment does not need `openml` for those tasks. M4
generation uses the same fixed-length CSV artifact options as
`run_m4_classification_frameworks.py`, for example `--m4-n-per-group 200` for a
quick smoke run. If the current `automlbenchmark` venv cannot import `fedot`,
the script automatically re-runs itself through `poetry run python` from
`~/Fedot.Industrial`; alternatively pass `--fedot-python "$(cd ~/Fedot.Industrial
&& env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry env info --executable)"`.

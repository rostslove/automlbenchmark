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
python scripts/run_diploma_agent_frameworks.py --framework all --setup skip --continue-on-error
```

The bootstrap writes isolated virtualenv paths into
`scripts/diploma_agent_frameworks.env`. AIDE is launched through `AIDE_PYTHON`
when the `aideml` package does not expose an `aide` command-line entry point.

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
.\scripts\run_diploma_agent_frameworks.ps1 -Framework all -Part classification -Extra "f._provider=openai"
```

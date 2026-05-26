from __future__ import annotations

import logging
import json
import os
import shutil
import shlex
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from amlb.results import save_predictions
from frameworks.shared.callee import output_subdir
from frameworks.shared.utils import Timer


log = logging.getLogger(__name__)

AGENT_FRAMEWORKS = {
    "autogluon-assistant",
    "aide",
    "autokaggle",
    "automl-agent",
    "ds-agent",
}
REGRESSION_TYPES = {"regression"}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OLLAMA_MODEL = "gpt-4o-mini"
AGENT_LLM_BASE_URL_ENV = "AGENT_LLM_BASE_URL"
AGENT_LLM_API_KEY_ENV = "AGENT_LLM_API_KEY"
AGENT_LLM_MODEL_ENV = "AGENT_LLM_MODEL"
LOGGED_LLM_ROUTES: set[tuple[str, str]] = set()
TRUE_VALUES = {"1", "true", "yes", "on"}


def run(dataset, config):
    params = dict(config.framework_params or {})
    agent = params.get("_agent_framework")
    if agent not in AGENT_FRAMEWORKS:
        raise ValueError(f"Unsupported _agent_framework: {agent!r}")

    log.info("\n**** AgentML adapter: %s ****\n", agent)
    workspace = Path(output_subdir("agentml", config)).resolve()
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    row_id = "id"
    target = dataset.target.name
    is_classification = config.type == "classification"
    _, test_df, truth, row_id, labels_path = export_task_files(
        dataset=dataset,
        config=config,
        input_dir=input_dir,
        row_id=row_id,
        target=target,
    )
    prompt_file = input_dir / "overview.txt"

    with Timer() as training:
        command_roots = launch_agent(
            agent=agent,
            params=params,
            config=config,
            input_dir=input_dir,
            output_dir=output_dir,
            prompt_file=prompt_file,
            row_id=row_id,
            labels_path=labels_path,
        )
    log.info("Finished agent run in %ss.", training.duration)

    with Timer() as predict:
        submission = load_submission(
            output_dir=output_dir,
            input_dir=input_dir,
            extra_roots=command_roots,
            row_id=row_id,
            target=target,
        )
        predictions, probabilities, probability_labels = parse_submission(
            submission=submission,
            test_ids=test_df[row_id],
            target=target,
            row_id=row_id,
            is_classification=is_classification,
        )
        if is_classification:
            predictions, truth, probabilities, probability_labels = normalize_classification_outputs(
                dataset=dataset,
                predictions=predictions,
                truth=truth,
                probabilities=probabilities,
                probability_labels=probability_labels,
            )
    log.info("Finished submission parsing in %ss.", predict.duration)

    save_predictions(
        dataset=dataset,
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=truth,
        probabilities=probabilities,
        probabilities_labels=probability_labels,
        target_is_encoded=False,
    )
    return dict(
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )


def normalize_classification_outputs(
    dataset,
    predictions: np.ndarray,
    truth: np.ndarray,
    probabilities: np.ndarray | None,
    probability_labels: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    predictions = stringify_labels(predictions)
    truth = stringify_labels(truth)
    if probabilities is not None:
        labels = [str(label) for label in (probability_labels or [])]
        return predictions, truth, probabilities, labels

    labels = classification_labels(dataset=dataset, predictions=predictions, truth=truth)
    label_to_index = {label: index for index, label in enumerate(labels)}
    one_hot = np.zeros((len(predictions), len(labels)), dtype=float)
    for row_index, prediction in enumerate(predictions):
        one_hot[row_index, label_to_index[str(prediction)]] = 1.0
    return predictions, truth, one_hot, labels


def stringify_labels(values: np.ndarray) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def classification_labels(dataset, predictions: np.ndarray, truth: np.ndarray) -> list[str]:
    encoder = getattr(getattr(dataset, "target", None), "label_encoder", None)
    classes = getattr(encoder, "classes", None)
    labels = [str(label) for label in classes] if classes is not None else []
    seen = set(labels)
    for value in [*truth, *predictions]:
        value = str(value)
        if value not in seen:
            labels.append(value)
            seen.add(value)
    return sorted(labels)


def export_task_files(dataset, config, input_dir: Path, row_id: str, target: str):
    X_train = as_frame(dataset.train.X)
    X_test = as_frame(dataset.test.X)
    y_train = as_series(dataset.train.y, target)
    y_test = as_series(dataset.test.y, target)
    row_id = choose_row_id(row_id, X_train.columns, X_test.columns, target)

    train_df = X_train.copy()
    train_df.insert(0, row_id, range(len(train_df)))
    train_df[target] = y_train.to_numpy()

    test_df = X_test.copy()
    test_df.insert(0, row_id, range(len(test_df)))

    sample_submission = pd.DataFrame(
        {
            row_id: test_df[row_id],
            target: default_prediction(y_train, config.type),
        }
    )

    train_df.to_csv(input_dir / "train.csv", index=False)
    test_df.to_csv(input_dir / "test.csv", index=False)
    sample_submission.to_csv(input_dir / "sample_submission.csv", index=False)
    evaluation_dir = input_dir.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    labels_path = evaluation_dir / "_amlb_test_labels.csv"
    pd.DataFrame({row_id: test_df[row_id], target: y_test}).to_csv(
        labels_path,
        index=False,
    )

    prompt = build_prompt(config, target, row_id)
    (input_dir / "overview.txt").write_text(prompt, encoding="utf-8")
    (input_dir / "data_description.txt").write_text(prompt, encoding="utf-8")
    return train_df, test_df, y_test.to_numpy(), row_id, labels_path


def as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reset_index(drop=True)
    return pd.DataFrame(value).reset_index(drop=True)


def as_series(value: Any, name: str) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        series = value.iloc[:, 0]
    elif isinstance(value, pd.Series):
        series = value
    else:
        series = pd.Series(np.asarray(value).reshape(-1))
    return series.reset_index(drop=True).rename(name)


def choose_row_id(preferred: str, *columns: Sequence[str]) -> str:
    used = {str(column) for group in columns for column in group}
    if preferred not in used:
        return preferred
    index = 1
    while f"__row_id_{index}__" in used:
        index += 1
    return f"__row_id_{index}__"


def default_prediction(y_train: pd.Series, task_type: str):
    if task_type in REGRESSION_TYPES:
        value = pd.to_numeric(y_train, errors="coerce").mean()
        return 0.0 if pd.isna(value) else float(value)
    mode = y_train.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else y_train.iloc[0]


def build_prompt(config, target: str, row_id: str) -> str:
    metrics = ", ".join(config.metrics or [config.metric])
    return textwrap.dedent(
        f"""
        You are solving an AutoML benchmark task.

        Task name: {config.name}
        Fold: {config.fold}
        Problem type: {config.type}
        Target column: {target}
        Benchmark metrics: {metrics}
        Time budget: {config.max_runtime_seconds} seconds
        CPU cores: {config.cores}

        Files:
        - train.csv contains features and the target column {target}.
        - test.csv contains test features and identifier column {row_id}; it has no target labels.
        - sample_submission.csv shows the required prediction format.

        Train only on train.csv. Use an internal validation split or cross-validation
        for model selection. Create submission.csv with exactly two required columns:
        {row_id}, {target}. Save submission.csv in the output directory requested by
        the launcher, or in the current task directory if no output directory is known.
        """
    ).strip()


def launch_agent(
    agent: str,
    params: dict[str, Any],
    config,
    input_dir: Path,
    output_dir: Path,
    prompt_file: Path,
    row_id: str,
    labels_path: Path,
) -> list[Path]:
    if agent == "autogluon-assistant":
        return run_autogluon_assistant(params, config, input_dir, output_dir, prompt_file)
    if agent == "aide":
        return run_aide(params, config, input_dir, output_dir)
    if agent == "autokaggle":
        return run_autokaggle(params, config, input_dir)
    if agent == "automl-agent":
        return run_automl_agent(params, input_dir, output_dir, prompt_file)
    if agent == "ds-agent":
        return run_ds_agent(
            params,
            config,
            input_dir,
            row_id=row_id,
            labels_path=labels_path,
        )
    raise ValueError(agent)


def run_autogluon_assistant(
    params: dict[str, Any],
    config,
    input_dir: Path,
    output_dir: Path,
    prompt_file: Path,
) -> list[Path]:
    command = resolve_command(params, "_command", "MLZERO_COMMAND", "mlzero")
    cmd = split_command(command) + [
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-t",
        prompt_file.read_text(encoding="utf-8"),
        "-n",
        str(params.get("_max_iterations", 5)),
    ]
    config_path = params.get("_config")
    env = external_env(params)
    if not config_path and (params.get("_provider") or env.get(AGENT_LLM_BASE_URL_ENV)):
        config_path = write_autogluon_assistant_config(output_dir, params, env)
    if config_path:
        patch_autogluon_assistant_builtin_configs(command, Path(config_path), env)
        cmd += ["-c", str(config_path)]
    run_external(cmd, cwd=output_dir, params=params, config=config, env=env)
    return [output_dir]


def run_aide(
    params: dict[str, Any],
    config,
    input_dir: Path,
    output_dir: Path,
) -> list[Path]:
    goal = (
        f"Build a strong tabular {config.type} model for {config.name}. "
        "Train on train.csv, predict test.csv, and write submission.csv."
    )
    eval_text = "Optimize the benchmark metric on validation data."
    if config.type == "regression":
        eval_text = "Minimize validation regression error."
    elif config.metric in {"auc", "auc_ovr"}:
        eval_text = "Maximize validation AUC when probabilities are available."
    elif config.metric == "f1":
        eval_text = "Maximize validation F1 score."

    env = external_env(params)
    code_model = params.get("_code_model")
    feedback_model = params.get("_feedback_model")
    if uses_agent_llm(env):
        code_model = code_model or resolve_model(
            params,
            "_code_model",
            "OPENROUTER_MODEL",
            "gpt-4-turbo",
            env=env,
        )
        feedback_model = feedback_model or resolve_model(
            params,
            "_feedback_model",
            "OPENROUTER_MODEL",
            "gpt-4o",
            env=env,
        )

    command = resolve_command(params, "_command", "AIDE_COMMAND", "aide")
    aide_cli_available = command_available(command)
    use_aide_cli = aide_cli_available and (
        not uses_agent_llm(env) or is_truthy(params.get("_use_cli"))
    )
    if use_aide_cli:
        cmd = split_command(command) + [
            f"data_dir={input_dir}",
            f"goal={goal}",
            f"eval={eval_text}",
            f"agent.steps={params.get('_steps', 20)}",
        ]
        if code_model:
            cmd.append(f"agent.code.model={code_model}")
        if feedback_model:
            cmd.append(f"agent.feedback.model={feedback_model}")
        run_external(cmd, cwd=output_dir, params=params, config=config, env=env)
        return [output_dir]

    if aide_cli_available:
        log.info(
            "AIDE CLI command `%s` is available, but Agent LLM routing is active; "
            "using AIDE Python API for OpenAI-compatible settings.",
            command,
        )
    else:
        log.info("AIDE CLI command `%s` is unavailable; using AIDE Python API.", command)
    adapter = Path(__file__).with_name("aide_runner.py")
    cmd = [
        resolve_python(params, "AIDE_PYTHON"),
        str(adapter),
        "--data-dir",
        str(input_dir),
        "--goal",
        goal,
        "--eval",
        eval_text,
        "--steps",
        str(params.get("_steps", 20)),
        "--output-dir",
        str(output_dir),
    ]
    if code_model:
        cmd += ["--code-model", str(code_model)]
    if feedback_model:
        cmd += ["--feedback-model", str(feedback_model)]
    run_external(cmd, cwd=output_dir, params=params, config=config, env=env)
    return [output_dir]


def run_autokaggle(params: dict[str, Any], config, input_dir: Path) -> list[Path]:
    repo = require_repo(params, "_repo", "AUTOKAGGLE_REPO", "AutoKaggle")
    competition = safe_name(f"{config.name}_fold{config.fold}")
    competition_dir = repo / "multi_agents" / "competition" / competition
    copytree_contents(input_dir, competition_dir, force=True)
    env = external_env(params)
    cmd = [
        resolve_python(params, "AUTOKAGGLE_PYTHON"),
        "framework.py",
        "--competition",
        competition,
        "--model",
        resolve_model(params, "_model", "OPENROUTER_MODEL", "gpt_4o", env=env),
    ]
    api_key_state = write_autokaggle_api_key(repo, env)
    try:
        run_external(cmd, cwd=repo, params=params, config=config, env=env)
    finally:
        restore_autokaggle_api_key(repo, api_key_state)
    return [
        competition_dir,
        repo / "multi_agents" / "experiments_history" / competition,
    ]


def run_automl_agent(
    params: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    prompt_file: Path,
) -> list[Path]:
    repo = require_repo(params, "_repo", "AUTOML_AGENT_REPO", "AutoML-Agent")
    adapter = Path(__file__).with_name("automl_agent_runner.py")
    env = external_env(params)
    cmd = [
        resolve_python(params, "AUTOML_AGENT_PYTHON"),
        str(adapter),
        "--repo",
        str(repo),
        "--data-path",
        str(input_dir / "train.csv"),
        "--prompt-file",
        str(prompt_file),
        "--llm",
        resolve_model(params, "_llm", "OPENROUTER_MODEL", "gpt-4", env=env),
        "--output-dir",
        str(output_dir),
    ]
    run_external(cmd, cwd=repo, params=params, config=None, env=env)
    return [output_dir, repo / "agent_workspace"]


def run_ds_agent(
    params: dict[str, Any],
    config,
    input_dir: Path,
    row_id: str,
    labels_path: Path,
) -> list[Path]:
    repo = require_repo(params, "_repo", "DS_AGENT_REPO", "DS-Agent")
    task_name = safe_name(f"{config.name}_fold{config.fold}")
    bench_dir = repo / "development" / "benchmarks" / task_name
    copytree_contents(input_dir, bench_dir, force=True)
    shutil.copy2(labels_path, bench_dir / "_amlb_test_labels.csv")
    write_ds_agent_task_files(bench_dir, config, row_id=row_id)
    runner_dir = repo / "development" / "MLAgentBench"
    env = external_env(params)
    cmd = [
        resolve_python(params, "DS_AGENT_PYTHON"),
        "runner.py",
        "--task",
        task_name,
        "--llm-name",
        resolve_model(
            params,
            "_llm_name",
            "OPENROUTER_MODEL",
            "gpt-3.5-turbo-16k",
            env=env,
        ),
        "--edit-script-llm-name",
        resolve_model(
            params,
            "_edit_llm_name",
            "OPENROUTER_MODEL",
            "gpt-3.5-turbo-16k",
            env=env,
        ),
    ]
    run_external(cmd, cwd=runner_dir, params=params, config=config, env=env)
    return [bench_dir, runner_dir / "workspace", runner_dir / "logs"]


def run_external(
    cmd: Sequence[str],
    cwd: Path,
    params: dict[str, Any],
    config,
    env: dict[str, str] | None = None,
) -> None:
    env = env or external_env(params)
    timeout = None
    if params.get("_command_timeout_seconds"):
        timeout = int(params["_command_timeout_seconds"])
    elif config is not None:
        timeout = int(getattr(config, "job_timeout_seconds", 0) or 0) or None
    if uses_agent_llm(env):
        log_agent_llm_route(env)
    log.info("Running external command in %s:\n%s", cwd, quote_cmd(cmd))
    capture_output = is_truthy(params.get("_capture_output")) or is_truthy(
        env.get("AGENTML_CAPTURE_OUTPUT")
    )
    if capture_output:
        completed = subprocess.run(
            list(map(str, cmd)),
            cwd=cwd,
            env=env,
            timeout=timeout,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.stdout:
            log.info("External command stdout tail:\n%s", tail_text(completed.stdout))
        if completed.stderr:
            log.error("External command stderr tail:\n%s", tail_text(completed.stderr))
    else:
        completed = subprocess.run(
            list(map(str, cmd)),
            cwd=cwd,
            env=env,
            timeout=timeout,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"External command failed with exit code {completed.returncode}: {quote_cmd(cmd)}"
        )


def external_env(params: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in dict(params.get("_env") or {}).items()})

    agent_base_url = env.get(AGENT_LLM_BASE_URL_ENV) or env.get("OLLAMA_OPENAI_BASE_URL")
    agent_model = (
        env.get(AGENT_LLM_MODEL_ENV)
        or env.get("LLM_MODEL_ALIAS")
        or env.get("OLLAMA_MODEL_ALIAS")
        or DEFAULT_OLLAMA_MODEL
    )
    if agent_base_url:
        env[AGENT_LLM_BASE_URL_ENV] = agent_base_url
        env[AGENT_LLM_API_KEY_ENV] = env.get(AGENT_LLM_API_KEY_ENV) or "ollama"
        env["OPENAI_API_KEY"] = env[AGENT_LLM_API_KEY_ENV]
        env["OPENAI_BASE_URL"] = agent_base_url
        env["OPENAI_API_BASE"] = agent_base_url
        env["OPENAI_API_BASE_URL"] = agent_base_url
        env["LITELLM_API_BASE"] = agent_base_url
        env["LITELLM_API_KEY"] = env[AGENT_LLM_API_KEY_ENV]
        if agent_model:
            env[AGENT_LLM_MODEL_ENV] = agent_model
            env["LITELLM_MODEL"] = agent_model
        else:
            env[AGENT_LLM_MODEL_ENV] = DEFAULT_OLLAMA_MODEL
            env["LITELLM_MODEL"] = DEFAULT_OLLAMA_MODEL
        env.pop("OPENROUTER_API_KEY", None)
        env.pop("OPENROUTER_MODEL", None)
        add_no_proxy(env, ["127.0.0.1", "localhost", "ollama"])
    elif env.get("OPENROUTER_API_KEY"):
        env["OPENAI_API_KEY"] = env["OPENROUTER_API_KEY"]
        env.setdefault("OPENAI_BASE_URL", OPENROUTER_BASE_URL)
        env.setdefault("OPENAI_API_BASE", env["OPENAI_BASE_URL"])
        env.setdefault("OPENROUTER_MODEL", "openrouter/free")
    elif env.get("OPENAI_BASE_URL"):
        env.setdefault("OPENAI_API_BASE", env["OPENAI_BASE_URL"])
    return env


def resolve_model(
    params: dict[str, Any],
    param_key: str,
    env_var: str,
    default: str,
    env: dict[str, str] | None = None,
) -> str:
    source_env = env or os.environ
    value = params.get(param_key)
    if value and str(value) != default:
        return str(value)

    if source_env.get(AGENT_LLM_MODEL_ENV):
        return str(source_env[AGENT_LLM_MODEL_ENV])
    base_url = source_env.get(AGENT_LLM_BASE_URL_ENV) or source_env.get("OLLAMA_OPENAI_BASE_URL")
    if base_url:
        return str(
            source_env.get("LLM_MODEL_ALIAS")
            or source_env.get("OLLAMA_MODEL_ALIAS")
            or DEFAULT_OLLAMA_MODEL
        )

    openai_base_url = source_env.get("OPENAI_BASE_URL") or source_env.get("OPENAI_API_BASE") or ""
    if is_local_ollama_url(openai_base_url):
        return str(
            source_env.get("LLM_MODEL_ALIAS")
            or source_env.get("OLLAMA_MODEL_ALIAS")
            or DEFAULT_OLLAMA_MODEL
        )

    if source_env.get("OPENROUTER_API_KEY") and source_env.get(env_var):
        return str(source_env[env_var])
    if source_env.get("OPENROUTER_API_KEY"):
        return str(source_env.get(env_var) or "openrouter/free")
    return str(value or source_env.get(env_var) or default)


def is_local_ollama_url(value: str) -> bool:
    return value.startswith("http://127.0.0.1:") or value.startswith("http://localhost:")


def uses_agent_llm(env: dict[str, str]) -> bool:
    base_url = (
        env.get(AGENT_LLM_BASE_URL_ENV)
        or env.get("OLLAMA_OPENAI_BASE_URL")
        or env.get("OPENAI_BASE_URL")
        or env.get("OPENAI_API_BASE")
        or ""
    )
    return bool(env.get(AGENT_LLM_MODEL_ENV) or env.get(AGENT_LLM_BASE_URL_ENV) or is_local_ollama_url(base_url))


def log_agent_llm_route(env: dict[str, str]) -> None:
    base_url = (
        env.get(AGENT_LLM_BASE_URL_ENV)
        or env.get("OLLAMA_OPENAI_BASE_URL")
        or env.get("OPENAI_BASE_URL")
        or env.get("OPENAI_API_BASE")
        or ""
    ).rstrip("/")
    model = env.get(AGENT_LLM_MODEL_ENV) or DEFAULT_OLLAMA_MODEL
    route = (base_url, model)
    if not base_url or route in LOGGED_LLM_ROUTES:
        return
    LOGGED_LLM_ROUTES.add(route)

    log.info("Agent LLM route: base_url=%s model=%s", base_url, model)
    models = list_openai_compatible_models(base_url, env)
    if models is None:
        return
    log.info("Agent LLM endpoint models: %s", ", ".join(models[:20]) or "<none>")
    if model not in models and f"{model}:latest" not in models:
        log.warning(
            "Agent LLM model `%s` is not listed by %s/models. "
            "Create an Ollama alias with `ollama cp` or rerun scripts/start_diploma_ollama.sh.",
            model,
            base_url,
        )


def list_openai_compatible_models(base_url: str, env: dict[str, str]) -> list[str] | None:
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {env.get('OPENAI_API_KEY', '')}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as err:
        log.warning("Could not probe Agent LLM endpoint %s/models: %s", base_url, err)
        return None

    return [
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]


def normalize_model_for_openai_provider(model: str, env: dict[str, str]) -> str:
    base_url = (
        env.get(AGENT_LLM_BASE_URL_ENV)
        or env.get("OLLAMA_OPENAI_BASE_URL")
        or env.get("OPENAI_BASE_URL")
        or env.get("OPENAI_API_BASE")
        or ""
    ).rstrip("/")
    if not base_url:
        return model
    models = list_openai_compatible_models(base_url, env)
    if models and model not in models and f"{model}:latest" in models:
        return f"{model}:latest"
    return model


def patch_autogluon_assistant_builtin_configs(
    command: str | Sequence[str],
    config_path: Path,
    env: dict[str, str],
) -> None:
    if not uses_agent_llm(env):
        return
    if str(env.get("AGENTML_PATCH_AUTOGLUON_CONFIGS", "1")).lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return

    parts = split_command(command)
    if not parts:
        return
    executable = shutil.which(parts[0]) or parts[0]
    executable_path = Path(executable).expanduser()
    python_candidates = [
        executable_path.parent / "python",
        executable_path.parent / "python3",
        executable_path.parent / "python.exe",
        Path(sys.executable),
    ]
    python = next((candidate for candidate in python_candidates if candidate.exists()), None)
    if python is None:
        log.warning(
            "Could not locate Python next to AutoGluonAssistant command `%s`; "
            "skipping built-in config patch.",
            parts[0],
        )
        return

    patch_script = r"""
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

config_path = Path(sys.argv[1]).resolve()
try:
    import autogluon.assistant as assistant
except Exception as err:
    print(f"Could not import autogluon.assistant: {err}", file=sys.stderr)
    raise SystemExit(2)

target_names = {
    "default.yaml",
    "bedrock.yaml",
    "openai.yaml",
    "anthropic.yaml",
    "sagemaker.yaml",
}
targets = []
for package_root in getattr(assistant, "__path__", []):
    config_root = Path(package_root) / "configs"
    if not config_root.is_dir():
        continue
    for target in config_root.rglob("*.yaml"):
        if target.name in target_names and target.is_file():
            targets.append(target)

deduped = []
seen = set()
for target in targets:
    key = str(target.resolve())
    if key not in seen:
        deduped.append(target)
        seen.add(key)

if not deduped:
    print("No AutoGluonAssistant provider configs found", file=sys.stderr)
    raise SystemExit(3)

agent_config = OmegaConf.load(config_path)
for target in deduped:
    backup = target.with_name(target.name + ".agentml.bak")
    if not backup.exists():
        shutil.copy2(target, backup)
    base_config = OmegaConf.load(backup)
    merged_config = OmegaConf.merge(base_config, agent_config)
    OmegaConf.save(config=merged_config, f=target)
    print(target)
"""
    completed = subprocess.run(
        [str(python), "-c", patch_script, str(config_path)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 0:
        if stdout:
            log.info(
                "Patched AutoGluonAssistant built-in LLM configs for Agent LLM:\n%s",
                stdout,
            )
        return
    log.warning(
        "Could not patch AutoGluonAssistant built-in LLM configs with %s "
        "(exit code %s). stdout=%s stderr=%s",
        python,
        completed.returncode,
        stdout,
        stderr,
    )


def add_no_proxy(env: dict[str, str], entries: Sequence[str]) -> None:
    for key in ("NO_PROXY", "no_proxy"):
        current = [item.strip() for item in env.get(key, "").split(",") if item.strip()]
        seen = set(current)
        for entry in entries:
            if entry not in seen:
                current.append(entry)
                seen.add(entry)
        env[key] = ",".join(current)


def write_autogluon_assistant_config(
    output_dir: Path,
    params: dict[str, Any],
    env: dict[str, str],
) -> Path:
    model = resolve_model(params, "_model", "OPENROUTER_MODEL", "gpt-4o-mini", env=env)
    model = normalize_model_for_openai_provider(model, env)
    proxy_url = env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE") or ""
    proxy_line = f'"{proxy_url}"' if proxy_url else "null"
    config_path = output_dir / "autogluon_assistant_llm.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""\
            per_execution_timeout: 86400
            max_file_group_size_to_show: 5
            num_example_files_to_show: 1
            max_chars_per_file: 768
            num_tutorial_retrievals: 30
            max_num_tutorials: 5
            max_user_input_length: 2048
            max_error_message_length: 2048
            max_tutorial_length: 32768
            configure_env: false
            condense_tutorials: true
            use_tutorial_summary: true
            continuous_improvement: false
            optimize_system_resources: false
            cleanup_unused_env: true
            enable_meta_prompting: false
            llm: &default_llm
              provider: openai
              model: "{model}"
              max_tokens: 4096
              proxy_url: {proxy_line}
              temperature: 0.1
              top_p: 0.9
              verbose: true
              multi_turn: false
              template: null
              add_coding_format_instruction: false
              apply_meta_prompting: false
            coder:
              <<: *default_llm
              multi_turn: true
            python_coder:
              <<: *default_llm
              multi_turn: true
              apply_meta_prompting: true
            bash_coder:
              <<: *default_llm
              multi_turn: true
            executer:
              <<: *default_llm
              max_stdout_length: 8192
              max_stderr_length: 2048
            meta_prompting:
              <<: *default_llm
            reader:
              <<: *default_llm
              details: false
            error_analyzer:
              <<: *default_llm
            retriever:
              <<: *default_llm
            reranker:
              <<: *default_llm
              temperature: 0.0
              top_p: 1.0
            description_file_retriever:
              <<: *default_llm
              temperature: 0.0
              top_p: 1.0
            task_descriptor:
              <<: *default_llm
              max_description_files_length_to_show: 1024
              max_description_files_length_for_summarization: 16384
              apply_meta_prompting: true
            tool_selector:
              <<: *default_llm
              temperature: 0.0
              top_p: 1.0
            """
        ),
        encoding="utf-8",
    )
    return config_path


def write_autokaggle_api_key(
    repo: Path,
    env: dict[str, str],
) -> tuple[bool, str] | None:
    api_key = env.get("OPENAI_API_KEY")
    base_url = env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE")
    if not api_key or not base_url:
        return None

    api_key_path = repo / "api_key.txt"
    existed = api_key_path.exists()
    previous = api_key_path.read_text(encoding="utf-8") if existed else ""
    api_key_path.write_text(f"{api_key}\n{base_url}\n", encoding="utf-8")
    return existed, previous


def restore_autokaggle_api_key(
    repo: Path,
    state: tuple[bool, str] | None,
) -> None:
    if state is None:
        return
    api_key_path = repo / "api_key.txt"
    existed, previous = state
    if existed:
        api_key_path.write_text(previous, encoding="utf-8")
    elif api_key_path.exists():
        api_key_path.unlink()


def load_submission(
    output_dir: Path,
    input_dir: Path,
    extra_roots: Sequence[Path],
    row_id: str,
    target: str,
) -> pd.DataFrame:
    candidates = []
    for root in [output_dir, input_dir, *extra_roots]:
        if root.exists():
            candidates.extend(root.rglob("submission.csv"))
            candidates.extend(root.rglob("predictions.csv"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "No submission.csv or predictions.csv found. "
            f"Searched: {[str(p) for p in [output_dir, input_dir, *extra_roots]]}"
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    submission = pd.read_csv(candidates[0])
    log.info("Using submission file: %s", candidates[0])
    if row_id not in submission.columns:
        raise ValueError(f"Submission is missing row id column {row_id!r}.")
    if target not in submission.columns:
        raise ValueError(f"Submission is missing target column {target!r}.")
    return submission


def parse_submission(
    submission: pd.DataFrame,
    test_ids: pd.Series,
    target: str,
    row_id: str,
    is_classification: bool,
) -> tuple[np.ndarray, np.ndarray | None, list[str] | None]:
    ordered = pd.DataFrame({row_id: test_ids}).merge(
        submission,
        on=row_id,
        how="left",
        validate="one_to_one",
    )
    if ordered[target].isna().any():
        missing = int(ordered[target].isna().sum())
        raise ValueError(f"Submission is missing predictions for {missing} test rows.")
    predictions = ordered[target].to_numpy()
    if not is_classification:
        predictions = pd.to_numeric(ordered[target], errors="raise").to_numpy()
        return predictions, None, None

    proba_cols = probability_columns(ordered, target=target, row_id=row_id)
    if not proba_cols:
        return predictions, None, None
    probabilities = ordered[proba_cols].to_numpy()
    labels = [probability_label(column) for column in proba_cols]
    return predictions, probabilities, labels


def probability_columns(frame: pd.DataFrame, target: str, row_id: str) -> list[str]:
    excluded = {target, row_id}
    prefixed = [
        column
        for column in frame.columns
        if column not in excluded
        and any(str(column).startswith(prefix) for prefix in ("proba_", "prob_", "p_"))
    ]
    return prefixed


def probability_label(column: str) -> str:
    for prefix in ("proba_", "prob_", "p_"):
        if str(column).startswith(prefix):
            return str(column)[len(prefix) :]
    return str(column)


def split_command(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command, posix=os.name != "nt")
    return list(command)


def resolve_command(
    params: dict[str, Any],
    param_key: str,
    env_var: str,
    default: str,
) -> str:
    value = params.get(param_key)
    if value and str(value) != default:
        return str(value)
    return str(os.environ.get(env_var) or value or default)


def resolve_python(params: dict[str, Any], env_var: str) -> str:
    return str(params.get("_python") or os.environ.get(env_var) or sys.executable)


def is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def tail_text(value: str, max_lines: int = 200) -> str:
    lines = value.splitlines()
    if len(lines) <= max_lines:
        return value.rstrip()
    omitted = len(lines) - max_lines
    return f"... omitted {omitted} lines ...\n" + "\n".join(lines[-max_lines:])


def command_available(command: str | Sequence[str]) -> bool:
    parts = split_command(command)
    if not parts:
        return False
    executable = Path(parts[0]).expanduser()
    return executable.exists() or shutil.which(parts[0]) is not None


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def require_repo(params: dict[str, Any], key: str, env_var: str, label: str) -> Path:
    raw = params.get(key) or os.environ.get(env_var)
    if not raw:
        raise ValueError(
            f"{label} repository path is required. Set {env_var} or pass -Xf.{key}=<path>."
        )
    repo = Path(str(raw)).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"{label} repository path does not exist: {repo}")
    return repo


def copytree_contents(source: Path, destination: Path, force: bool) -> None:
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def write_ds_agent_task_files(bench_dir: Path, config, row_id: str) -> None:
    train = pd.read_csv(bench_dir / "train.csv", nrows=1)
    target = train.columns[-1]
    (bench_dir / "prepared").write_text("", encoding="utf-8")
    (bench_dir / "research_problem.txt").write_text(
        (bench_dir / "overview.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    train_py = f"""\
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

TARGET = {target!r}
ROW_ID = {row_id!r}
TASK_TYPE = {config.type!r}

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
features = [c for c in train.columns if c not in (TARGET, ROW_ID)]
X_train, X_valid, y_train, y_valid = train_test_split(
    pd.get_dummies(train[features]),
    train[TARGET],
    test_size=0.2,
    random_state=42,
)
model = RandomForestRegressor(random_state=42) if TASK_TYPE == "regression" else RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
valid_pred = model.predict(X_valid)
if TASK_TYPE == "regression":
    print("validation_rmse", mean_squared_error(y_valid, valid_pred, squared=False))
else:
    print("validation_accuracy", accuracy_score(y_valid, valid_pred))
test_X = pd.get_dummies(test[features])
test_X = test_X.reindex(columns=X_train.columns, fill_value=0)
pred = model.predict(test_X)
pd.DataFrame({{ROW_ID: test[ROW_ID], TARGET: pred}}).to_csv("submission.csv", index=False)
"""
    (bench_dir / "train.py").write_text(train_py, encoding="utf-8")
    submission_py = f"""\
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

TARGET = {target!r}
ROW_ID = {row_id!r}
TASK_TYPE = {config.type!r}

def evaluate_file(submission_path, labels_path="_amlb_test_labels.csv"):
    sub = pd.read_csv(submission_path)
    labels = pd.read_csv(labels_path)
    merged = labels.merge(sub, on=ROW_ID, suffixes=("_true", "_pred"))
    if TASK_TYPE == "regression":
        return mean_squared_error(merged[f"{{TARGET}}_true"], merged[f"{{TARGET}}_pred"], squared=False)
    return accuracy_score(merged[f"{{TARGET}}_true"], merged[f"{{TARGET}}_pred"])
"""
    (bench_dir / "submission.py").write_text(submission_py, encoding="utf-8")

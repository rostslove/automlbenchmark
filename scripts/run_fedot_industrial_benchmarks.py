#!/usr/bin/env python
"""Run Fedot.Industrial directly on diploma_mixed and M4 classification tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


CLASSIFICATION_TASKS = {
    "kc2_binary_classification",
    "iris_multiclass_classification",
    "credit_g_binary_classification",
    "m4_frequency_classification",
}
REGRESSION_TASKS = {
    "cholesterol_regression",
    "autoMpg_regression",
    "kin8nm_regression",
}
M4_TASK_NAME = "m4_frequency_classification"
M4_TARGET_COLUMN = "frequency_group"
RESULT_COLUMNS = [
    "framework",
    "benchmark",
    "task",
    "task_type",
    "openml_task_id",
    "fold",
    "status",
    "duration_seconds",
    "train_rows",
    "test_rows",
    "n_features",
    "metric",
    "metric_value",
    "error",
]


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    benchmark: str
    task_type: str
    metrics: tuple[str, ...]
    folds: int
    max_runtime_seconds: int
    openml_task_id: int | None = None
    target: str | None = None
    train_paths: tuple[Path, ...] = ()
    test_paths: tuple[Path, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Fedot.Industrial from ~/Fedot.Industrial on diploma_mixed "
            "OpenML tasks and/or generated M4 frequency classification folds."
        )
    )
    parser.add_argument(
        "--suite",
        choices=["all", "diploma", "m4"],
        default="all",
        help="Task suite to run. Default: all.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=default_benchmark_root(),
        help="Path to automlbenchmark checkout. Default: DIPLOMA_BENCHMARK_ROOT or this repo.",
    )
    parser.add_argument(
        "--fedot-root",
        type=Path,
        default=default_fedot_root(),
        help="Path to Fedot.Industrial checkout. Default: FEDOT_INDUSTRIAL_ROOT or ~/Fedot.Industrial.",
    )
    parser.add_argument(
        "--fedot-python",
        default=os.environ.get("FEDOT_INDUSTRIAL_PYTHON"),
        help=(
            "Python executable with Fedot.Industrial dependencies. "
            "Default: auto re-run through `poetry run python` in --fedot-root."
        ),
    )
    parser.add_argument(
        "--no-poetry-reexec",
        action="store_true",
        help="Do not auto re-run through the Fedot.Industrial Poetry environment.",
    )
    parser.add_argument(
        "--benchmark-yaml",
        type=Path,
        default=None,
        help="Diploma benchmark YAML. Default: <benchmark-root>/resources/benchmarks/diploma_mixed.yaml.",
    )
    parser.add_argument(
        "--part",
        choices=["all", "classification", "regression"],
        default="all",
        help="Diploma task subset. M4 is classification and is skipped for --part regression.",
    )
    parser.add_argument(
        "--task",
        nargs="*",
        default=None,
        help="Explicit task names. Overrides --part filtering.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        nargs="*",
        default=[0],
        help="Fold numbers to run. Default: 0. Pass --fold without values to run all folds.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=None,
        help="Fedot.Industrial timeout per task/fold. Default: ceil(max_runtime_seconds / 60).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("FEDOT_N_JOBS") or os.environ.get("N_JOBS") or 0),
        help="Fedot.Industrial n_jobs. 0 means use all available backend workers.",
    )
    parser.add_argument(
        "--industrial-strategy",
        default="default",
        help="Fedot.Industrial strategy name to place in industrial_config. Default: default.",
    )
    parser.add_argument(
        "--strategy-param",
        action="append",
        default=[],
        help="Extra industrial strategy parameter as key=value. Can be repeated.",
    )
    parser.add_argument(
        "--logging-level",
        type=int,
        default=50,
        help="Fedot logging level. Default: 50 (critical).",
    )
    parser.add_argument(
        "--openml-cache",
        type=Path,
        default=None,
        help="Optional OpenML cache directory.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Default: <benchmark-root>/results/fedot_industrial/<timestamp>.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Write y_true/y_pred CSV files for every successful fold.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining folds after a failure.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected task/fold jobs and exit.",
    )

    parser.add_argument(
        "--m4-output-dir",
        type=Path,
        default=Path("data/m4_frequency_classification"),
        help="Where M4 artifacts/folds/YAML are stored. Default: data/m4_frequency_classification.",
    )
    parser.add_argument(
        "--m4-groups",
        nargs="+",
        default=["Yearly", "Monthly", "Quarterly", "Daily"],
        help="M4 groups to classify. Default: Yearly Monthly Quarterly Daily.",
    )
    parser.add_argument(
        "--m4-n-per-group",
        type=int,
        default=0,
        help="Rows per M4 group. Use 0 or a negative value for all rows. Default: all rows.",
    )
    parser.add_argument(
        "--m4-window-length",
        type=int,
        default=50,
        help="Fixed M4 feature length. Use 0 for full loaded history. Default: 50.",
    )
    parser.add_argument(
        "--m4-no-standardize",
        action="store_true",
        help="Disable per-series z-score standardization for generated M4 rows.",
    )
    parser.add_argument(
        "--m4-folds",
        type=int,
        default=2,
        help="Number of stratified M4 folds to export. Default: 2.",
    )
    parser.add_argument(
        "--m4-test-size",
        type=float,
        default=0.2,
        help="M4 test fraction per class. Default: 0.2.",
    )
    parser.add_argument(
        "--m4-seed",
        type=int,
        default=42,
        help="M4 artifact/fold random seed. Default: 42.",
    )
    parser.add_argument(
        "--m4-force",
        action="store_true",
        help="Recreate generated M4 artifacts and fold CSV files even if they exist.",
    )
    parser.add_argument(
        "--m4-chunk-rows",
        type=int,
        default=512,
        help="CSV chunk size while reading M4 source files. Default: 512.",
    )
    parser.add_argument(
        "--m4-max-runtime-seconds",
        type=int,
        default=600,
        help="Runtime budget stored for the M4 task. Default: 600.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_benchmark_root() -> Path:
    env_path = os.environ.get("DIPLOMA_BENCHMARK_ROOT")
    return Path(env_path).expanduser() if env_path else repo_root()


def default_fedot_root() -> Path:
    env_path = os.environ.get("FEDOT_INDUSTRIAL_ROOT")
    if env_path:
        return Path(env_path).expanduser()
    home_path = Path.home() / "Fedot.Industrial"
    if home_path.exists() or os.name != "nt":
        return home_path
    windows_path = Path(r"D:\Diploma\Fedot.Industrial")
    return windows_path if windows_path.exists() else home_path


def default_outdir(benchmark_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return benchmark_root / "results" / "fedot_industrial" / stamp


def configure_fedot_root(fedot_root: Path) -> None:
    root = fedot_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Fedot.Industrial checkout not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def ensure_fedot_runtime(args: argparse.Namespace) -> None:
    configure_fedot_root(args.fedot_root)
    try:
        import fedot  # noqa: F401
        import fedot_ind.api.main  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        if args.no_poetry_reexec or os.environ.get("FEDOT_INDUSTRIAL_REEXEC"):
            raise SystemExit(
                "Fedot.Industrial dependencies are not importable in this Python "
                f"({sys.executable}): {exc}. Run from the Poetry environment or pass "
                "`--fedot-python $(cd ~/Fedot.Industrial && poetry env info --executable)`."
            ) from exc
        reexec_in_fedot_env(args, exc)


def reexec_in_fedot_env(args: argparse.Namespace, reason: BaseException) -> None:
    fedot_root = args.fedot_root.expanduser().resolve()
    script_path = Path(__file__).resolve()
    env = clean_poetry_env()
    env["FEDOT_INDUSTRIAL_REEXEC"] = "1"
    env["FEDOT_INDUSTRIAL_ROOT"] = str(fedot_root)
    env["DIPLOMA_BENCHMARK_ROOT"] = str(args.benchmark_root)
    python_path_entries = [str(repo_root()), str(fedot_root)]
    if env.get("PYTHONPATH"):
        python_path_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries)

    explicit_python = Path(args.fedot_python).expanduser() if args.fedot_python else None
    if explicit_python and explicit_python.resolve() != Path(sys.executable).resolve():
        cmd = [str(explicit_python), str(script_path), *sys.argv[1:]]
        cwd = repo_root()
        launcher_text = cmd[0]
    else:
        poetry_python = resolve_poetry_python(fedot_root, env)
        poetry = find_poetry(env)
        if poetry_python is None and poetry is None:
            raise SystemExit(
                "Fedot.Industrial dependencies are not importable in the current venv "
                f"({reason}), and `poetry` was not found on PATH. Either run:\n"
                f"  cd {fedot_root} && env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry run python {script_path} {' '.join(sys.argv[1:])}\n"
                "or pass a real Fedot.Industrial Poetry interpreter from "
                "`env -u VIRTUAL_ENV -u POETRY_ACTIVE poetry env info --executable`."
            )
        if poetry_python is not None:
            cmd = [str(poetry_python), str(script_path), *strip_fedot_python_arg(sys.argv[1:])]
            launcher_text = str(poetry_python)
        else:
            cmd = [str(poetry), "run", "python", str(script_path), *strip_fedot_python_arg(sys.argv[1:])]
            launcher_text = f"{poetry} run python"
        cwd = fedot_root

    print(
        "Fedot.Industrial dependencies are not in the current Python; "
        f"re-running via {launcher_text} from {fedot_root}."
    )
    completed = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    raise SystemExit(int(completed.returncode))


def clean_poetry_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "VIRTUAL_ENV",
        "POETRY_ACTIVE",
        "__PYVENV_LAUNCHER__",
        "PYTHONHOME",
    ):
        env.pop(key, None)
    path_parts = [
        part
        for part in env.get("PATH", "").split(os.pathsep)
        if part and Path(part).resolve() != Path(sys.executable).resolve().parent
    ]
    env["PATH"] = os.pathsep.join(path_parts)
    return env


def resolve_poetry_python(fedot_root: Path, env: dict[str, str]) -> Path | None:
    poetry = find_poetry(env)
    if poetry is None:
        return None
    completed = subprocess.run(
        [poetry, "env", "info", "--executable"],
        cwd=fedot_root,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    candidate = completed.stdout.strip()
    if completed.returncode == 0 and candidate:
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists() and candidate_path.resolve() != Path(sys.executable).resolve():
            return candidate_path
    return None


def find_poetry(env: dict[str, str]) -> str | None:
    return shutil.which("poetry", path=env.get("PATH")) or shutil.which("poetry")


def strip_fedot_python_arg(argv: Sequence[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == "--fedot-python":
            skip_next = True
            continue
        if item.startswith("--fedot-python="):
            continue
        stripped.append(item)
    return stripped


def load_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return parse_simple_amlb_yaml(text)


def parse_simple_amlb_yaml(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or line.strip() == "---":
            continue
        stripped = line.strip()
        if stripped.startswith("- name:"):
            current = {"name": parse_yaml_scalar(stripped.split(":", 1)[1].strip())}
            items.append(current)
            continue
        if current is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()
        if value:
            current[key.strip()] = parse_yaml_scalar(value)
    return items


def parse_yaml_scalar(value: str) -> Any:
    value = value.strip().strip("'\"")
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_yaml_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        return value


def load_diploma_tasks(args: argparse.Namespace) -> list[BenchmarkTask]:
    benchmark_yaml = args.benchmark_yaml or (
        args.benchmark_root / "resources" / "benchmarks" / "diploma_mixed.yaml"
    )
    raw = load_yaml(benchmark_yaml)
    defaults: dict[str, Any] = {"folds": 1, "max_runtime_seconds": 600}
    tasks: list[BenchmarkTask] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        if item.get("name") == "__defaults__":
            defaults.update(item)
            continue
        if "openml_task_id" not in item:
            continue
        name = str(item["name"])
        metrics = item.get("metric", ())
        if isinstance(metrics, str):
            metrics = [metrics]
        task_type = "classification" if name in CLASSIFICATION_TASKS or name.endswith("_classification") else "regression"
        tasks.append(
            BenchmarkTask(
                name=name,
                benchmark="diploma_mixed",
                task_type=task_type,
                metrics=tuple(str(metric) for metric in metrics),
                folds=int(item.get("folds", defaults.get("folds", 1))),
                max_runtime_seconds=int(
                    item.get("max_runtime_seconds", defaults.get("max_runtime_seconds", 600))
                ),
                openml_task_id=int(item["openml_task_id"]),
            )
        )
    return tasks


def load_m4_task(args: argparse.Namespace) -> BenchmarkTask:
    if args.dry_run:
        return BenchmarkTask(
            name=M4_TASK_NAME,
            benchmark=M4_TASK_NAME,
            task_type="classification",
            metrics=("acc", "f1", "logloss", "balacc"),
            folds=args.m4_folds,
            max_runtime_seconds=args.m4_max_runtime_seconds,
            target=M4_TARGET_COLUMN,
        )

    import argparse as argparse_module
    import run_m4_classification_frameworks as m4_runner

    output_dir = args.m4_output_dir
    if not output_dir.is_absolute():
        output_dir = args.benchmark_root / output_dir
    m4_args = argparse_module.Namespace(
        groups=args.m4_groups,
        n_per_group=args.m4_n_per_group,
        window_length=args.m4_window_length,
        no_standardize=args.m4_no_standardize,
        folds=args.m4_folds,
        test_size=args.m4_test_size,
        seed=args.m4_seed,
        output_dir=output_dir,
        force=args.m4_force,
        chunk_rows=args.m4_chunk_rows,
    )
    groups = m4_runner.normalize_groups(args.m4_groups)
    metadata = m4_runner.create_m4_artifact(m4_args, groups)
    _, split_rows = m4_runner.create_amlb_folds(m4_args, metadata)
    return BenchmarkTask(
        name=M4_TASK_NAME,
        benchmark=M4_TASK_NAME,
        task_type="classification",
        metrics=("acc", "f1", "logloss", "balacc"),
        folds=args.m4_folds,
        max_runtime_seconds=args.m4_max_runtime_seconds,
        target=M4_TARGET_COLUMN,
        train_paths=tuple(Path(row["train_path"]) for row in split_rows),
        test_paths=tuple(Path(row["test_path"]) for row in split_rows),
    )


def collect_tasks(args: argparse.Namespace) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    include_diploma = args.suite in {"all", "diploma"}
    include_m4 = args.suite in {"all", "m4"}
    if include_diploma:
        tasks.extend(load_diploma_tasks(args))
    if include_m4 and (args.part != "regression" or task_filter_includes_m4(args.task)):
        tasks.append(load_m4_task(args))
    return select_tasks(tasks, args)


def task_filter_includes_m4(task_names: list[str] | None) -> bool:
    return bool(task_names and M4_TASK_NAME in set(task_names))


def select_tasks(tasks: list[BenchmarkTask], args: argparse.Namespace) -> list[BenchmarkTask]:
    if args.task:
        wanted = set(args.task)
        selected = [task for task in tasks if task.name in wanted]
        missing = sorted(wanted - {task.name for task in selected})
        if missing:
            raise ValueError(f"Unknown benchmark task(s): {', '.join(missing)}")
        return selected
    if args.part == "classification":
        return [task for task in tasks if task.task_type == "classification"]
    if args.part == "regression":
        return [task for task in tasks if task.task_type == "regression"]
    return tasks


def selected_folds(task: BenchmarkTask, explicit_folds: list[int] | None) -> list[int]:
    folds = list(range(task.folds)) if not explicit_folds else list(explicit_folds)
    invalid = [fold for fold in folds if fold < 0 or fold >= task.folds]
    if invalid:
        raise ValueError(f"{task.name} has folds [0, {task.folds - 1}], got {invalid}")
    return folds


def configure_openml(cache_dir: Path | None) -> None:
    try:
        import openml
    except ImportError as exc:
        raise SystemExit(
            "The `openml` package is required for diploma tasks. Install it into this venv."
        ) from exc

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        setter = getattr(openml.config, "set_cache_directory", None) or getattr(
            openml.config,
            "set_root_cache_directory",
            None,
        )
        if setter is not None:
            setter(str(cache_dir))
    try:
        openml.config.set_retry_policy("robot")
    except Exception:
        pass


def load_openml_fold(task: BenchmarkTask, fold: int) -> tuple[Any, Any, Any, Any]:
    import openml
    import pandas as pd

    if task.openml_task_id is None:
        raise ValueError(f"{task.name} does not have openml_task_id")
    openml_task = openml.tasks.get_task(task.openml_task_id, download_data=True)
    dataset = openml_task.get_dataset()
    target_name = openml_task.target_name[0] if isinstance(openml_task.target_name, list) else openml_task.target_name
    X, y, _, _ = dataset.get_data(target=target_name, dataset_format="dataframe")
    train_idx, test_idx = openml_task.get_train_test_split_indices(fold, repeat=0, sample=0)
    y_series = pd.Series(y)
    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y_series.iloc[train_idx].reset_index(drop=True),
        y_series.iloc[test_idx].reset_index(drop=True),
    )


def load_csv_fold(task: BenchmarkTask, fold: int) -> tuple[Any, Any, Any, Any]:
    import pandas as pd

    if not task.train_paths or not task.test_paths or task.target is None:
        raise ValueError(f"{task.name} does not have CSV fold paths")
    train = pd.read_csv(task.train_paths[fold])
    test = pd.read_csv(task.test_paths[fold])
    if task.target not in train.columns:
        raise ValueError(f"{task.train_paths[fold]} is missing target {task.target!r}")
    if task.target not in test.columns:
        raise ValueError(f"{task.test_paths[fold]} is missing target {task.target!r}")
    return (
        train.drop(columns=[task.target]).reset_index(drop=True),
        test.drop(columns=[task.target]).reset_index(drop=True),
        train[task.target].reset_index(drop=True),
        test[task.target].reset_index(drop=True),
    )


def load_fold(task: BenchmarkTask, fold: int) -> tuple[Any, Any, Any, Any]:
    if task.openml_task_id is not None:
        return load_openml_fold(task, fold)
    return load_csv_fold(task, fold)


def encode_features(X_train: Any, X_test: Any) -> tuple[Any, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer

    train = pd.DataFrame(X_train).copy()
    test = pd.DataFrame(X_test).copy()
    for column in train.columns:
        if is_categorical(train[column]):
            train_values = train[column].astype("string").fillna("__missing__")
            test_values = test[column].astype("string").fillna("__missing__")
            categories = {value: index for index, value in enumerate(pd.unique(train_values))}
            train[column] = train_values.map(categories).astype(float)
            test[column] = test_values.map(categories).fillna(-1).astype(float)
        else:
            train[column] = pd.to_numeric(train[column], errors="coerce")
            test[column] = pd.to_numeric(test[column], errors="coerce")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train)
    x_test = imputer.transform(test.reindex(columns=train.columns))
    return np.asarray(x_train, dtype=float), np.asarray(x_test, dtype=float)


def is_categorical(series: Any) -> bool:
    import pandas as pd

    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )


def encode_target(y_train: Any, y_test: Any, task_type: str) -> tuple[Any, Any, int]:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    if task_type == "classification":
        encoder = LabelEncoder()
        encoder.fit(pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True).astype(str))
        return (
            encoder.transform(pd.Series(y_train).astype(str)),
            encoder.transform(pd.Series(y_test).astype(str)),
            len(encoder.classes_),
        )
    return (
        pd.Series(y_train).astype(float).to_numpy(),
        pd.Series(y_test).astype(float).to_numpy(),
        0,
    )


def build_api_config(task: BenchmarkTask, fold: int, outdir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from fedot_ind.core.repository.config_repository import (
        DEFAULT_CLF_API_CONFIG,
        DEFAULT_REG_API_CONFIG,
    )

    api_config = deepcopy(DEFAULT_CLF_API_CONFIG if task.task_type == "classification" else DEFAULT_REG_API_CONFIG)
    problem = "classification" if task.task_type == "classification" else "regression"
    timeout = args.timeout_minutes or max(1, math.ceil(task.max_runtime_seconds / 60))
    n_jobs = int(args.n_jobs) if int(args.n_jobs) != 0 else -1

    api_config["industrial_config"]["problem"] = problem
    strategy = str(args.industrial_strategy or "default").strip().lower() or "default"
    if strategy != "default":
        api_config["industrial_config"]["strategy"] = strategy
        api_config["industrial_config"]["learning_strategy"] = strategy
        api_config["industrial_config"]["strategy_params"] = {
            "problem": problem,
            "data_type": "table",
            "timeout": timeout,
            "n_jobs": n_jobs,
            **parse_strategy_params(args.strategy_param),
        }

    learning_params = api_config["learning_config"].setdefault("learning_strategy_params", {})
    learning_params.update(
        {
            "timeout": timeout,
            "n_jobs": n_jobs,
            "logging_level": int(args.logging_level),
        }
    )
    api_config["learning_config"]["optimisation_loss"] = {"quality_loss": primary_metric(task)}
    api_config["automl_config"]["task"] = problem

    output_folder = outdir / "artifacts" / task.name / str(fold)
    output_folder.mkdir(parents=True, exist_ok=True)
    compute_config = api_config["compute_config"]
    compute_config["output_folder"] = str(output_folder)
    compute_config["automl_folder"] = {
        "optimisation_history": str(output_folder / "opt_hist"),
        "composition_results": str(output_folder / "comp_res"),
    }
    distributed = compute_config.get("distributed")
    if isinstance(distributed, dict):
        distributed["n_workers"] = 1
        distributed["threads_per_worker"] = max(1, n_jobs if n_jobs > 0 else 1)
    return api_config


def parse_strategy_params(items: Iterable[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --strategy-param {item!r}; expected key=value.")
        key, value = item.split("=", 1)
        params[key.strip()] = parse_scalar(value.strip())
    return params


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def primary_metric(task: BenchmarkTask) -> str:
    if task.task_type == "classification":
        return "accuracy" if "acc" in task.metrics else "f1"
    return "rmse" if "rmse" in task.metrics else "r2"


def fit_predict(task: BenchmarkTask, fold: int, outdir: Path, args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from fedot_ind.api.main import FedotIndustrial

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = load_fold(task, fold)
    X_train, X_test = encode_features(X_train_raw, X_test_raw)
    y_train, y_test, n_classes = encode_target(y_train_raw, y_test_raw, task.task_type)
    api_config = build_api_config(task, fold, outdir, args)
    model = FedotIndustrial(**api_config)
    try:
        model.fit(input_data=(X_train, y_train))
        y_pred = np.asarray(model.predict((X_test, y_test))).reshape(-1)
        y_proba = None
        if task.task_type == "classification":
            try:
                y_proba = np.asarray(model.predict_proba((X_test, y_test)))
            except Exception:
                y_proba = None
        metrics = compute_task_metrics(task, y_test, y_pred, y_proba, n_classes)
    finally:
        shutdown = getattr(model, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
    return {
        "metrics": metrics,
        "y_true": y_test,
        "y_pred": y_pred,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
    }


def compute_task_metrics(
    task: BenchmarkTask,
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None,
    n_classes: int,
) -> dict[str, float | None]:
    import numpy as np
    from sklearn import metrics as skm
    from sklearn.preprocessing import LabelEncoder

    if task.task_type == "classification":
        pred = np.asarray(y_pred).reshape(-1)
        if np.issubdtype(pred.dtype, np.number):
            pred = np.rint(pred.astype(float)).astype(int)
        else:
            pred = LabelEncoder().fit_transform(pred.astype(str))
        if n_classes > 0:
            pred = np.clip(pred, 0, n_classes - 1)
        true = np.asarray(y_true).reshape(-1).astype(int)
        proba = normalize_proba(y_proba, n_classes)
        values: dict[str, float | None] = {
            "acc": float(skm.accuracy_score(true, pred)),
            "f1": float(skm.f1_score(true, pred, average="weighted", zero_division=0)),
            "balacc": float(skm.balanced_accuracy_score(true, pred)),
            "auc": None,
            "auc_ovr": None,
            "logloss": None,
        }
        if proba is not None:
            try:
                if n_classes == 2:
                    values["auc"] = float(skm.roc_auc_score(true, proba[:, 1]))
                else:
                    labels = np.arange(n_classes)
                    values["auc_ovr"] = float(
                        skm.roc_auc_score(true, proba, multi_class="ovr", labels=labels)
                    )
            except Exception:
                pass
            try:
                values["logloss"] = float(skm.log_loss(true, proba, labels=np.arange(n_classes)))
            except Exception:
                pass
        return {metric: values.get(metric) for metric in task.metrics}

    true_reg = np.asarray(y_true, dtype=float).reshape(-1)
    pred_reg = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(true_reg), len(pred_reg))
    true_reg, pred_reg = true_reg[:n], pred_reg[:n]
    values = {
        "rmse": float(np.sqrt(skm.mean_squared_error(true_reg, pred_reg))),
        "r2": float(skm.r2_score(true_reg, pred_reg)),
        "mae": float(skm.mean_absolute_error(true_reg, pred_reg)),
    }
    return {metric: values.get(metric) for metric in task.metrics}


def normalize_proba(y_proba: Any | None, n_classes: int) -> Any | None:
    import numpy as np

    if y_proba is None or n_classes <= 1:
        return None
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1 and n_classes == 2:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.ndim != 2 or proba.shape[1] != n_classes:
        return None
    proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = proba.sum(axis=1, keepdims=True)
    bad_rows = row_sum.squeeze() <= 0
    proba = proba / np.where(row_sum > 0, row_sum, 1.0)
    if np.any(bad_rows):
        proba[bad_rows] = 1.0 / n_classes
    return proba


def metric_rows(
    task: BenchmarkTask,
    fold: int,
    status: str,
    duration: float,
    result: dict[str, Any] | None = None,
    error: str = "",
) -> list[dict[str, Any]]:
    result = result or {}
    metrics = result.get("metrics") or {metric: None for metric in task.metrics}
    rows = []
    for metric in task.metrics:
        rows.append(
            {
                "framework": "FedotIndustrial",
                "benchmark": task.benchmark,
                "task": task.name,
                "task_type": task.task_type,
                "openml_task_id": task.openml_task_id or "",
                "fold": fold,
                "status": status,
                "duration_seconds": round(duration, 3),
                "train_rows": result.get("train_rows", ""),
                "test_rows": result.get("test_rows", ""),
                "n_features": result.get("n_features", ""),
                "metric": metric,
                "metric_value": metrics.get(metric),
                "error": error,
            }
        )
    return rows


def append_rows(csv_path: Path, jsonl_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    row_list = list(rows)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(row_list)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for row in row_list:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_predictions(outdir: Path, task: BenchmarkTask, fold: int, result: dict[str, Any]) -> None:
    import numpy as np
    import pandas as pd

    predictions_dir = outdir / "predictions" / task.name / str(fold)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(result["y_true"]).reshape(-1)
    y_pred = np.asarray(result["y_pred"]).reshape(-1)[: len(y_true)]
    pd.DataFrame(
        {
            "row_index": np.arange(len(y_true)),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(predictions_dir / "predictions.csv", index=False)


def print_jobs(tasks: list[BenchmarkTask], args: argparse.Namespace) -> None:
    for task in tasks:
        print(
            f"{task.benchmark}/{task.name}: type={task.task_type}, "
            f"folds={selected_folds(task, args.fold)}, metrics={list(task.metrics)}"
        )


def main() -> int:
    args = parse_args()
    args.benchmark_root = args.benchmark_root.expanduser().resolve()
    tasks = collect_tasks(args)
    if args.dry_run:
        print_jobs(tasks, args)
        return 0

    ensure_fedot_runtime(args)
    if any(task.openml_task_id is not None for task in tasks):
        configure_openml(args.openml_cache)
    outdir = args.outdir or default_outdir(args.benchmark_root)
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"
    jsonl_path = outdir / "results.jsonl"
    failures = 0

    for task in tasks:
        for fold in selected_folds(task, args.fold):
            started = time.monotonic()
            print(f"Running Fedot.Industrial: benchmark={task.benchmark} task={task.name} fold={fold}")
            try:
                result = fit_predict(task, fold, outdir, args)
                duration = time.monotonic() - started
                rows = metric_rows(task, fold, "ok", duration, result=result)
                append_rows(csv_path, jsonl_path, rows)
                if args.save_predictions:
                    save_predictions(outdir, task, fold, result)
                metric_text = ", ".join(f"{row['metric']}={row['metric_value']}" for row in rows)
                print(f"OK {task.name} fold={fold}: {metric_text}")
            except Exception as exc:
                failures += 1
                duration = time.monotonic() - started
                err = repr(exc)
                append_rows(csv_path, jsonl_path, metric_rows(task, fold, "failed", duration, error=err))
                (outdir / f"{task.name}_fold{fold}_traceback.txt").write_text(
                    traceback.format_exc(),
                    encoding="utf-8",
                )
                print(f"FAILED {task.name} fold={fold}: {err}", file=sys.stderr)
                if not args.continue_on_error:
                    print(f"Results written to {outdir}")
                    return 1

    print(f"Results written to {outdir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

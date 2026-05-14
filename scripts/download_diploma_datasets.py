#!/usr/bin/env python
"""
Download the OpenML datasets used by resources/benchmarks/diploma_mixed.yaml
and materialize the exact OpenML train/test folds used by AMLB.

The script writes one train/test pair per dataset and fold, plus a summary CSV
with split sizes and target column names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import openml
except ImportError as e:
    raise SystemExit(
        "Missing dependency: openml. Install project dependencies first:\n"
        "  python -m pip install -r requirements.txt"
    ) from e

try:
    from ruamel.yaml import YAML
except ImportError as e:
    raise SystemExit(
        "Missing dependency: ruamel.yaml. Install project dependencies first:\n"
        "  python -m pip install -r requirements.txt"
    ) from e


DEFAULT_BENCHMARK = Path("resources/benchmarks/diploma_mixed.yaml")
DEFAULT_OUTPUT_DIR = Path("data/diploma_mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download diploma_mixed OpenML train/test folds."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help=f"Benchmark YAML path. Default: {DEFAULT_BENCHMARK}",
    )
    parser.add_argument(
        "--output-dir",
        "--data-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for materialized train/test split files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Fold numbers to export. Default: folds from benchmark __defaults__.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output file format. Default: csv.",
    )
    parser.add_argument(
        "--openml-cache",
        "--download-dir",
        type=Path,
        default=None,
        help="Optional OpenML download/cache directory for raw OpenML files.",
    )
    parser.add_argument(
        "--keep-original-column-order",
        action="store_true",
        help="Do not move the target column to the end of the exported files.",
    )
    return parser.parse_args()


def load_benchmark(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    yaml = YAML(typ="safe")
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.load(f)

    defaults: dict[str, Any] = {}
    tasks: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("name") == "__defaults__":
            defaults = entry
            continue
        if entry.get("enabled", True) is False:
            continue
        if entry.get("openml_task_id") is None:
            continue
        tasks.append(entry)
    return defaults, tasks


def target_name_from_task(task: Any) -> str:
    if getattr(task, "target_name", None):
        return task.target_name

    for item in getattr(task, "input", []):
        if getattr(item, "name", None) == "source_data":
            data_set = getattr(item, "data_set", None)
            if data_set and getattr(data_set, "target_feature", None):
                return data_set.target_feature

    raise ValueError(f"Could not determine target column for OpenML task {task.task_id}.")


def problem_type_from_task(task: Any) -> str:
    task_type = getattr(task, "task_type", "").lower()
    if "regression" in task_type:
        return "regression"
    labels = getattr(task, "class_labels", None) or []
    return "binary" if len(labels) == 2 else "multiclass"


def load_task_dataframe(task: Any, target_name: str) -> pd.DataFrame:
    dataset = task.get_dataset()
    data, *_ = dataset.get_data(dataset_format="dataframe")

    if target_name not in data.columns:
        X, y, *_ = dataset.get_data(dataset_format="dataframe", target=target_name)
        data = X.copy()
        data[target_name] = y

    ignored = set(dataset.ignore_attribute or []) | set(dataset.row_id_attribute or [])
    ignored.discard(target_name)
    if ignored:
        data = data.drop(columns=[c for c in ignored if c in data.columns])
    return data


def split_indices(task: Any, fold: int) -> tuple[Any, Any]:
    # AMLB calls this method with the fold as the first positional argument.
    return task.get_train_test_split_indices(fold)


def save_frame(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def main() -> int:
    args = parse_args()
    if args.openml_cache is not None:
        args.openml_cache.mkdir(parents=True, exist_ok=True)
        try:
            openml.config.set_cache_directory(str(args.openml_cache))
        except AttributeError:
            openml.config.set_root_cache_directory(str(args.openml_cache))

    defaults, task_defs = load_benchmark(args.benchmark)
    folds = args.folds if args.folds is not None else list(range(int(defaults.get("folds", 2))))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for task_def in task_defs:
        task_name = task_def["name"]
        task_id = int(task_def["openml_task_id"])
        metrics = task_def.get("metric", [])

        print(f"Downloading {task_name} (OpenML task {task_id})...")
        task = openml.tasks.get_task(task_id)
        target_name = target_name_from_task(task)
        problem_type = problem_type_from_task(task)
        data = load_task_dataframe(task, target_name)

        if not args.keep_original_column_order:
            columns = [c for c in data.columns if c != target_name] + [target_name]
            data = data.loc[:, columns]

        task_dir = args.output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "task_name": task_name,
            "openml_task_id": task_id,
            "openml_dataset_id": getattr(task.get_dataset(), "dataset_id", None),
            "problem_type": problem_type,
            "target": target_name,
            "metrics": metrics,
            "rows": int(len(data)),
            "columns": list(data.columns),
            "folds_exported": folds,
        }
        (task_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for fold in folds:
            train_idx, test_idx = split_indices(task, fold)
            train = data.iloc[train_idx].reset_index(drop=True)
            test = data.iloc[test_idx].reset_index(drop=True)

            suffix = args.format
            train_path = task_dir / f"fold_{fold}" / f"train.{suffix}"
            test_path = task_dir / f"fold_{fold}" / f"test.{suffix}"
            save_frame(train, train_path, args.format)
            save_frame(test, test_path, args.format)

            total = len(train) + len(test)
            summary_rows.append(
                {
                    "task": task_name,
                    "openml_task_id": task_id,
                    "problem_type": problem_type,
                    "fold": fold,
                    "target": target_name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "total_rows": total,
                    "train_percent": round(100 * len(train) / total, 4),
                    "test_percent": round(100 * len(test) / total, 4),
                    "metrics": ",".join(metrics),
                    "train_path": str(train_path),
                    "test_path": str(test_path),
                }
            )
            print(
                f"  fold {fold}: train={len(train)} "
                f"({100 * len(train) / total:.2f}%), "
                f"test={len(test)} ({100 * len(test) / total:.2f}%)"
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nDone. Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

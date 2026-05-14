#!/usr/bin/env python
"""
Prepare M4 frequency-group classification data and run AMLB frameworks on it.

Dataset construction follows the idea from
industrial-learning-agent/backend/benchmarks/m4_benchmark.py:
each M4 time series becomes one fixed-length row, the class label is the M4
frequency group, and the artifact is stored as X.npy, y.npy, manifest JSON,
and a small preview CSV.

For AMLB frameworks, the script also exports stratified CSV train/test folds
and creates a generated benchmark YAML that points to those folds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


M4_GROUPS = ("Daily", "Weekly", "Monthly", "Quarterly", "Yearly")
DEFAULT_GROUPS = ("Yearly", "Monthly", "Quarterly", "Daily")
DEFAULT_FRAMEWORKS = (
    "AutoGluon",
    "flaml",
    "H2OAutoML",
    "lightautoml",
    "mljarsupervised",
    "TPOT",
    "RandomForest",
)
M4_SOURCE_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
M4_TARGET_COLUMN = "frequency_group"
M4_TASK_TYPE = "ts_classification"
M4_ARTIFACT_KIND = "mafis_m4_numpy_v1"
PREVIEW_ROWS = 10
PREVIEW_COLS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AMLB frameworks on M4 frequency-group classification."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="M4 groups to classify. Default: Yearly Monthly Quarterly Daily.",
    )
    parser.add_argument(
        "--n-per-group",
        type=int,
        default=0,
        help="Rows per M4 group. Use 0 or a negative value for all rows. Default: all rows.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=50,
        help="Fixed feature length. Use 0 for full loaded history. Default: 50.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable per-series z-score standardization.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=2,
        help="Number of stratified train/test folds to export. Default: 2.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Stratified test fraction per class for generated folds. Default: 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for artifact shuffling and folds. Default: 42.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/m4_frequency_classification"),
        help="Where to write artifacts, folds, and generated YAML.",
    )
    parser.add_argument(
        "--framework",
        default="all",
        help="Framework name, 'all', or comma-separated list. Default: all.",
    )
    parser.add_argument(
        "--constraint",
        default="test",
        help="AMLB constraint name. Default: test.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "docker", "singularity", "aws"],
        default="local",
        help="AMLB run mode. Default: local.",
    )
    parser.add_argument(
        "--setup",
        choices=["auto", "skip", "force", "only"],
        default="auto",
        help="AMLB framework setup mode. Default: auto.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for runbenchmark.py. Default: current interpreter.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Optional AMLB output directory. Default: repo results/.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create the M4 artifact, folds, and benchmark YAML; do not run frameworks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate generated artifacts and fold CSV files even if they exist.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=512,
        help="CSV chunk size while reading M4 source files. Default: 512.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_groups(groups: Sequence[str]) -> tuple[str, ...]:
    aliases = {group.lower(): group for group in M4_GROUPS}
    aliases.update(
        {
            "yearli": "Yearly",
            "yearly": "Yearly",
            "daily": "Daily",
            "monthly": "Monthly",
            "quarterly": "Quarterly",
            "weekly": "Weekly",
        }
    )
    selected: list[str] = []
    for raw in groups:
        key = raw.strip().strip(",.;:").lower()
        if not key:
            continue
        if key not in aliases:
            allowed = ", ".join(M4_GROUPS)
            raise ValueError(f"Unknown M4 group '{raw}'. Allowed groups: {allowed}.")
        group = aliases[key]
        if group not in selected:
            selected.append(group)
    if len(selected) < 2:
        raise ValueError("At least two M4 groups are required for classification.")
    return tuple(selected)


def m4_csv_path(output_dir: Path, group: str) -> Path:
    return output_dir / "datasets" / f"{group}-train.csv"


def m4_train_url(group: str) -> str:
    return f"{M4_SOURCE_URL}/Train/{group}-train.csv"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "automlbenchmark-m4-classification/1.0"})
    try:
        with urlopen(request, timeout=120) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp_path.replace(destination)
    except (OSError, URLError) as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Could not download M4 file {url}: {exc}") from exc


def download_group_if_missing(output_dir: Path, group: str) -> None:
    csv_path = m4_csv_path(output_dir, group)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    if csv_path.exists():
        csv_path.unlink()
    print(f"Downloading {group} from M4...")
    download_file(m4_train_url(group), csv_path)


def iter_wide_group_chunks(
    output_dir: Path,
    group: str,
    max_rows: int | None,
    chunksize: int,
) -> Iterable[np.ndarray]:
    download_group_if_missing(output_dir, group)
    csv_path = m4_csv_path(output_dir, group)
    reader = pd.read_csv(
        csv_path,
        nrows=max_rows,
        chunksize=max(64, chunksize),
        low_memory=False,
    )
    for chunk in reader:
        id_col = chunk.columns[0]
        yield chunk.drop(columns=[id_col]).to_numpy(dtype=np.float32, copy=True)


def series_lengths(values: np.ndarray) -> np.ndarray:
    lengths = np.count_nonzero(~np.isnan(values), axis=1).astype(np.int64, copy=False)
    lengths[lengths == 0] = 1
    return lengths


def prepare_series(row: np.ndarray, window_length: int, standardize: bool) -> np.ndarray:
    series = row[~np.isnan(row)].astype(np.float32, copy=False)
    if series.size == 0:
        series = np.zeros(1, dtype=np.float32)
    if standardize and series.size > 1:
        std = float(series.std()) or 1.0
        series = (series - float(series.mean())) / std
    if series.size < window_length:
        pad = np.zeros(window_length - series.size, dtype=np.float32)
        return np.concatenate([pad, series])
    return series[-window_length:]


def prepare_series_batch(values: np.ndarray, window_length: int, standardize: bool) -> np.ndarray:
    return np.stack(
        [prepare_series(row, window_length, standardize) for row in values]
    ).astype(np.float32, copy=False)


def scan_m4_groups(
    output_dir: Path,
    groups: Sequence[str],
    n_per_group: int | None,
    chunksize: int,
) -> tuple[int, list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    total_rows = 0
    min_len: int | None = None
    max_len = 0
    sum_len = 0
    source_files: list[dict[str, Any]] = []
    class_balance: dict[str, int] = {}
    for group in groups:
        rows_read = 0
        for values in iter_wide_group_chunks(output_dir, group, n_per_group, chunksize):
            lengths = series_lengths(values)
            rows_read += int(values.shape[0])
            total_rows += int(values.shape[0])
            batch_min = int(lengths.min()) if lengths.size else 0
            batch_max = int(lengths.max()) if lengths.size else 0
            min_len = batch_min if min_len is None else min(min_len, batch_min)
            max_len = max(max_len, batch_max)
            sum_len += int(lengths.sum())
        csv_path = m4_csv_path(output_dir, group)
        class_balance[group] = rows_read
        source_files.append(
            {
                "group": group,
                "path": str(csv_path),
                "exists": csv_path.exists(),
                "size_mb": round(csv_path.stat().st_size / (1024 * 1024), 2)
                if csv_path.exists()
                else 0,
                "rows_read": rows_read,
            }
        )
    if total_rows == 0:
        raise ValueError("No M4 rows were loaded. Check groups and n_per_group.")
    length_info = {
        "min_series_length": int(min_len or 1),
        "max_series_length": int(max_len or 1),
        "mean_series_length": round(float(sum_len / total_rows), 2),
    }
    return total_rows, source_files, length_info, class_balance


def artifact_stem(
    groups: Sequence[str],
    n_per_group: int | None,
    window_length: int | None,
    standardize: bool,
    seed: int,
) -> str:
    groups_tag = "-".join(groups)
    n_tag = "all" if n_per_group is None else str(n_per_group)
    w_tag = "full" if window_length is None else str(window_length)
    std_tag = "std" if standardize else "raw"
    return f"m4_{groups_tag}_n{n_tag}_w{w_tag}_{std_tag}_seed{seed}"


def create_m4_artifact(args: argparse.Namespace, groups: tuple[str, ...]) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    n_per_group = None if args.n_per_group <= 0 else max(1, args.n_per_group)
    requested_window = None if args.window_length <= 0 else max(8, args.window_length)
    standardize = not args.no_standardize
    total_rows, source_files, scan_info, class_balance = scan_m4_groups(
        output_dir,
        groups,
        n_per_group,
        args.chunk_rows,
    )
    length = int(scan_info["max_series_length"] if requested_window is None else requested_window)
    stem = artifact_stem(groups, n_per_group, requested_window, standardize, args.seed)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / f"{stem}.json"
    x_path = artifact_dir / f"{stem}_X.npy"
    y_path = artifact_dir / f"{stem}_y.npy"
    preview_path = artifact_dir / f"{stem}_preview.csv"
    if args.force or not (manifest_path.exists() and x_path.exists() and y_path.exists()):
        print(
            f"Building M4 artifact: groups={list(groups)}, rows={total_rows}, "
            f"window_length={length}, standardize={standardize}"
        )
        x_mem = np.lib.format.open_memmap(
            x_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, length),
        )
        y_mem = np.lib.format.open_memmap(
            y_path,
            mode="w+",
            dtype="<U32",
            shape=(total_rows,),
        )
        output_order = np.random.default_rng(args.seed).permutation(total_rows)
        source_offset = 0
        for group in groups:
            for values in iter_wide_group_chunks(output_dir, group, n_per_group, args.chunk_rows):
                batch = prepare_series_batch(values, length, standardize)
                batch_size = int(batch.shape[0])
                output_idx = output_order[source_offset : source_offset + batch_size]
                x_mem[output_idx] = batch
                y_mem[output_idx] = group
                source_offset += batch_size
        x_mem.flush()
        y_mem.flush()
        preview_width = min(PREVIEW_COLS, length)
        preview_df = pd.DataFrame(
            np.asarray(x_mem[:PREVIEW_ROWS, :preview_width]),
            columns=[f"f_{i}" for i in range(preview_width)],
        )
        preview_df[M4_TARGET_COLUMN] = np.asarray(y_mem[:PREVIEW_ROWS]).astype(str)
        preview_df.to_csv(preview_path, index=False)
    else:
        print(f"Reusing existing M4 artifact: {manifest_path}")
    metadata = {
        "kind": M4_ARTIFACT_KIND,
        "storage_format": M4_ARTIFACT_KIND,
        "artifact_path": str(manifest_path),
        "x_path": str(x_path),
        "y_path": str(y_path),
        "preview_csv_path": str(preview_path),
        "target_column": M4_TARGET_COLUMN,
        "task_type": M4_TASK_TYPE,
        "n_samples": int(total_rows),
        "n_features": int(length),
        "dtype": "float32",
        "groups": list(groups),
        "group_labels": {str(i): group for i, group in enumerate(groups)},
        "window_length": int(length),
        "window_length_mode": "full_history" if requested_window is None else "fixed",
        "requested_window_length": requested_window,
        "n_per_group": n_per_group,
        "all_samples": n_per_group is None,
        "standardize": standardize,
        "min_series_length": scan_info["min_series_length"],
        "max_series_length": scan_info["max_series_length"],
        "mean_series_length": scan_info["mean_series_length"],
        "class_balance": class_balance,
        "source_files": source_files,
        "shuffled": True,
        "seed": args.seed,
    }
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def stratified_split_indices(
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        if idx.size < 2:
            raise ValueError(f"Class {label!r} has fewer than two samples.")
        rng.shuffle(idx)
        n_test = int(round(idx.size * test_size))
        n_test = min(max(1, n_test), idx.size - 1)
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def write_csv_split(
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    feature_columns: Sequence[str],
    chunk_size: int = 5000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for start in range(0, len(indices), chunk_size):
        batch_idx = indices[start : start + chunk_size]
        df = pd.DataFrame(np.asarray(X[batch_idx]), columns=feature_columns)
        df[M4_TARGET_COLUMN] = np.asarray(y[batch_idx]).astype(str)
        df.to_csv(path, mode="w" if first else "a", header=first, index=False)
        first = False


def create_amlb_folds(args: argparse.Namespace, metadata: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    output_dir = args.output_dir.resolve()
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    X = np.load(metadata["x_path"], mmap_mode="r")
    y = np.load(metadata["y_path"], mmap_mode="r")
    labels = np.asarray(y).astype(str)
    feature_columns = [f"f_{i}" for i in range(int(metadata["n_features"]))]
    split_rows: list[dict[str, Any]] = []
    train_paths: list[Path] = []
    test_paths: list[Path] = []
    for fold in range(args.folds):
        train_path = folds_dir / f"m4_frequency_classification_train_{fold}.csv"
        test_path = folds_dir / f"m4_frequency_classification_test_{fold}.csv"
        train_paths.append(train_path)
        test_paths.append(test_path)
        train_idx, test_idx = stratified_split_indices(labels, args.test_size, args.seed + fold)
        if args.force or not (train_path.exists() and test_path.exists()):
            print(
                f"Writing fold {fold}: train={len(train_idx)} "
                f"({100 * len(train_idx) / len(labels):.2f}%), "
                f"test={len(test_idx)} ({100 * len(test_idx) / len(labels):.2f}%)"
            )
            write_csv_split(train_path, X, y, train_idx, feature_columns)
            write_csv_split(test_path, X, y, test_idx, feature_columns)
        else:
            print(f"Reusing fold {fold}: {train_path.name}, {test_path.name}")
        split_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "total_rows": int(len(labels)),
                "train_percent": round(100 * len(train_idx) / len(labels), 4),
                "test_percent": round(100 * len(test_idx) / len(labels), 4),
                "train_path": str(train_path),
                "test_path": str(test_path),
            }
        )
    pd.DataFrame(split_rows).to_csv(output_dir / "fold_summary.csv", index=False)
    benchmark_path = output_dir / "m4_frequency_classification.yaml"
    write_benchmark_yaml(benchmark_path, train_paths, test_paths, args.folds)
    return benchmark_path, split_rows


def yaml_string(value: Path | str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def write_benchmark_yaml(
    path: Path,
    train_paths: Sequence[Path],
    test_paths: Sequence[Path],
    folds: int,
) -> None:
    lines = [
        "---",
        "",
        "- name: m4_frequency_classification",
        "  dataset:",
        "    train:",
    ]
    for train_path in train_paths:
        lines.append(f"      - {yaml_string(train_path)}")
    lines.append("    test:")
    for test_path in test_paths:
        lines.append(f"      - {yaml_string(test_path)}")
    lines.extend(
        [
            f"    target: {M4_TARGET_COLUMN}",
            "    type: multiclass",
            "  metric: [acc, f1, logloss, balacc]",
            f"  folds: {folds}",
            '  description: "M4 frequency-group classification from fixed-length time-series windows."',
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_frameworks(value: str) -> list[str]:
    if value.lower() == "all":
        return list(DEFAULT_FRAMEWORKS)
    frameworks = [item.strip() for item in value.split(",") if item.strip()]
    if not frameworks:
        raise ValueError("No frameworks selected.")
    return frameworks


def run_frameworks(args: argparse.Namespace, benchmark_path: Path) -> int:
    failures: list[str] = []
    for framework in parse_frameworks(args.framework):
        cmd = [
            args.python,
            "runbenchmark.py",
            framework,
            str(benchmark_path),
            args.constraint,
            "-m",
            args.mode,
            "-s",
            args.setup,
        ]
        if args.results_dir is not None:
            cmd.extend(["-o", str(args.results_dir)])
        print()
        print(
            f"===== Running {framework} on M4 classification "
            f"({args.constraint}, {args.mode}, setup={args.setup}) ====="
        )
        completed = subprocess.run(cmd, cwd=repo_root(), check=False)
        if completed.returncode == 0:
            print(f"===== {framework}: OK =====")
        else:
            print(f"===== {framework}: FAILED ({completed.returncode}) =====", file=sys.stderr)
            failures.append(framework)
    if failures:
        print(f"Failed frameworks: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    groups = normalize_groups(args.groups)
    if args.folds < 1:
        raise ValueError("--folds must be at least 1.")
    metadata = create_m4_artifact(args, groups)
    benchmark_path, split_rows = create_amlb_folds(args, metadata)
    print()
    print(f"Manifest: {metadata['artifact_path']}")
    print(f"Preview CSV: {metadata['preview_csv_path']}")
    print(f"Generated AMLB benchmark: {benchmark_path}")
    print(f"Fold summary: {args.output_dir.resolve() / 'fold_summary.csv'}")
    for row in split_rows:
        print(
            f"fold {row['fold']}: train={row['train_percent']}%, "
            f"test={row['test_percent']}%"
        )
    if args.prepare_only:
        return 0
    return run_frameworks(args, benchmark_path)


if __name__ == "__main__":
    raise SystemExit(main())

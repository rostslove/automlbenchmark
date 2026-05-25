#!/usr/bin/env python
"""Run agentic AutoML frameworks through the standard AMLB runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BENCHMARK = "diploma_mixed"
DEFAULT_FRAMEWORKS = (
    "AutoGluonAssistant",
    "AutoKaggle",
    "AIDE",
    "AutoMLAgent",
    "DSAgent",
)
CLASSIFICATION_TASKS = (
    "kc2_binary_classification",
    "iris_multiclass_classification",
    "credit_g_binary_classification",
)
REGRESSION_TASKS = (
    "cholesterol_regression",
    "autoMpg_regression",
    "kin8nm_regression",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run agentic frameworks on resources/benchmarks/diploma_mixed.yaml via runbenchmark.py."
    )
    parser.add_argument(
        "--framework",
        "-f",
        default="all",
        help="Framework name, 'all', or comma-separated list. Default: all.",
    )
    parser.add_argument(
        "--constraint",
        "-c",
        default="test",
        help="AMLB constraint name. Default: test.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["local", "docker", "singularity", "aws"],
        default="local",
        help="AMLB run mode. Default: local.",
    )
    parser.add_argument(
        "--part",
        "-p",
        choices=["all", "classification", "regression"],
        default="all",
        help="Task subset. Default: all.",
    )
    parser.add_argument(
        "--task",
        "-t",
        nargs="*",
        default=None,
        help="Explicit AMLB task names. Overrides --part.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        nargs="*",
        default=None,
        help="Optional fold numbers to run.",
    )
    parser.add_argument(
        "--setup",
        "-s",
        choices=["auto", "skip", "force", "only"],
        default="auto",
        help="AMLB setup mode. Default: auto.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke runbenchmark.py. Default: current interpreter.",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=Path,
        default=None,
        help="Optional AMLB output directory.",
    )
    parser.add_argument(
        "--extra",
        "-X",
        action="append",
        default=[],
        help="Extra runbenchmark.py override, e.g. -X f._repo=D:\\repo or -X f._provider=openai.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining frameworks after a failure.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_frameworks(value: str) -> list[str]:
    if value.lower() == "all":
        return list(DEFAULT_FRAMEWORKS)
    frameworks = [item.strip() for item in value.split(",") if item.strip()]
    if not frameworks:
        raise ValueError("No frameworks selected.")
    return frameworks


def task_args(args: argparse.Namespace) -> list[str]:
    if args.task is not None and len(args.task) > 0:
        return ["-t", *args.task]
    if args.part == "classification":
        return ["-t", *CLASSIFICATION_TASKS]
    if args.part == "regression":
        return ["-t", *REGRESSION_TASKS]
    return []


def fold_args(args: argparse.Namespace) -> list[str]:
    if args.fold is None:
        return []
    return ["-f", *[str(fold) for fold in args.fold]]


def extra_args(args: argparse.Namespace) -> list[str]:
    extras: list[str] = []
    for item in args.extra:
        item = item.strip()
        if not item:
            continue
        extras.extend(["-X", item])
    return extras


def run_framework(framework: str, args: argparse.Namespace) -> int:
    cmd = [
        args.python,
        "runbenchmark.py",
        framework,
        BENCHMARK,
        args.constraint,
        "-m",
        args.mode,
        "-s",
        args.setup,
        *task_args(args),
        *fold_args(args),
        *extra_args(args),
    ]
    if args.outdir is not None:
        cmd.extend(["-o", str(args.outdir)])

    print()
    print(
        f"===== Running {framework} on {BENCHMARK} "
        f"({args.part}, {args.constraint}, {args.mode}, setup={args.setup}) ====="
    )
    completed = subprocess.run(cmd, cwd=repo_root(), check=False)
    if completed.returncode == 0:
        print(f"===== {framework}: OK =====")
    else:
        print(f"===== {framework}: FAILED ({completed.returncode}) =====", file=sys.stderr)
    return int(completed.returncode)


def main() -> int:
    args = parse_args()
    failures: list[tuple[str, int]] = []
    for framework in parse_frameworks(args.framework):
        exit_code = run_framework(framework, args)
        if exit_code != 0:
            failures.append((framework, exit_code))
            if not args.continue_on_error:
                break
    if failures:
        print("Failed frameworks:", file=sys.stderr)
        for framework, exit_code in failures:
            print(f"- {framework}: exit code {exit_code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

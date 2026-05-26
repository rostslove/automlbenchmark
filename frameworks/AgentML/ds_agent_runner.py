#!/usr/bin/env python
"""Run DS-Agent with AMLB-generated benchmark tasks."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-dir", type=Path, required=True)
    args, runner_args = parser.parse_known_args()
    return args, runner_args


def read_research_problem(task_dir: Path) -> str:
    for name in ("research_problem.txt", "overview.txt", "data_description.txt"):
        path = task_dir / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return f"Solve the benchmark task in {task_dir.name}."


def patch_prepare_task(runner_dir: Path) -> None:
    sys.path.insert(0, str(runner_dir))

    import prepare_task

    benchmarks_dir = runner_dir / "benchmarks"

    def get_task_info(task: str) -> tuple[str, str]:
        task_dir = benchmarks_dir / task
        if not task_dir.is_dir():
            raise ValueError(
                f"task {task} not supported in benchmarks; expected directory {task_dir}"
            )
        return task, read_research_problem(task_dir)

    prepare_task.get_task_info = get_task_info


def main() -> int:
    args, runner_args = parse_args()
    runner_dir = args.runner_dir.resolve()
    runner_path = runner_dir / "runner.py"
    if not runner_path.is_file():
        raise FileNotFoundError(f"DS-Agent runner.py not found: {runner_path}")

    patch_prepare_task(runner_dir)
    os.chdir(runner_dir)
    sys.argv = [str(runner_path), *runner_args]
    runpy.run_path(str(runner_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

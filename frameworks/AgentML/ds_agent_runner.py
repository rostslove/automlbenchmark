#!/usr/bin/env python
"""Run DS-Agent with AMLB-generated benchmark tasks."""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-dir", type=Path, required=True)
    args, runner_args = parser.parse_known_args()
    return args, runner_args


def option_value(args: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, item in enumerate(args):
        if item == name and index + 1 < len(args):
            return args[index + 1]
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def read_research_problem(task_dir: Path) -> str:
    for name in ("research_problem.txt", "overview.txt", "data_description.txt"):
        path = task_dir / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return f"Solve the benchmark task in {task_dir.name}."


def patch_prepare_task(runner_dir: Path) -> None:
    sys.path.insert(0, str(runner_dir))
    sys.path.insert(0, str(runner_dir.parent))
    benchmarks_dir = runner_dir / "benchmarks"

    def get_task_info(task: str) -> tuple[str, str]:
        task_dir = benchmarks_dir / task
        if not task_dir.is_dir():
            raise ValueError(
                f"task {task} not supported in benchmarks; expected directory {task_dir}"
            )
        return task, read_research_problem(task_dir)

    for module_name in ("prepare_task", f"{runner_dir.name}.prepare_task"):
        try:
            module = __import__(module_name, fromlist=["get_task_info"])
        except ImportError:
            continue
        module.get_task_info = get_task_info


def prepare_run_dirs(runner_dir: Path, runner_args: list[str]) -> None:
    task = option_value(runner_args, "--task")
    work_dir = option_value(runner_args, "--work-dir")
    log_dir = option_value(runner_args, "--log-dir")
    if task and work_dir:
        source = runner_dir / "benchmarks" / task
        destination = Path(work_dir) / task
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.mkdir(parents=True, exist_ok=True)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)


def patch_mkdir_parent_creation() -> None:
    original_mkdir = os.mkdir

    def mkdir(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], mode: int = 0o777, *args: Any, **kwargs: Any) -> None:
        try:
            return original_mkdir(path, mode, *args, **kwargs)
        except FileNotFoundError:
            if args or kwargs:
                raise
            ensure_directory(Path(path).parent, mode=mode, mkdir=original_mkdir)
            return original_mkdir(path, mode)

    os.mkdir = mkdir


def ensure_directory(path: Path, mode: int, mkdir: Any) -> None:
    missing = []
    current = path
    while current and not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        try:
            mkdir(directory, mode)
        except FileExistsError:
            pass


def main() -> int:
    args, runner_args = parse_args()
    runner_dir = args.runner_dir.resolve()
    runner_path = runner_dir / "runner.py"
    if not runner_path.is_file():
        raise FileNotFoundError(f"DS-Agent runner.py not found: {runner_path}")

    os.chdir(runner_dir)
    patch_prepare_task(runner_dir)
    prepare_run_dirs(runner_dir, runner_args)
    patch_mkdir_parent_creation()
    sys.argv = [str(runner_path), *runner_args]
    runpy.run_path(str(runner_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Small bridge for AutoML-Agent's AgentManager API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from agent_manager import AgentManager

    prompt = args.prompt_file.read_text(encoding="utf-8")
    prompt = (
        prompt
        + "\n\nThe labeled training file passed to AgentManager is: "
        + str(args.data_path.resolve())
        + "\nThe full task directory with train.csv, test.csv and sample_submission.csv is: "
        + str(args.data_path.resolve().parent)
        + "\nWrite submission.csv under: "
        + str(output_dir)
    )
    manager = AgentManager(
        llm=args.llm,
        interactive=False,
        data_path=str(args.data_path.resolve()),
    )
    manager.initiate_chat(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

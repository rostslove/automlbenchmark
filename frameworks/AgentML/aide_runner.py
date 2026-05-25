#!/usr/bin/env python
"""Run AIDE ML through its Python API when the CLI entry point is unavailable."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def newest_submission(search_root: Path) -> Path | None:
    candidates = [
        path
        for path in search_root.rglob("submission.csv")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> int:
    import aide

    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exp = aide.Experiment(
        data_dir=str(args.data_dir.resolve()),
        goal=args.goal,
        eval=args.eval,
    )
    solution = exp.run(steps=args.steps)

    metadata = {
        "valid_metric": getattr(solution, "valid_metric", None),
        "solution_type": type(solution).__name__,
    }
    if hasattr(solution, "code"):
        (output_dir / "best_solution.py").write_text(solution.code, encoding="utf-8")
    (output_dir / "aide_solution.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    submission = newest_submission(output_dir) or newest_submission(Path.cwd())
    if submission is not None and submission.resolve() != (output_dir / "submission.csv").resolve():
        shutil.copy2(submission, output_dir / "submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

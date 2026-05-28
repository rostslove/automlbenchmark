#!/usr/bin/env python
"""Prepare M4 classification folds and run AgentML frameworks through AMLB."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import run_diploma_agent_frameworks as agents


DEFAULT_AGENT_FRAMEWORKS = ("AutoMLAgent", "DSAgent", "AIDE")
DEFAULT_M4_GROUPS = ("Yearly", "Monthly", "Quarterly", "Daily")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AgentML frameworks on generated M4 frequency-group classification."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_M4_GROUPS),
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
        help="Where to write M4 artifacts, folds, and generated YAML.",
    )
    parser.add_argument(
        "--framework",
        "-f",
        default="all",
        help="Framework name, 'all', or comma-separated list. Default: all AgentML frameworks.",
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
        "--fold",
        type=int,
        nargs="*",
        default=[0],
        help="Optional AMLB fold numbers to run. Default: 0. Pass no values to run all generated folds.",
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
        "--results-dir",
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
        help="Extra runbenchmark.py override, e.g. -X f._repo=/path/to/repo.",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Route OpenAI-compatible LLM calls to a local Ollama endpoint.",
    )
    parser.add_argument(
        "--ollama-url",
        default=(
            os.environ.get("AGENT_LLM_BASE_URL")
            or os.environ.get("OLLAMA_OPENAI_BASE_URL")
            or agents.DEFAULT_OLLAMA_BASE_URL
        ),
        help=f"Ollama OpenAI-compatible base URL. Default: {agents.DEFAULT_OLLAMA_BASE_URL}.",
    )
    parser.add_argument(
        "--ollama-model",
        default=(
            os.environ.get("AGENT_LLM_MODEL")
            or os.environ.get("LLM_MODEL_ALIAS")
            or os.environ.get("OLLAMA_MODEL_ALIAS")
            or agents.DEFAULT_OLLAMA_MODEL
        ),
        help=f"Framework-facing Ollama model alias. Default: {agents.DEFAULT_OLLAMA_MODEL}.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining frameworks after a failure.",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip external command and repository preflight checks.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create M4 artifacts, folds, and benchmark YAML; do not run frameworks.",
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


def parse_frameworks(value: str) -> list[str]:
    if value.lower() in {"all", "agents", "agentml"}:
        return list(DEFAULT_AGENT_FRAMEWORKS)
    frameworks = [item.strip() for item in value.split(",") if item.strip()]
    if not frameworks:
        raise ValueError("No frameworks selected.")
    return frameworks


def fold_args(args: argparse.Namespace) -> list[str]:
    if args.fold is None or len(args.fold) == 0:
        return []
    return ["-f", *[str(fold) for fold in args.fold]]


def load_m4_runner() -> object:
    import run_m4_classification_frameworks as m4_runner

    return m4_runner


def run_framework(
    framework: str,
    args: argparse.Namespace,
    benchmark_path: Path,
    repo_root: Path,
) -> int:
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
        *fold_args(args),
        *agents.extra_args(args, framework),
    ]
    if args.results_dir is not None:
        cmd.extend(["-o", str(args.results_dir)])

    print()
    print(
        f"===== Running {framework} on M4 classification "
        f"({args.constraint}, {args.mode}, setup={args.setup}) ====="
    )
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    if completed.returncode == 0:
        print(f"===== {framework}: OK =====")
    else:
        print(f"===== {framework}: FAILED ({completed.returncode}) =====", file=sys.stderr)
    return int(completed.returncode)


def main() -> int:
    args = parse_args()
    if args.folds < 1:
        raise ValueError("--folds must be at least 1.")

    frameworks = parse_frameworks(args.framework)
    if not args.prepare_only:
        agents.configure_ollama(args)
        agents.ensure_agentml_installed_marker(frameworks, args.setup)
        if not args.no_preflight:
            agents.preflight(frameworks, args)

    m4_runner = load_m4_runner()
    groups = m4_runner.normalize_groups(args.groups)
    metadata = m4_runner.create_m4_artifact(args, groups)
    benchmark_path, split_rows = m4_runner.create_amlb_folds(args, metadata)
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

    failures: list[tuple[str, int]] = []
    for framework in frameworks:
        exit_code = run_framework(framework, args, benchmark_path, m4_runner.repo_root())
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

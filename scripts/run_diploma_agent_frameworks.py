#!/usr/bin/env python
"""Run agentic AutoML frameworks through the standard AMLB runner."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


BENCHMARK = "diploma_mixed"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:32b"
DEFAULT_FRAMEWORKS = (
    "AutoGluonAssistant",
    "AutoKaggle",
    "AIDE",
    "AutoMLAgent",
    "DSAgent",
)
AGENTML_MODULE_DIR = Path("frameworks") / "AgentML"
EXTERNAL_VERSION = "external"
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
REPO_FRAMEWORKS = {
    "AutoKaggle": ("AUTOKAGGLE_REPO", "AUTOKAGGLE_PYTHON", ("framework.py",)),
    "AutoMLAgent": ("AUTOML_AGENT_REPO", "AUTOML_AGENT_PYTHON", ("agent_manager",)),
    "DSAgent": ("DS_AGENT_REPO", "DS_AGENT_PYTHON", ("development", "MLAgentBench", "runner.py")),
}
COMMAND_FRAMEWORKS = {
    "AutoGluonAssistant": ("MLZERO_COMMAND", "mlzero"),
}
AIDE_COMMAND_ENV = "AIDE_COMMAND"
AIDE_PYTHON_ENV = "AIDE_PYTHON"


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
        help="Extra runbenchmark.py override, e.g. -X f._repo=D:\\repo.",
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
            or DEFAULT_OLLAMA_BASE_URL
        ),
        help=f"Ollama OpenAI-compatible base URL. Default: {DEFAULT_OLLAMA_BASE_URL}.",
    )
    parser.add_argument(
        "--ollama-model",
        default=(
            os.environ.get("AGENT_LLM_MODEL")
            or os.environ.get("LLM_MODEL")
            or os.environ.get("OLLAMA_MODEL")
            or DEFAULT_OLLAMA_MODEL
        ),
        help=f"Ollama model id. Default: {DEFAULT_OLLAMA_MODEL}.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining frameworks after a failure.",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip checks for external commands and repository paths before running AMLB.",
    )
    return parser.parse_args()


def configure_ollama(args: argparse.Namespace) -> None:
    if not args.ollama:
        return

    for key in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(key, None)

    os.environ["AGENT_LLM_BASE_URL"] = args.ollama_url
    os.environ["AGENT_LLM_API_KEY"] = "ollama"
    os.environ["AGENT_LLM_MODEL"] = args.ollama_model
    os.environ["LLM_MODEL"] = args.ollama_model
    os.environ["OLLAMA_MODEL"] = args.ollama_model
    os.environ["OPENAI_API_KEY"] = "ollama"
    os.environ["OPENAI_BASE_URL"] = args.ollama_url
    os.environ["OPENAI_API_BASE"] = args.ollama_url
    append_no_proxy(["127.0.0.1", "localhost", "ollama"])
    print(f"Using Ollama endpoint {args.ollama_url} with model {args.ollama_model}.")


def append_no_proxy(entries: list[str]) -> None:
    for key in ("NO_PROXY", "no_proxy"):
        current = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        seen = set(current)
        for entry in entries:
            if entry not in seen:
                current.append(entry)
                seen.add(entry)
        os.environ[key] = ",".join(current)


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


def extra_args(args: argparse.Namespace, framework: str | None = None) -> list[str]:
    extras: list[str] = []
    for item in args.extra:
        item = item.strip()
        if not item:
            continue
        extras.extend(["-X", item])

    overrides = framework_overrides(args)
    if framework == "AutoGluonAssistant" and "_command" not in overrides:
        command = os.environ.get("MLZERO_COMMAND")
        if command:
            extras.extend(["-X", f"f._command={command}"])
    if framework == "AIDE" and "_python" not in overrides:
        python = os.environ.get(AIDE_PYTHON_ENV)
        if python:
            extras.extend(["-X", f"f._python={python}"])
    return extras


def framework_overrides(args: argparse.Namespace) -> dict[str, str]:
    overrides = {}
    for item in args.extra:
        if not item.startswith("f.") or "=" not in item:
            continue
        key, value = item[2:].split("=", 1)
        overrides[key] = value
    return overrides


def preflight(frameworks: list[str], args: argparse.Namespace) -> None:
    overrides = framework_overrides(args)
    missing: list[str] = []

    for framework, (env_var, default_command) in COMMAND_FRAMEWORKS.items():
        if framework not in frameworks:
            continue
        command = overrides.get("_command") or os.environ.get(env_var) or default_command
        executable = command_executable(command)
        if not command_exists(command):
            missing.append(
                f"{framework}: command `{executable}` not found. Install it or pass "
                f"`--extra f._command=/absolute/path/to/{executable}` or set `{env_var}`."
            )

    if "AIDE" in frameworks:
        preflight_aide(overrides, missing)

    for framework, (env_var, python_env_var, required_parts) in REPO_FRAMEWORKS.items():
        if framework not in frameworks:
            continue
        repo = overrides.get("_repo") or os.environ.get(env_var)
        if not repo:
            missing.append(
                f"{framework}: set `{env_var}` to the framework checkout path "
                f"or pass `--extra f._repo=/path/to/repo` when running only this framework."
            )
            continue
        repo_path = Path(repo).expanduser()
        required_path = repo_path.joinpath(*required_parts)
        if not required_path.exists():
            missing.append(
                f"{framework}: `{repo_path}` does not look valid "
                f"(missing `{required_path}`)."
            )
        python_path = overrides.get("_python") or os.environ.get(python_env_var)
        if python_path and not Path(python_path).expanduser().exists():
            missing.append(
                f"{framework}: `{python_env_var}` points to a missing Python executable: {python_path}."
            )

    if missing:
        print("Agent framework preflight failed:", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Bootstrap example:", file=sys.stderr)
        print("  bash scripts/setup_diploma_agent_frameworks.sh", file=sys.stderr)
        print("  source scripts/diploma_agent_frameworks.env", file=sys.stderr)
        print("  bash scripts/start_diploma_ollama.sh", file=sys.stderr)
        print("  source scripts/diploma_ollama.env", file=sys.stderr)
        print(
            "  python scripts/run_diploma_agent_frameworks.py --framework all --setup skip --ollama --continue-on-error",
            file=sys.stderr,
        )
        raise SystemExit(2)


def preflight_aide(overrides: dict[str, str], missing: list[str]) -> None:
    command = overrides.get("_command") or os.environ.get(AIDE_COMMAND_ENV)
    if command and command_exists(command):
        return
    if command:
        print(
            f"AIDE command `{command_executable(command)}` is not available; "
            "preflight will try the Python API.",
            file=sys.stderr,
        )

    python_path = overrides.get("_python") or os.environ.get(AIDE_PYTHON_ENV) or sys.executable
    if not executable_exists(python_path):
        missing.append(
            f"AIDE: `{AIDE_PYTHON_ENV}` points to a missing Python executable: {python_path}."
        )
        return

    completed = subprocess.run(
        [python_path, "-c", "import aide"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        detail_text = f" Last import error: {detail[-1]}" if detail else ""
        missing.append(
            "AIDE: command `aide` was not found and the selected Python cannot import `aide`. "
            f"Set `{AIDE_PYTHON_ENV}` to the bootstrap CLI venv Python or re-run "
            f"`bash scripts/setup_diploma_agent_frameworks.sh`.{detail_text}"
        )


def command_executable(command: str) -> str:
    try:
        parts = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        parts = command.split()
    return parts[0] if parts else command


def command_exists(command: str) -> bool:
    return executable_exists(command_executable(command))


def executable_exists(executable: str) -> bool:
    return Path(executable).expanduser().exists() or shutil.which(executable) is not None


def ensure_agentml_installed_marker(frameworks: list[str], setup: str) -> None:
    if setup != "skip":
        return
    if not any(framework in DEFAULT_FRAMEWORKS for framework in frameworks):
        return
    installed = repo_root() / AGENTML_MODULE_DIR / ".setup" / "installed"
    installed.parent.mkdir(parents=True, exist_ok=True)
    existing = installed.read_text(encoding="utf-8").splitlines() if installed.exists() else []
    if EXTERNAL_VERSION not in existing:
        installed.write_text("\n".join([*existing, EXTERNAL_VERSION, ""]), encoding="utf-8")


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
        *extra_args(args, framework),
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
    configure_ollama(args)
    frameworks = parse_frameworks(args.framework)
    ensure_agentml_installed_marker(frameworks, args.setup)
    if not args.no_preflight:
        preflight(frameworks, args)
    failures: list[tuple[str, int]] = []
    for framework in frameworks:
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

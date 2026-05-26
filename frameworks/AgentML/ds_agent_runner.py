#!/usr/bin/env python
"""Run DS-Agent with AMLB-generated benchmark tasks."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
import time
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


def local_base_url() -> str | None:
    return (
        os.environ.get("AGENT_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OLLAMA_OPENAI_BASE_URL")
    )


def local_api_key() -> str:
    return os.environ.get("AGENT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"


def local_model(default: str | None = None) -> str:
    return os.environ.get("AGENT_LLM_MODEL") or default or "gpt-4o-mini"


class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def attrdict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({key: attrdict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [attrdict(item) for item in value]
    return value


def stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "value", "message"):
            if key in content:
                return stringify_content(content[key])
        return json.dumps(content, ensure_ascii=False, default=str)
    if isinstance(content, list):
        return "\n".join(stringify_content(item) for item in content if item is not None)
    return str(content)


def sanitize_messages(messages: Any, prompt: Any = None) -> list[dict[str, str]]:
    if not messages:
        return [{"role": "user", "content": stringify_content(prompt)}]
    if not isinstance(messages, list):
        return [{"role": "user", "content": stringify_content(messages)}]

    sanitized = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role") or "user")
            content = message.get("content", "")
        else:
            role = str(getattr(message, "role", None) or "user")
            content = getattr(message, "content", message)
        sanitized.append({"role": role, "content": stringify_content(content)})
    return sanitized or [{"role": "user", "content": stringify_content(prompt)}]


def extract_chat_output(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return stringify_content(getattr(choices[0], "text", ""))
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        function = getattr(tool_calls[0], "function", None)
        arguments = getattr(function, "arguments", None)
        if arguments:
            return stringify_content(arguments)
    return stringify_content(getattr(message, "content", ""))


def completion_response(content: str, response: Any = None) -> AttrDict:
    usage = getattr(response, "usage", None)
    return attrdict(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "text": content,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
        }
    )


def configure_openai_v1_compat(default_model: str | None = None) -> None:
    base_url = local_base_url()
    if not base_url:
        return

    api_key = local_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_BASE"] = base_url

    try:
        import openai
        from openai import OpenAI
    except Exception:
        return

    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def chat_create(*args: Any, **kwargs: Any) -> AttrDict:
        model = kwargs.get("model") or local_model(default_model)
        messages = sanitize_messages(kwargs.get("messages"), kwargs.get("prompt"))
        request: dict[str, Any] = {"model": model, "messages": messages}
        for key in (
            "temperature",
            "max_tokens",
            "stop",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ):
            if kwargs.get(key) is not None:
                request[key] = kwargs[key]
        started = time.perf_counter()
        response = client.chat.completions.create(**request)
        content = extract_chat_output(response)
        if kwargs.get("log_file"):
            elapsed = time.perf_counter() - started
            Path(kwargs["log_file"]).write_text(
                f"local_openai_chat_completions elapsed={elapsed:.3f}s\n{content}",
                encoding="utf-8",
            )
        return completion_response(content, response)

    def completion_create(*args: Any, **kwargs: Any) -> AttrDict:
        prompt = kwargs.get("prompt")
        if prompt is None and args:
            prompt = args[0]
        kwargs = dict(kwargs)
        kwargs["messages"] = [{"role": "user", "content": stringify_content(prompt)}]
        return chat_create(**kwargs)

    class ChatCompletion:
        create = staticmethod(chat_create)
        acreate = staticmethod(chat_create)

    class Completion:
        create = staticmethod(completion_create)
        acreate = staticmethod(completion_create)

    openai.ChatCompletion = ChatCompletion
    openai.Completion = Completion


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


def prepare_retrieval_data_dirs(runner_dir: Path) -> None:
    data_dir = runner_dir.parent / "data"
    fallback_case = (
        "AutoML tabular baseline case\n\n"
        "Inspect train.csv and test.csv, identify the target column from the task description, "
        "fit robust sklearn tabular baselines with preprocessing for numeric and categorical "
        "features, validate with a held-out split, and write submission.csv with the required "
        "id and prediction columns.\n"
    )
    for name in ("nlp_cases", "tabular_cases"):
        case_dir = data_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_file = case_dir / "amlb_tabular_baseline.txt"
        if not case_file.exists():
            case_file.write_text(fallback_case, encoding="utf-8")


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
    prepare_retrieval_data_dirs(runner_dir)
    patch_mkdir_parent_creation()
    configure_openai_v1_compat(option_value(runner_args, "--llm-name"))
    sys.argv = [str(runner_path), *runner_args]
    runpy.run_path(str(runner_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

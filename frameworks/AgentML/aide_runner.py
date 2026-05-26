#!/usr/bin/env python
"""Run AIDE ML through its Python API when the CLI entry point is unavailable."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-model", default=None)
    parser.add_argument("--feedback-model", default=None)
    parser.add_argument("--report-model", default=None)
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


def local_base_url() -> str | None:
    return (
        os.environ.get("AGENT_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OLLAMA_OPENAI_BASE_URL")
    )


def local_api_key() -> str:
    return os.environ.get("AGENT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"


def configure_openai_env() -> tuple[str | None, str]:
    base_url = local_base_url()
    api_key = local_api_key()
    if not base_url:
        return None, api_key
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key
    return base_url, api_key


def configure_aide_backend(base_url: str, api_key: str, default_model: str | None) -> None:
    import inspect

    from openai import BadRequestError, OpenAI

    import aide.backend as aide_backend
    import aide.agent as aide_agent
    import aide.backend.backend_openai as backend_openai

    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    original_query = backend_openai.query
    original_signature = inspect.signature(original_query)
    original_backend_query = aide_backend.query
    original_backend_signature = inspect.signature(original_backend_query)

    def setup_openai_client() -> OpenAI:
        return client

    def local_query(*args: Any, **kwargs: Any) -> tuple[str, float, int, int, dict[str, Any]]:
        values = bind_query_arguments(original_signature, args, kwargs)
        system_message = values.get("system_message")
        user_message = values.get("user_message")
        model = values.get("model") or default_model
        if not model:
            raise RuntimeError(
                "AIDE local OpenAI-compatible backend is active, but no model was provided. "
                "Set AGENT_LLM_MODEL or pass --code-model/--feedback-model."
            )

        messages = build_messages(backend_openai, system_message, user_message)
        temperature = values.get("temperature")
        max_tokens = values.get("max_tokens") or values.get("max_completion_tokens")
        func_spec = values.get("func_spec")

        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        tools = normalize_tools(func_spec)
        if tools:
            request["tools"] = tools
            request["tool_choice"] = tool_choice(tools)

        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**request)
        except BadRequestError:
            if not tools:
                raise
            request.pop("tools", None)
            request.pop("tool_choice", None)
            response = client.chat.completions.create(**request)
        elapsed = time.perf_counter() - started

        output = extract_output(response)
        usage = getattr(response, "usage", None)
        in_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        info = {
            "backend": "local_openai_chat_completions",
            "base_url": base_url,
            "model": model,
        }
        return output, elapsed, in_tokens, out_tokens, info

    backend_openai._client = client
    backend_openai._setup_openai_client = setup_openai_client
    backend_openai.query = local_query
    patch_module_references(aide_backend, original_query, local_query)

    def local_backend_query(*args: Any, **kwargs: Any) -> str:
        values = bind_query_arguments(original_backend_signature, args, kwargs)
        output, _, _, _, _ = local_query(**values)
        return output

    aide_backend.query = local_backend_query
    patch_module_references(aide_agent, original_backend_query, local_backend_query)


def bind_query_arguments(signature: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        return dict(signature.bind_partial(*args, **kwargs).arguments)
    except TypeError:
        values = dict(kwargs)
        positional = [
            "system_message",
            "user_message",
            "model",
            "temperature",
            "max_tokens",
            "func_spec",
        ]
        for name, value in zip(positional, args):
            values.setdefault(name, value)
        return values


def build_messages(backend_openai: Any, system_message: Any, user_message: Any) -> list[dict[str, str]]:
    formatter = getattr(backend_openai, "opt_messages_to_list", None)
    if formatter is not None:
        for call in (
            lambda: formatter(system_message=system_message, user_message=user_message),
            lambda: formatter(system_message, user_message),
        ):
            try:
                return sanitize_messages(call())
            except TypeError:
                continue

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    return sanitize_messages(messages)


def sanitize_messages(messages: Any) -> list[dict[str, str]]:
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
    return sanitized


def normalize_tools(func_spec: Any) -> list[dict[str, Any]] | None:
    if not func_spec:
        return None
    specs = func_spec if isinstance(func_spec, list) else [func_spec]
    tools = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if spec.get("type") == "function" and "function" in spec:
            tools.append(spec)
        elif "function" in spec:
            tools.append({"type": "function", "function": spec["function"]})
        else:
            tools.append({"type": "function", "function": spec})
    return tools or None


def tool_choice(tools: list[dict[str, Any]]) -> Any:
    first_function = tools[0].get("function", {})
    name = first_function.get("name")
    if name:
        return {"type": "function", "function": {"name": name}}
    return "auto"


def extract_output(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        function = getattr(tool_calls[0], "function", None)
        arguments = getattr(function, "arguments", None)
        if arguments:
            return str(arguments)
    return stringify_content(getattr(message, "content", ""))


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
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(stringify_content(item))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                parts.append(str(text or item))
        return "\n".join(part for part in parts if part)
    return str(content)


def patch_module_references(module: Any, original: Any, replacement: Any) -> None:
    for name, value in vars(module).items():
        if value is original:
            setattr(module, name, replacement)
        elif isinstance(value, dict):
            for key, item in list(value.items()):
                if item is original:
                    value[key] = replacement


def patch_aide_metric_normalization() -> None:
    import aide.agent as aide_agent

    original = aide_agent.Agent.parse_exec_result

    def parse_exec_result(self: Any, *args: Any, **kwargs: Any) -> Any:
        for value in args:
            normalize_metric_inplace(value)
        for value in kwargs.values():
            normalize_metric_inplace(value)
        return original(self, *args, **kwargs)

    aide_agent.Agent.parse_exec_result = parse_exec_result


def normalize_metric_inplace(value: Any) -> None:
    if isinstance(value, dict) and "metric" in value:
        value["metric"] = coerce_metric(value.get("metric"))


def coerce_metric(value: Any) -> float:
    if isinstance(value, float):
        return value
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    args = parse_args()
    base_url, api_key = configure_openai_env()
    aide_cli_args = [sys.argv[0]]
    if args.code_model:
        aide_cli_args.append(f"agent.code.model={args.code_model}")
    if args.feedback_model:
        aide_cli_args.append(f"agent.feedback.model={args.feedback_model}")
    if args.report_model:
        aide_cli_args.append(f"report.model={args.report_model}")
    sys.argv = aide_cli_args

    import aide
    patch_aide_metric_normalization()

    if base_url:
        configure_aide_backend(
            base_url=base_url,
            api_key=api_key,
            default_model=args.code_model or args.feedback_model or args.report_model,
        )

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

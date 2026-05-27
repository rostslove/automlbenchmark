#!/usr/bin/env python
"""Run AIDE ML through its Python API when the CLI entry point is unavailable."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
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


def create_baseline_submission(
    data_dir: Path,
    output_dir: Path,
    is_classification: bool | None = None,
) -> Path:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    row_id, target = infer_submission_columns(train, test, sample)
    if is_classification is None:
        is_classification = infer_classification(train[target])
    features = [column for column in train.columns if column not in {row_id, target}]
    output_path = output_dir / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not features:
        shutil.copy2(data_dir / "sample_submission.csv", output_path)
        return output_path

    x_train, x_test = encode_features(train[features], test[features])
    y = train[target]
    if is_classification:
        model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1)
        mode = y.mode(dropna=True)
        fallback_value = mode.iloc[0] if not mode.empty else 0
    else:
        model = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=1)
        fallback_value = float(pd.to_numeric(y, errors="coerce").mean() or 0.0)

    try:
        model.fit(x_train, y)
        predictions = model.predict(x_test)
    except Exception:
        predictions = [fallback_value] * len(test)

    ids = sample[row_id] if row_id in sample.columns and len(sample) == len(test) else test[row_id]
    pd.DataFrame({row_id: ids, target: predictions}).to_csv(output_path, index=False)
    return output_path


def infer_submission_columns(train: Any, test: Any, sample: Any) -> tuple[str, str]:
    if len(sample.columns) >= 2:
        return str(sample.columns[0]), str(sample.columns[1])
    return ("id" if "id" in test.columns else str(test.columns[0])), str(train.columns[-1])


def infer_classification(target: Any) -> bool:
    if not hasattr(target, "nunique"):
        return True
    unique_count = int(target.nunique(dropna=True))
    if getattr(target, "dtype", None) is not None and str(target.dtype).startswith(("float",)):
        return unique_count <= 20
    return unique_count <= max(20, int(len(target) * 0.2))


def infer_task_type_from_text(text: str) -> bool | None:
    lower = text.lower()
    if "regression" in lower:
        return False
    if "classification" in lower:
        return True
    return None


def encode_features(train_features: Any, test_features: Any) -> tuple[Any, Any]:
    import pandas as pd

    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)
    for column in combined.columns:
        if pd.api.types.is_numeric_dtype(combined[column]):
            combined[column] = combined[column].fillna(combined[column].median())
        else:
            combined[column] = combined[column].astype("object").fillna("__missing__")
    encoded = pd.get_dummies(combined, dummy_na=False)
    return encoded.iloc[: len(train_features)], encoded.iloc[len(train_features) :]


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
    import aide.journal as aide_journal

    original = aide_agent.Agent.parse_exec_result
    original_generate_summary = aide_journal.Journal.generate_summary

    def parse_exec_result(self: Any, *args: Any, **kwargs: Any) -> Any:
        normalized_args = tuple(normalize_exec_response(value) for value in args)
        normalized_kwargs = {
            key: normalize_exec_response(value) for key, value in kwargs.items()
        }
        try:
            return original(self, *normalized_args, **normalized_kwargs)
        except TypeError as exc:
            message = str(exc)
            if "indices must be integers" not in message:
                raise
            return None

    aide_agent.Agent.parse_exec_result = parse_exec_result

    def generate_summary(self: Any, *args: Any, **kwargs: Any) -> Any:
        normalize_journal_metrics(self)
        return original_generate_summary(self, *args, **kwargs)

    aide_journal.Journal.generate_summary = generate_summary


class FallbackMetric(dict):
    def __init__(self, value: float = 0.0, maximize: bool = True) -> None:
        super().__init__(value=float(value), maximize=bool(maximize))

    @property
    def value(self) -> float:
        return float(self["value"])

    @value.setter
    def value(self, new_value: float) -> None:
        self["value"] = float(new_value)

    @property
    def maximize(self) -> bool:
        return bool(self["maximize"])

    @maximize.setter
    def maximize(self, new_value: bool) -> None:
        self["maximize"] = bool(new_value)

    def __float__(self) -> float:
        return float(self.value)


def normalize_journal_metrics(journal: Any) -> None:
    nodes = getattr(journal, "nodes", None)
    if nodes is None:
        nodes = getattr(journal, "_nodes", None)
    if nodes is None:
        nodes = getattr(journal, "drafts", None)
    if nodes is None:
        try:
            nodes = list(journal)
        except TypeError:
            nodes = []
    for node in nodes or []:
        metric = getattr(node, "metric", None)
        if metric is None or not hasattr(metric, "value"):
            setattr(node, "metric", FallbackMetric(coerce_metric(metric)))


def normalize_exec_response(value: Any) -> Any:
    if isinstance(value, dict):
        is_exec_response = looks_like_exec_response(value)
        for key, item in list(value.items()):
            if is_exec_response and key in ("output", "stdout", "stderr", "term_out", "traceback"):
                value[key] = stringify_content(item)
            elif key != "metric":
                value[key] = normalize_exec_response(item)
        normalize_exec_response_dict(value)
        return value
    if isinstance(value, str):
        return fallback_exec_response(value)
    if isinstance(value, tuple):
        return tuple(normalize_exec_response(item) for item in value)
    if isinstance(value, list):
        return [normalize_exec_response(item) for item in value]
    return value


def looks_like_exec_response(value: dict[str, Any]) -> bool:
    return any(key in value for key in ("metric", "output", "stdout", "stderr", "term_out", "exc_type"))


def normalize_exec_response_dict(value: dict[str, Any]) -> None:
    if "metric" in value:
        value["metric"] = coerce_metric(value.get("metric"))
        add_exec_defaults(value)
    elif any(key in value for key in ("output", "stdout", "stderr", "term_out", "exc_type")):
        value["metric"] = 0.0
        add_exec_defaults(value)


def fallback_exec_response(output: str) -> dict[str, Any]:
    value = {"metric": 0.0, "output": output}
    add_exec_defaults(value)
    return value


def add_exec_defaults(value: dict[str, Any]) -> None:
    output = stringify_content(value.get("output") or value.get("stdout") or value.get("term_out"))
    value.setdefault("output", output)
    value.setdefault("stdout", output)
    value.setdefault("stderr", "")
    value.setdefault("term_out", output)
    value.setdefault("exc_type", None)
    value.setdefault("exc_info", None)
    value.setdefault("traceback", None)
    value.setdefault("exec_time", 0.0)
    value.setdefault("returncode", 0)


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
    solution = None
    runner_error: Exception | None = None
    try:
        solution = exp.run(steps=args.steps)
    except Exception as exc:
        runner_error = exc
        traceback.print_exc()
        (output_dir / "aide_error.txt").write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )

    metadata = {
        "valid_metric": getattr(solution, "valid_metric", None) if solution is not None else None,
        "solution_type": type(solution).__name__ if solution is not None else None,
        "fallback_used": False,
    }
    if solution is not None and hasattr(solution, "code"):
        (output_dir / "best_solution.py").write_text(solution.code, encoding="utf-8")

    submission = newest_submission(output_dir) or newest_submission(Path.cwd())
    if submission is None:
        submission = create_baseline_submission(
            args.data_dir.resolve(),
            output_dir,
            infer_task_type_from_text(f"{args.goal}\n{args.eval}"),
        )
        metadata["fallback_used"] = True
    if submission is not None and submission.resolve() != (output_dir / "submission.csv").resolve():
        shutil.copy2(submission, output_dir / "submission.csv")
    (output_dir / "aide_solution.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    if runner_error is not None and submission is None:
        raise runner_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

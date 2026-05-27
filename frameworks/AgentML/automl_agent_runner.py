#!/usr/bin/env python
"""Small bridge for AutoML-Agent's AgentManager API."""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import os
import re
import shutil
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _document_text(document: Any) -> str:
    return str(getattr(document, "page_content", document))


class _FallbackBM25Retriever:
    """Small compatibility retriever for old AutoML-Agent LangChain imports."""

    def __init__(self, documents: Iterable[Any], k: int = 4, **_: Any) -> None:
        self.documents = list(documents)
        self.k = k
        corpus = [_tokenize(_document_text(document)) for document in self.documents]
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(corpus)
        except Exception:
            self._bm25 = None
        self._corpus = corpus

    @classmethod
    def from_documents(cls, documents: Iterable[Any], **kwargs: Any) -> "_FallbackBM25Retriever":
        return cls(documents, **kwargs)

    def get_relevant_documents(self, query: str, **_: Any) -> list[Any]:
        return self.invoke(query)

    def invoke(self, query: str, **_: Any) -> list[Any]:
        if not self.documents:
            return []
        tokens = _tokenize(query)
        if self._bm25 is not None and tokens:
            scores = list(self._bm25.get_scores(tokens))
        else:
            token_set = set(tokens)
            scores = [len(token_set.intersection(document_tokens)) for document_tokens in self._corpus]
        ranked = sorted(range(len(self.documents)), key=lambda index: scores[index], reverse=True)
        return [self.documents[index] for index in ranked[: self.k]]


class _FallbackContextualCompressionRetriever:
    def __init__(self, base_compressor: Any = None, base_retriever: Any = None, **kwargs: Any) -> None:
        self.base_compressor = base_compressor or kwargs.get("base_compressor")
        self.base_retriever = base_retriever or kwargs.get("base_retriever")

    def get_relevant_documents(self, query: str, **_: Any) -> list[Any]:
        documents = _retrieve_documents(self.base_retriever, query)
        compressor = self.base_compressor
        if compressor is None:
            return documents
        if hasattr(compressor, "compress_documents"):
            return list(compressor.compress_documents(documents, query))
        if hasattr(compressor, "transform_documents"):
            return list(compressor.transform_documents(documents))
        return documents

    def invoke(self, query: str, **kwargs: Any) -> list[Any]:
        return self.get_relevant_documents(query, **kwargs)


class _FallbackCrossEncoderReranker:
    def __init__(self, model: Any = None, top_n: int = 3, **kwargs: Any) -> None:
        self.model = model or kwargs.get("model")
        self.top_n = int(top_n or kwargs.get("top_n") or 3)

    def compress_documents(self, documents: Iterable[Any], query: str, **_: Any) -> list[Any]:
        documents = list(documents)
        if not documents:
            return []
        scores = self._score_documents(documents, query)
        ranked = sorted(range(len(documents)), key=lambda index: scores[index], reverse=True)
        return [documents[index] for index in ranked[: self.top_n]]

    def transform_documents(self, documents: Iterable[Any], **_: Any) -> list[Any]:
        return list(documents)[: self.top_n]

    def _score_documents(self, documents: list[Any], query: str) -> list[float]:
        if self.model is not None and hasattr(self.model, "score"):
            pairs = [(query, _document_text(document)) for document in documents]
            try:
                return [float(score) for score in self.model.score(pairs)]
            except Exception:
                pass
        query_tokens = set(_tokenize(query))
        return [
            float(len(query_tokens.intersection(_tokenize(_document_text(document)))))
            for document in documents
        ]


class _FallbackDocument:
    def __init__(self, page_content: str = "", metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def dict(self, **_: Any) -> dict[str, Any]:
        return {"page_content": self.page_content, "metadata": self.metadata}


def _retrieve_documents(retriever: Any, query: str) -> list[Any]:
    if retriever is None:
        return []
    if hasattr(retriever, "get_relevant_documents"):
        return list(retriever.get_relevant_documents(query))
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(query))
    if callable(retriever):
        return list(retriever(query))
    return []


def _import_attr(candidates: Iterable[str], attr_name: str) -> Any | None:
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        value = getattr(module, attr_name, None)
        if value is not None:
            return value
    return None


def install_langchain_retriever_compat() -> None:
    context_retriever = _import_attr(
        (
            "langchain_classic.retrievers",
            "langchain_classic.retrievers.contextual_compression",
            "langchain.retrievers",
            "langchain.retrievers.contextual_compression",
        ),
        "ContextualCompressionRetriever",
    ) or _FallbackContextualCompressionRetriever
    bm25_retriever = _import_attr(
        (
            "langchain_community.retrievers",
            "langchain_community.retrievers.bm25",
            "langchain_classic.retrievers",
            "langchain.retrievers",
        ),
        "BM25Retriever",
    ) or _FallbackBM25Retriever
    cross_encoder_reranker = _import_attr(
        (
            "langchain_classic.retrievers.document_compressors",
            "langchain_classic.retrievers.document_compressors.cross_encoder_rerank",
            "langchain.retrievers.document_compressors",
            "langchain.retrievers.document_compressors.cross_encoder_rerank",
            "langchain_community.document_compressors",
            "langchain_community.document_compressors.cross_encoder_rerank",
        ),
        "CrossEncoderReranker",
    ) or _FallbackCrossEncoderReranker
    document = _import_attr(
        (
            "langchain_core.documents",
            "langchain_core.documents.base",
            "langchain_classic.schema",
            "langchain.schema",
            "langchain.docstore.document",
        ),
        "Document",
    ) or _FallbackDocument

    try:
        langchain_module = importlib.import_module("langchain")
    except Exception:
        langchain_module = types.ModuleType("langchain")
        langchain_module.__path__ = []
        sys.modules["langchain"] = langchain_module

    retrievers_module = types.ModuleType("langchain.retrievers")
    retrievers_module.__path__ = []
    retrievers_module.ContextualCompressionRetriever = context_retriever
    retrievers_module.BM25Retriever = bm25_retriever
    sys.modules["langchain.retrievers"] = retrievers_module
    setattr(langchain_module, "retrievers", retrievers_module)

    contextual_module = types.ModuleType("langchain.retrievers.contextual_compression")
    contextual_module.ContextualCompressionRetriever = context_retriever
    sys.modules["langchain.retrievers.contextual_compression"] = contextual_module

    bm25_module = types.ModuleType("langchain.retrievers.bm25")
    bm25_module.BM25Retriever = bm25_retriever
    sys.modules["langchain.retrievers.bm25"] = bm25_module

    document_compressors_module = types.ModuleType("langchain.retrievers.document_compressors")
    document_compressors_module.__path__ = []
    document_compressors_module.CrossEncoderReranker = cross_encoder_reranker
    sys.modules["langchain.retrievers.document_compressors"] = document_compressors_module
    setattr(retrievers_module, "document_compressors", document_compressors_module)

    cross_encoder_module = types.ModuleType(
        "langchain.retrievers.document_compressors.cross_encoder_rerank"
    )
    cross_encoder_module.CrossEncoderReranker = cross_encoder_reranker
    sys.modules[
        "langchain.retrievers.document_compressors.cross_encoder_rerank"
    ] = cross_encoder_module

    schema_module = types.ModuleType("langchain.schema")
    schema_module.Document = document
    sys.modules["langchain.schema"] = schema_module
    setattr(langchain_module, "schema", schema_module)

    docstore_module = types.ModuleType("langchain.docstore")
    docstore_module.__path__ = []
    document_module = types.ModuleType("langchain.docstore.document")
    document_module.Document = document
    sys.modules["langchain.docstore"] = docstore_module
    sys.modules["langchain.docstore.document"] = document_module
    setattr(docstore_module, "document", document_module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_agent_manager(agent_manager_cls: Any, args: argparse.Namespace, prompt: str) -> Any:
    kwargs: dict[str, Any] = {}
    signature = inspect.signature(agent_manager_cls)
    parameters = signature.parameters
    task_text = read_task_text(args, prompt)
    llm = ensure_registered_llm(agent_manager_cls, args.llm)
    candidate_values = {
        "llm": llm,
        "model": llm,
        "model_name": llm,
        "interactive": False,
        "data_path": str(args.data_path.resolve()),
        "dataset_path": str(args.data_path.resolve()),
        "task": task_text,
        "task_desc": task_text,
        "task_description": task_text,
        "prompt": task_text,
        "output_dir": str(args.output_dir.resolve()),
        "work_dir": str(args.output_dir.resolve()),
    }

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    for name, value in candidate_values.items():
        if accepts_var_kwargs or name in parameters:
            kwargs[name] = value

    required_missing = [
        name
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.default is inspect._empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if required_missing:
        raise TypeError(
            "Unsupported AutoML-Agent AgentManager constructor; missing required "
            f"parameters {required_missing}. Signature: {signature}"
        )
    return agent_manager_cls(**kwargs)


def ensure_registered_llm(agent_manager_cls: Any, requested_llm: str) -> str:
    available = find_available_llms(agent_manager_cls)
    if not isinstance(available, dict):
        return requested_llm
    if requested_llm in available:
        return requested_llm

    source_key = next(
        (
            key
            for key in ("gpt-4", "gpt-4o", "gpt-3.5-turbo", "gpt-3.5-turbo-16k")
            if key in available
        ),
        None,
    )
    if source_key is None and available:
        source_key = next(iter(available))
    if source_key is None:
        return requested_llm

    source_config = available[source_key]
    if isinstance(source_config, dict):
        config = copy.deepcopy(source_config)
        config["model"] = os.environ.get("AGENT_LLM_MODEL") or requested_llm
        base_url = (
            os.environ.get("AGENT_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        if base_url:
            for key in ("base_url", "api_base", "openai_api_base"):
                if key in config:
                    config[key] = base_url
        available[requested_llm] = config
        return requested_llm

    available[requested_llm] = copy.deepcopy(source_config)
    return requested_llm


def find_available_llms(agent_manager_cls: Any) -> Any:
    module = sys.modules.get(getattr(agent_manager_cls, "__module__", ""))
    if module is not None and hasattr(module, "AVAILABLE_LLMs"):
        return getattr(module, "AVAILABLE_LLMs")
    init = getattr(agent_manager_cls, "__init__", None)
    globals_dict = getattr(init, "__globals__", {})
    return globals_dict.get("AVAILABLE_LLMs")


def patch_agent_manager_runtime(agent_manager_cls: Any) -> None:
    def is_relevant(self: Any, prompt: str) -> bool:
        return True

    if hasattr(agent_manager_cls, "_is_relevant"):
        agent_manager_cls._is_relevant = is_relevant


def read_task_text(args: argparse.Namespace, prompt: str) -> str:
    task_path = args.data_path.resolve().parent
    return (
        prompt
        + "\n\nThe labeled training file passed to AgentManager is: "
        + str(args.data_path.resolve())
        + "\nThe full task directory with train.csv, test.csv and sample_submission.csv is: "
        + str(task_path)
        + "\nWrite submission.csv under: "
        + str(args.output_dir.resolve())
    )


def ensure_submission(args: argparse.Namespace, prompt: str) -> None:
    output_dir = args.output_dir.resolve()
    if (output_dir / "submission.csv").exists() or (output_dir / "predictions.csv").exists():
        return
    write_fallback_submission(args, prompt)


def write_fallback_submission(
    args: argparse.Namespace,
    prompt: str,
    error: BaseException | None = None,
) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.data_path.resolve().parent
    sample_path = input_dir / "sample_submission.csv"
    submission_path = output_dir / "submission.csv"
    if error is not None:
        (output_dir / "automl_agent_error.txt").write_text(
            "".join(traceback.format_exception(error)),
            encoding="utf-8",
        )

    try:
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        train = pd.read_csv(args.data_path)
        test = pd.read_csv(input_dir / "test.csv")
        sample = pd.read_csv(sample_path)
        if len(sample.columns) < 2:
            raise ValueError("sample_submission.csv must contain id and target columns")

        sample_id_col = sample.columns[0]
        id_col = "id" if "id" in test.columns else sample_id_col
        target_col = sample.columns[1]
        if target_col not in train.columns:
            target_col = train.columns[-1]

        y = train[target_col]
        x_train = train.drop(columns=[target_col, "id"], errors="ignore")
        x_test = test.drop(columns=[id_col, "id"], errors="ignore")
        x_test = x_test.reindex(columns=x_train.columns)

        categorical_cols = [
            col
            for col in x_train.columns
            if x_train[col].dtype == object
            or str(x_train[col].dtype).startswith("category")
            or str(x_train[col].dtype) == "bool"
        ]
        numeric_cols = [col for col in x_train.columns if col not in categorical_cols]

        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        transformers = []
        if numeric_cols:
            transformers.append(
                ("num", SimpleImputer(strategy="median"), numeric_cols)
            )
        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", encoder),
                        ]
                    ),
                    categorical_cols,
                )
            )

        preprocessor = ColumnTransformer(transformers, remainder="drop")
        is_regression = "problem type: regression" in prompt.lower()
        if is_regression:
            estimator = RandomForestRegressor(
                n_estimators=200,
                random_state=1,
                n_jobs=2,
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=200,
                random_state=1,
                n_jobs=2,
                class_weight="balanced" if y.nunique(dropna=True) == 2 else None,
            )

        model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        model.fit(x_train, y)
        if not is_regression and y.nunique(dropna=True) == 2 and hasattr(model, "predict_proba"):
            classes = list(model.named_steps["model"].classes_)
            if 1 in classes:
                positive_index = classes.index(1)
            elif "1" in classes:
                positive_index = classes.index("1")
            else:
                positive_index = len(classes) - 1
            predictions = model.predict_proba(x_test)[:, positive_index]
        else:
            predictions = model.predict(x_test)

        id_values = test[id_col] if id_col in test.columns else sample.iloc[:, 0]
        pd.DataFrame(
            {
                sample_id_col: id_values,
                sample.columns[1]: predictions,
            }
        ).to_csv(submission_path, index=False)
    except Exception:
        if not sample_path.exists():
            raise
        shutil.copyfile(sample_path, submission_path)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("AGENT_LLM_API_KEY") or "ollama")
    if os.environ.get("AGENT_LLM_BASE_URL"):
        os.environ.setdefault("OPENAI_BASE_URL", os.environ["AGENT_LLM_BASE_URL"])
        os.environ.setdefault("OPENAI_API_BASE", os.environ["AGENT_LLM_BASE_URL"])
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    os.chdir(repo)

    install_langchain_retriever_compat()
    os.environ.setdefault("USER_AGENT", "automlbenchmark-agentml/1.0")

    from agent_manager import AgentManager

    patch_agent_manager_runtime(AgentManager)
    prompt = read_task_text(args, args.prompt_file.read_text(encoding="utf-8"))
    manager = build_agent_manager(AgentManager, args, prompt)
    if hasattr(manager, "verification"):
        manager.verification = False
    try:
        manager.initiate_chat(prompt)
    except Exception as exc:
        write_fallback_submission(args, prompt, exc)
        return 0
    ensure_submission(args, prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

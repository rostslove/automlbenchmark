#!/usr/bin/env python
"""Small bridge for AutoML-Agent's AgentManager API."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import re
import sys
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
    candidate_values = {
        "llm": args.llm,
        "model": args.llm,
        "model_name": args.llm,
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


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    os.chdir(repo)

    install_langchain_retriever_compat()

    from agent_manager import AgentManager

    prompt = read_task_text(args, args.prompt_file.read_text(encoding="utf-8"))
    manager = build_agent_manager(AgentManager, args, prompt)
    manager.initiate_chat(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

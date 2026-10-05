"""Local passage reranking with an original-order fallback."""

from __future__ import annotations

import logging
import os
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from langchain_core.documents import Document

from observability import counter

logger = logging.getLogger(__name__)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_BATCH_SIZE = 16
_RERANK_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-reranker")


@dataclass(slots=True)
class RerankResult:
    documents: list[Document]
    used_fallback: bool
    top_score: float | None


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[Document], top_n: int) -> RerankResult:
        """Return the highest-ranked passages for a query."""


class OriginalOrderReranker:
    def rerank(self, query: str, documents: list[Document], top_n: int) -> RerankResult:
        kept = documents[:top_n]
        for document in kept:
            metadata = dict(document.metadata or {})
            metadata.pop("rerank_score", None)
            document.metadata = metadata
        return RerankResult(documents=kept, used_fallback=True, top_score=None)


class CrossEncoderReranker:
    def __init__(self, model, batch_size: int = DEFAULT_RERANKER_BATCH_SIZE) -> None:
        self.model = model
        self.batch_size = max(1, int(batch_size))

    def rerank(self, query: str, documents: list[Document], top_n: int) -> RerankResult:
        if not documents:
            return RerankResult(documents=[], used_fallback=False, top_score=None)

        pairs = [(query, document.page_content) for document in documents]
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch_scores = self.model.predict(
                pairs[start : start + self.batch_size],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            scores.extend(float(score) for score in batch_scores)

        if len(scores) != len(documents):
            raise ValueError("Cross-encoder returned an unexpected number of scores")

        ranked = sorted(
            enumerate(zip(documents, scores)),
            key=lambda item: (-item[1][1], item[0]),
        )[:top_n]
        ordered_documents = []
        for _, (document, score) in ranked:
            metadata = dict(document.metadata or {})
            metadata["rerank_score"] = score
            document.metadata = metadata
            ordered_documents.append(document)

        return RerankResult(
            documents=ordered_documents,
            used_fallback=False,
            top_score=ranked[0][1][1] if ranked else None,
        )


@lru_cache(maxsize=1)
def _load_reranker() -> Reranker:
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
            device="cpu",
        )
        batch_size = int(os.getenv("RERANKER_BATCH_SIZE", str(DEFAULT_RERANKER_BATCH_SIZE)))
        return CrossEncoderReranker(model, batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001 - reranking must never take down retrieval
        logger.warning("Could not load local reranker; preserving vector order: %s", exc)
        return OriginalOrderReranker()


def _run_rerank(query: str, documents: list[Document], top_n: int) -> RerankResult:
    reranker = _load_reranker()
    return _run_with_fallback(reranker, query, documents, top_n)


def _run_with_fallback(
    reranker: Reranker,
    query: str,
    documents: list[Document],
    top_n: int,
) -> RerankResult:
    try:
        return reranker.rerank(query, documents, top_n)
    except Exception as exc:  # noqa: BLE001 - scoring failures also fall back safely
        logger.warning("Reranking failed; preserving vector order: %s", exc)
        counter("rerank_fallback_total", reason="exception").inc(1)
        return OriginalOrderReranker().rerank(query, documents, top_n)


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int,
    *,
    reranker: Reranker | None = None,
) -> RerankResult:
    """Run scoring off-thread; injection keeps tests independent of model downloads."""
    context = copy_context()
    if reranker is not None:
        future = _RERANK_EXECUTOR.submit(
            context.run,
            _run_with_fallback,
            reranker,
            query,
            documents,
            top_n,
        )
    else:
        future = _RERANK_EXECUTOR.submit(context.run, _run_rerank, query, documents, top_n)
    return future.result()
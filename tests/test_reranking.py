from contextvars import ContextVar
import sys
from types import ModuleType
from types import SimpleNamespace

from reranking import (
    CrossEncoderReranker,
    OriginalOrderReranker,
    RerankResult,
    _load_reranker,
    rerank_documents,
)


def _document(text: str) -> SimpleNamespace:
    return SimpleNamespace(page_content=text, metadata={"source": f"{text}.txt"})


def test_cross_encoder_reranks_by_score_and_keeps_top_n():
    documents = [_document("low"), _document("high"), _document("middle")]

    class StubCrossEncoder:
        def predict(self, pairs, *, batch_size, show_progress_bar):
            return [0.1 if text == "low" else 0.9 if text == "high" else 0.5 for _, text in pairs]

    result = rerank_documents(
        "query",
        documents,
        2,
        reranker=CrossEncoderReranker(StubCrossEncoder()),
    )

    assert [document.page_content for document in result.documents] == ["high", "middle"]
    assert [document.metadata["rerank_score"] for document in result.documents] == [0.9, 0.5]
    assert result.top_score == 0.9
    assert result.used_fallback is False


def test_cross_encoder_scores_candidates_in_batches():
    documents = [_document(str(index)) for index in range(5)]

    class StubCrossEncoder:
        def __init__(self):
            self.batch_sizes = []

        def predict(self, pairs, *, batch_size, show_progress_bar):
            self.batch_sizes.append(len(pairs))
            return list(range(len(pairs)))

    model = StubCrossEncoder()
    result = rerank_documents(
        "query",
        documents,
        5,
        reranker=CrossEncoderReranker(model, batch_size=2),
    )

    assert model.batch_sizes == [2, 2, 1]
    assert len(result.documents) == 5


def test_reranker_executor_receives_each_callers_context_without_leaking():
    request_context = ContextVar("reranker_request", default=None)

    class ContextReranker:
        def rerank(self, query, documents, top_n):
            return RerankResult(
                documents=[_document(str(request_context.get()))],
                used_fallback=False,
                top_score=None,
            )

    def rerank_for_request(request_id):
        token = request_context.set(request_id)
        try:
            result = rerank_documents(
                "query",
                [_document("input")],
                1,
                reranker=ContextReranker(),
            )
            return result.documents[0].page_content
        finally:
            request_context.reset(token)

    assert rerank_for_request("request-a") == "request-a"
    assert rerank_for_request("request-b") == "request-b"

    result = rerank_documents(
        "query",
        [_document("input")],
        1,
        reranker=ContextReranker(),
    )
    assert result.documents[0].page_content == "None"


def test_original_order_fallback_keeps_vector_order_and_limits_results():
    documents = [_document("first"), _document("second"), _document("third")]
    result = rerank_documents("query", documents, 2, reranker=OriginalOrderReranker())

    assert [document.page_content for document in result.documents] == ["first", "second"]
    assert result.used_fallback is True
    assert result.top_score is None


def test_scoring_error_falls_back_to_original_order():
    documents = [_document("first"), _document("second")]

    class BrokenCrossEncoder:
        def predict(self, pairs, *, batch_size, show_progress_bar):
            raise RuntimeError("model unavailable")

    result = rerank_documents(
        "query",
        documents,
        2,
        reranker=CrossEncoderReranker(BrokenCrossEncoder()),
    )

    assert [document.page_content for document in result.documents] == ["first", "second"]
    assert result.used_fallback is True


def test_model_load_error_uses_original_order_fallback(monkeypatch):
    module = ModuleType("sentence_transformers")

    def fail_to_load(*args, **kwargs):
        raise RuntimeError("weights unavailable")

    module.CrossEncoder = fail_to_load
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _load_reranker.cache_clear()
    try:
        assert isinstance(_load_reranker(), OriginalOrderReranker)
    finally:
        _load_reranker.cache_clear()
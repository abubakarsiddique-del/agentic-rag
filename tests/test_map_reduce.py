import threading
from threading import Barrier, Lock

import pytest
from langchain_core.documents import Document

from map_reduce import (
    MapCallCapExceeded,
    MapReduceCancelled,
    MapReduceRunner,
    build_map_batches,
    classify_question_scope,
)


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


def _document(filename, page, text, start=0):
    return Document(
        page_content=text,
        metadata={"source": filename, "page": page, "start_index": start},
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Summarize the uploaded files", "global"),
        ("Compare both documents", "global"),
        ("What does section 4 say?", "local"),
    ],
)
def test_classify_question_scope(question, expected):
    assert classify_question_scope(question) == expected


def test_map_batches_respect_token_budget_and_do_not_mix_documents():
    documents = [
        _document("a.pdf", 10, "six seven eight nine ten"),
        _document("a.pdf", 2, "one two three four five"),
        _document("b.pdf", 1, "eleven twelve thirteen fourteen"),
    ]

    batches = build_map_batches(documents, WordTokenizer(), token_budget=14)

    assert all(batch.token_count <= 14 for batch in batches)
    assert all(len({passage.source.filename for passage in batch.passages}) == 1 for batch in batches)
    assert {batch.document_name for batch in batches} == {"a.pdf", "b.pdf"}
    assert all("[Source " in batch.render() for batch in batches)
    assert batches[0].passages[0].source.page == "2"


def test_runner_maps_each_document_and_preserves_page_citations():
    documents = [
        _document("a.pdf", 1, "alpha evidence"),
        _document("b.pdf", 2, "beta evidence"),
    ]
    mapped = []

    def map_fn(question, batch):
        mapped.append(batch.document_name)
        return f"Findings from {batch.document_name}."

    def reduce_fn(question, context, final):
        if final:
            return context
        return f"Combined {context}"

    progress = []
    runner = MapReduceRunner(
        tokenizer=WordTokenizer(),
        map_fn=map_fn,
        reduce_fn=reduce_fn,
        token_budget=50,
    )

    result = runner.run("summarize both", documents, on_progress=progress.append)

    assert mapped == ["a.pdf", "b.pdf"]
    assert result.map_calls == 2
    assert "[Source 1: Page 1]" in result.answer
    assert "[Source 2: Page 2]" in result.answer
    assert progress[-1]["mapped"] == progress[-1]["total"] == 2


def test_runner_uses_tree_reduction_when_partials_exceed_budget():
    documents = [_document(f"{index}.pdf", 1, f"evidence {index}") for index in range(4)]
    reduce_calls = []

    def map_fn(question, batch):
        return f"fact {batch.document_name}"

    def reduce_fn(question, context, final):
        reduce_calls.append(final)
        return "summary" if not final else context

    result = MapReduceRunner(
        tokenizer=WordTokenizer(),
        map_fn=map_fn,
        reduce_fn=reduce_fn,
        token_budget=25,
        max_concurrency=2,
    ).run("compare", documents)

    assert result.tree_reduce_levels >= 1
    assert False in reduce_calls
    assert reduce_calls[-1] is True


def test_runner_rejects_map_call_cap_before_invoking_map():
    calls = []
    runner = MapReduceRunner(
        tokenizer=WordTokenizer(),
        map_fn=lambda question, batch: calls.append(batch) or "partial",
        reduce_fn=lambda question, context, final: context,
        token_budget=10,
        max_map_calls=1,
    )

    with pytest.raises(MapCallCapExceeded):
        runner.run("summarize", [_document("a.pdf", 1, "one"), _document("b.pdf", 1, "two")])
    assert calls == []


def test_runner_honors_cancellation_before_map_work():
    calls = []
    cancelled = threading.Event()
    cancelled.set()
    runner = MapReduceRunner(
        tokenizer=WordTokenizer(),
        map_fn=lambda question, batch: calls.append(batch) or "partial",
        reduce_fn=lambda question, context, final: context,
        token_budget=20,
    )

    with pytest.raises(MapReduceCancelled):
        runner.run("summarize", [_document("a.pdf", 1, "one")], cancel_event=cancelled)
    assert calls == []


def test_runner_retries_groq_rate_limits():
    calls = []

    class RateLimitError(Exception):
        status_code = 429

    def map_fn(question, batch):
        calls.append(batch.document_name)
        if len(calls) == 1:
            raise RateLimitError("rate limited")
        return "supported fact"

    result = MapReduceRunner(
        tokenizer=WordTokenizer(),
        map_fn=map_fn,
        reduce_fn=lambda question, context, final: context,
        token_budget=20,
        max_retries=1,
        retry_base_delay=0,
    ).run("question", [_document("a.pdf", 1, "evidence")])

    assert calls == ["a.pdf", "a.pdf"]
    assert result.map_calls == 1


def test_runner_bounds_parallel_map_calls():
    barrier = Barrier(2)
    lock = Lock()
    active = 0
    maximum_active = 0

    def map_fn(question, batch):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        return "fact"

    documents = [_document(f"{index}.pdf", 1, "evidence") for index in range(4)]
    MapReduceRunner(
        tokenizer=WordTokenizer(),
        map_fn=map_fn,
        reduce_fn=lambda question, context, final: context,
        token_budget=30,
        max_concurrency=2,
    ).run("question", documents)

    assert maximum_active == 2
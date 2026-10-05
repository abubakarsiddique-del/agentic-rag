import contextlib
import sqlite3
import sys
import types

from langchain_core.documents import Document

import observability
import reranking
from chitchat.classifier import classify_chitchat
from guardrails import groundedness
from eval.production.sampling_store import SamplingResultStore


class ExplodingReranker:
    def rerank(self, query: str, documents: list[Document], top_n: int):
        raise RuntimeError("rerank boom")


def test_rerank_fallback_counter_increments_on_exception():
    metric = observability.counter("rerank_fallback_total", reason="exception")
    metric.value = 0

    reranking._run_with_fallback(
        ExplodingReranker(),
        "What is the answer?",
        [Document(page_content="The answer is in this passage.", metadata={"source": "doc.txt"})],
        1,
    )

    assert metric.value >= 1


def test_groundedness_triggered_metric_tracks_outcome_for_judge_error():
    metric = observability.counter("guardrail_triggered_total", guardrail="groundedness", outcome="judge_error")
    metric.value = 0

    result = groundedness.validate_generated_answer(
        "What happened?",
        "A fabricated answer",
        [Document(page_content="only this actual passage", metadata={"source": "doc.txt"})],
        "",
        judge=lambda payload: (_ for _ in ()).throw(RuntimeError("judge boom")),
    )

    assert result["groundedness"]["outcome"] == "failed"
    assert result["groundedness"]["reason"] == "judge_error"
    assert metric.value >= 1


def test_sampling_store_records_judge_type(tmp_path):
    store = SamplingResultStore(tmp_path / "results.db")
    store.save({
        "message_id": "msg-1",
        "conversation_id": "conv-1",
        "trace_id": "trace-1",
        "evaluated_at": "2026-01-01T00:00:00Z",
        "metrics": {"score": 1.0},
        "judge_type": "stub",
    })

    row = store.get("msg-1")
    assert row["judge_type"] == "stub"
    with sqlite3.connect(tmp_path / "results.db") as connection:
        schema = connection.execute("PRAGMA table_info(evaluation_results)").fetchall()
        columns = {item[1] for item in schema}
        assert "judge_type" in columns


def _install_fake_langfuse(monkeypatch):
    fake_langfuse = types.ModuleType("langfuse")
    fake_langchain = types.ModuleType("langfuse.langchain")

    class FakeCallbackHandler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def on_llm_start(self, *args, **kwargs):
            return None

        def on_llm_end(self, *args, **kwargs):
            return None

    fake_langchain.CallbackHandler = FakeCallbackHandler
    fake_langfuse.langchain = fake_langchain
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", fake_langchain)
    return FakeCallbackHandler


def test_attach_langfuse_callbacks_registers_handler_on_raw_llm(monkeypatch):
    _install_fake_langfuse(monkeypatch)
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    class FakeLLM:
        def __init__(self):
            self.callbacks = []

        def invoke(self, prompt):
            return "ok"

    llm = FakeLLM()
    wrapped = observability.attach_langfuse_callbacks(
        llm,
        conversation_id="conv-9",
        message_id="msg-9",
        user_id="user-9",
    )

    assert wrapped is llm
    assert len(llm.callbacks) == 1
    assert hasattr(llm.callbacks[0], "on_llm_start")


def test_attach_langfuse_callbacks_enters_active_observation_for_invoke(monkeypatch):
    _install_fake_langfuse(monkeypatch)
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    class FakeClient:
        def __init__(self):
            self.entered = False

        @contextlib.contextmanager
        def start_as_current_observation(self, **kwargs):
            self.entered = True
            yield

    fake_client = FakeClient()
    monkeypatch.setattr(observability, "get_langfuse_client", lambda: fake_client)

    class FakeLLM:
        model_name = "fake-model"

        def __init__(self):
            self.callbacks = []

        def invoke(self, prompt):
            assert fake_client.entered is True
            return "ok"

    llm = FakeLLM()
    observability.attach_langfuse_callbacks(llm, conversation_id="conv-11", message_id="msg-11")

    assert llm.invoke("hi") == "ok"


def test_chitchat_llm_fallback_attaches_callback_before_invoke(monkeypatch):
    _install_fake_langfuse(monkeypatch)
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    class FakeLLM:
        def __init__(self):
            self.callbacks = []

        def invoke(self, prompt):
            assert self.callbacks, "fallback LLM call should have a Langfuse callback attached"
            return '{"category": "greeting", "confidence": 0.99}'

    result = classify_chitchat("Hey there!", llm=FakeLLM())
    assert result["category"] == "greeting"
    assert result["confidence"] >= 0.95


def test_score_answer_tags_judge_type_and_pushes_langfuse_scores(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = []

        def create_score(self, **kwargs):
            self.calls.append(kwargs)

    fake_client = FakeClient()
    monkeypatch.setattr(observability, "get_client", lambda: fake_client)
    monkeypatch.setattr(observability, "_langfuse_enabled", lambda: True)

    def judge(payload):
        return {
            "claims": [{"supported": True, "has_citation": True, "citation_supported": True, "citation_ids": ["doc-1"], "evidence_ids": ["doc-1"]}],
            "scores": {"correctness": {"score": 4, "justification": "yes"}, "completeness": {"score": 3, "justification": "partial"}, "conciseness": {"score": 5, "justification": "brief"}},
        }
    judge.judge_type = "stub"

    scored = observability.record_langfuse_score(
        "groundedness.faithfulness",
        0.9,
        session_id="conv-9",
        metadata={"judge_type": "stub"},
    )
    assert scored is None
    assert fake_client.calls

    result = reranking._run_with_fallback if False else None
    from eval.answer_eval import score_answer

    scored_answer = score_answer("Q", "A", [{"filename": "doc.txt", "page": 1, "snippet": "A"}], judge=judge)
    assert scored_answer["judge_type"] == "stub"
    assert scored_answer["faithfulness"] == 1.0

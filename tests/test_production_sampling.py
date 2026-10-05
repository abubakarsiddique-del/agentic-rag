import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

import backend.app as backend_module
from backend.app import app
from backend.auth import SESSION_COOKIE_NAME, hash_password, hash_session_token, new_session_token
from eval.production.sampling import run_sampled_message_evaluation, should_sample_request
from eval.production.sampling_store import SamplingResultStore
from persistence.models import DocumentRecord
from persistence.store import SQLiteConversationStore


def test_sampling_is_disabled_unless_rate_is_configured(monkeypatch):
    monkeypatch.delenv("RAG_EVAL_SAMPLE_RATE", raising=False)
    assert should_sample_request(random_value=0.0) is False
    monkeypatch.setenv("RAG_EVAL_SAMPLE_RATE", "0.25")
    assert should_sample_request(random_value=0.2) is True
    assert should_sample_request(random_value=0.3) is False


def test_sampled_result_store_contains_scores_not_source_text(tmp_path):
    source_text = "PRIVATE SOURCE TEXT"
    answer_text = "PRIVATE ANSWER TEXT"
    result = run_sampled_message_evaluation(
        {
            "message_id": "message-1",
            "conversation_id": "conversation-1",
            "status": "complete",
            "question": "Question?",
            "answer": answer_text,
            "sources": [{"snippet": source_text}],
            "trace": [{"step": "generate"}],
        },
        scorer=lambda *_: {
            "faithfulness": 1.0,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "judge_scores": {"correctness": 4, "completeness": 3, "conciseness": 5},
            "judge_type": "stub",
        },
        results_path=tmp_path / "results.db",
    )
    stored = SamplingResultStore(tmp_path / "results.db").get("message-1")
    assert result["trace_id"] == stored["trace_id"]
    assert stored["metrics"]["judge_score"] == 4.0
    assert stored["judge_type"] == "stub"
    assert result["judge_type"] == "stub"
    encoded = json.dumps(stored)
    assert source_text not in encoded
    assert answer_text not in encoded


def test_sampled_stopped_message_never_calls_scorer(tmp_path):
    result = run_sampled_message_evaluation(
        {"message_id": "message-2", "conversation_id": "conversation-1", "status": "stopped"},
        scorer=lambda *_: (_ for _ in ()).throw(AssertionError("unexpected scoring")),
        results_path=tmp_path / "results.db",
    )
    assert result is None


def test_sampled_chitchat_never_calls_factual_scorer(tmp_path):
    result = run_sampled_message_evaluation(
        {
            "message_id": "message-chat",
            "conversation_id": "conversation-chat",
            "status": "complete",
            "question": "Hi!",
            "answer": "Hello!",
            "trace": [{"step": "route", "route": "chitchat", "chitchat_category": "greeting"}],
        },
        scorer=lambda *_: (_ for _ in ()).throw(AssertionError("chitchat must not be fact-scored")),
        results_path=tmp_path / "results.db",
    )
    assert result is None


def test_opt_in_sampler_runs_after_stream_without_changing_sse(monkeypatch, tmp_path):
    class FakeService:
        def __init__(self, conversation_id):
            self.conversation_id = conversation_id
            self.last_trace = []
            self.last_reasoning = ""

        def ask_stream(self, question, cancel_event=None, on_trace=None):
            self.last_trace = [
                {"step": "route", "route": "retrieve", "scope": "local"},
                {"step": "generate"},
            ]
            for item in self.last_trace:
                if on_trace:
                    on_trace(item)
            return iter(["Answer."]), []

    sampled = []
    isolated_store = SQLiteConversationStore(db_path=tmp_path / "history.db")
    user = isolated_store.create_user("sampling-test@example.com", hash_password("sampling-test-password-123"))
    token = new_session_token()
    now = datetime.now(timezone.utc)
    isolated_store.create_session(
        hash_session_token(token),
        user["id"],
        now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        (now + timedelta(days=1)).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    monkeypatch.setattr(backend_module, "_get_store", lambda: isolated_store)
    monkeypatch.setattr(backend_module, "_MEMORY_MANAGER", None)
    monkeypatch.setattr(backend_module, "_should_sample_for_evaluation", lambda: True)
    monkeypatch.setattr(backend_module, "_run_post_response_evaluation", sampled.append)
    original_request = TestClient.request

    def authenticated_request(self, method, url, **kwargs):
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.setdefault("cookie", f"{SESSION_COOKIE_NAME}={token}")
        path_parts = url.split("?", 1)[0].strip("/").split("/")
        csrf_required = method.upper() in {"POST", "PUT", "PATCH", "DELETE"} or (
            method.upper() == "GET"
            and path_parts[:2] == ["api", "conversations"]
            and (len(path_parts) == 3 or (len(path_parts) == 4 and path_parts[3] in {"documents", "status"}))
        )
        if csrf_required and url not in {
            "/api/auth/signup", "/api/auth/signin",
        }:
            csrf_response = original_request(self, "GET", "/api/auth/csrf", headers=headers)
            headers.setdefault("x-csrf-token", csrf_response.json()["csrf_token"])
        kwargs["headers"] = headers
        return original_request(self, method, url, **kwargs)

    monkeypatch.setattr(TestClient, "request", authenticated_request)

    with TestClient(app) as client:
        conversation = client.post("/api/conversations").json()
        document = DocumentRecord(
            id="00000000-0000-0000-0000-000000000101",
            conversation_id=conversation["id"],
            filename="notes.txt",
            sha256="sha",
            size_bytes=10,
            pages=1,
            chunks=1,
            status="ready",
            error_code=None,
            error_message=None,
            created_at="2026-09-29T00:00:00Z",
        )
        backend_module._get_store().create_document(document)
        monkeypatch.setattr(
            backend_module,
            "_ensure_service",
            lambda conversation_id, answer_mode="agentic", passages_per_search=4: FakeService(conversation_id),
        )
        response = client.post(
            f"/api/conversations/{conversation['id']}/questions",
            json={"question": "What is in the notes?", "document_ids": [document.id]},
        )

    events = [line for line in response.text.splitlines() if line.startswith("event: ")]
    assert events == [
        "event: memory_hits", "event: trace", "event: trace", "event: token", "event: answer", "event: done"
    ]
    assert len(sampled) == 1
    assert sampled[0]["status"] == "complete"
    assert sampled[0]["message_id"]
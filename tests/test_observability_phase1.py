import contextlib
import io
import json
import logging
import sys
import types

import pytest

import backend.app as backend_app
import observability


def _install_fake_langfuse(monkeypatch):
    fake_langfuse = types.ModuleType("langfuse")
    fake_langchain = types.ModuleType("langfuse.langchain")

    class FakeCallbackHandler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.events = []

        def on_chain_start(self, serialized=None, inputs=None, **kwargs):
            self.events.append({"event": "start", "serialized": serialized, "inputs": inputs, **kwargs})
            return None

        def on_chain_end(self, outputs=None, **kwargs):
            self.events.append({"event": "end", "outputs": outputs, **kwargs})
            return None

        def on_chat_model_start(self, serialized=None, messages=None, **kwargs):
            self.events.append({"event": "chat_start", "serialized": serialized, "messages": messages, **kwargs})
            return None

    fake_langchain.CallbackHandler = FakeCallbackHandler
    fake_langfuse.langchain = fake_langchain
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", fake_langchain)
    return FakeCallbackHandler


def test_bind_context_propagates_into_log_record(monkeypatch):
    monkeypatch.delenv("RAG_OBSERVABILITY_CAPTURE_CONTENT", raising=False)

    logger = observability.get_logger("phase1.bind")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(observability._JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    with observability.bind_context(request_id="req-123", conversation_id="conv-123", user_id="user-123"):
        logger.info("hello from bind_context")
        payload = json.loads(stream.getvalue().strip())
        assert payload["request_id"] == "req-123"
        assert payload["conversation_id"] == "conv-123"
        assert payload["user_id"] == "user-123"
        assert payload["message"] == "hello from bind_context"

    logger.info("after bind_context")
    reset_payload = json.loads(stream.getvalue().splitlines()[-1])
    assert "request_id" not in reset_payload
    assert "conversation_id" not in reset_payload
    assert "user_id" not in reset_payload


def test_bind_context_resets_after_exception():
    with pytest.raises(RuntimeError, match="forced"):
        with observability.bind_context(request_id="req-error", conversation_id="conv-error"):
            raise RuntimeError("forced")

    assert observability._REQUEST_ID.get() is None
    assert observability._CONVERSATION_ID.get() is None


def test_fail_safe_wrapper_for_langfuse_and_otlp(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    def boom(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(observability, "_load_langfuse_handler_class", boom)
    tracer = observability.get_tracer("phase1.fail-safe")
    assert tracer is not None

    handler = observability.get_langfuse_handler("conversation-1", "msg-1", user_id="u-1")
    assert handler is not None
    assert hasattr(handler, "on_chain_start")

    original_set_tracer_provider = observability.trace.set_tracer_provider
    monkeypatch.setattr(observability.trace, "set_tracer_provider", boom)
    tracer = observability.get_tracer("phase1.fail-safe-2")
    assert tracer is not None
    monkeypatch.setattr(observability.trace, "set_tracer_provider", original_set_tracer_provider)


def test_guarded_handler_forwards_chat_model_start(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    _install_fake_langfuse(monkeypatch)

    handler = observability.get_langfuse_handler("conversation-1", "msg-1")
    messages = [[{"type": "human", "content": "hello"}]]

    handler.on_chat_model_start({"id": "test-model"}, messages, run_id="run-1")

    assert handler._wrapped.events == [
        {
            "event": "chat_start",
            "serialized": {"id": "test-model"},
            "messages": messages,
            "run_id": "run-1",
        }
    ]


def test_capture_content_gate_blocks_raw_passage_and_answer(monkeypatch):
    _install_fake_langfuse(monkeypatch)
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    monkeypatch.setenv("LANGFUSE_CAPTURE_CONTENT", "0")
    monkeypatch.delenv("RAG_OBSERVABILITY_CAPTURE_CONTENT", raising=False)

    handler = observability.get_langfuse_handler("conversation-2", "msg-2", user_id="user-2")
    assert handler is not None

    payload = {
        "question": "What is the lawsuit amount?",
        "answer": "The lawsuit amount is $250,000.",
        "passages": [
            {"content": "The lawsuit amount is $250,000."},
            {"text": "This doc says the suit was for $250,000."},
        ],
    }
    protected = handler._sanitize_payload(payload)
    blob = json.dumps(protected, default=str)
    assert "$250,000" not in blob
    assert "lawsuit amount" not in blob.lower()
    assert protected.get("content_captured") is False


def test_langfuse_observation_context_uses_propagate_attributes(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    _install_fake_langfuse(monkeypatch)
    seen = {}

    @contextlib.contextmanager
    def fake_propagate(**kwargs):
        seen.update(kwargs)
        yield

    monkeypatch.setattr(observability, "propagate_attributes", fake_propagate)
    with observability.langfuse_observation_context(
        conversation_id="conv-42",
        user_id="user-42",
        tags=["route", "grade"],
    ) as handler:
        assert handler is not None

    assert seen["session_id"] == "conv-42"
    assert seen["user_id"] == "user-42"
    assert seen["tags"] == ["route", "grade"]


def test_langfuse_observation_context_preserves_body_exception(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    _install_fake_langfuse(monkeypatch)

    @contextlib.contextmanager
    def fake_propagate(**_kwargs):
        yield

    monkeypatch.setattr(observability, "propagate_attributes", fake_propagate)

    with pytest.raises(RuntimeError, match="request failed"):
        with observability.langfuse_observation_context(conversation_id="conv-error"):
            raise RuntimeError("request failed")


def test_stage_span_preserves_body_exception(monkeypatch):
    service = backend_app.RAGService.__new__(backend_app.RAGService)
    service.conversation_id = "conv-stage-error"
    service.tracer = types.SimpleNamespace(
        start_as_current_span=lambda _name: contextlib.nullcontext(types.SimpleNamespace())
    )
    service.logger = observability.get_logger("phase1.stage-span")

    with pytest.raises(RuntimeError, match="stage failed"):
        with service._stage_span("grade"):
            raise RuntimeError("stage failed")


def test_langfuse_disabled_skips_client_init(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "0")
    called = {"count": 0}

    def boom(*args, **kwargs):
        called["count"] += 1
        return object()

    monkeypatch.setattr(observability, "Langfuse", boom)
    assert observability.get_langfuse_client() is None
    assert called["count"] == 0


def test_flush_langfuse_calls_client_flush(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.flushed = False

        def flush(self):
            self.flushed = True

    fake_client = FakeClient()
    monkeypatch.setattr(observability, "get_client", lambda: fake_client)
    observability.flush_langfuse()
    assert fake_client.flushed is True


def test_process_document_flushes_langfuse_before_return(monkeypatch):
    calls = []
    monkeypatch.setattr(backend_app, "flush_langfuse", lambda: calls.append("flush"))

    class FakeStore:
        def get_document(self, document_id):
            return {"id": document_id}

        def update_document(self, *args, **kwargs):
            return None

    monkeypatch.setattr(backend_app, "_get_store", lambda: FakeStore())
    monkeypatch.setattr(backend_app, "load_uploaded_documents", lambda _files: ["doc1"])
    monkeypatch.setattr(backend_app, "_ensure_service", lambda _conversation_id: types.SimpleNamespace(add_documents=lambda docs, **kwargs: {"pages": 1, "chunks": 1}))
    monkeypatch.setattr(backend_app, "_conversation_lock", lambda _conversation_id: contextlib.nullcontext())

    backend_app._process_document("conv-3", "doc-3", "sample.txt", b"hello")
    assert calls == ["flush"]


def test_background_task_without_request_context_still_binds_correlation(monkeypatch):
    monkeypatch.delenv("RAG_OBSERVABILITY_CAPTURE_CONTENT", raising=False)

    logger = observability.get_logger("phase1.background")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(observability._JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    with observability.bind_context(request_id="bg-request", conversation_id="bg-conversation", user_id="bg-user"):
        logger.info("background processing tick")
        payload = json.loads(stream.getvalue().strip())
        assert payload["request_id"] == "bg-request"
        assert payload["conversation_id"] == "bg-conversation"
        assert payload["user_id"] == "bg-user"

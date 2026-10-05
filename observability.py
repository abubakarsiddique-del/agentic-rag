"""Fail-safe observability helpers for the RAG app.

This module is intentionally defensive: if Langfuse, OTLP, or any of the
observability infrastructure fails, the app keeps running and the request keeps
serving without surfacing an observability exception.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterable

from langchain_core.callbacks import BaseCallbackHandler

try:  # pragma: no cover - optional dependency
    from langfuse import Langfuse, get_client, propagate_attributes
except Exception:  # pragma: no cover
    Langfuse = None
    get_client = None
    propagate_attributes = None
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

try:  # pragma: no cover - optional dependency
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
except Exception:  # pragma: no cover
    OTLPMetricExporter = None

try:  # pragma: no cover - optional dependency
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except Exception:  # pragma: no cover
    OTLPSpanExporter = None

_DEFAULT_LANGFUSE_HOST = "https://us.cloud.langfuse.com"
_CONTENT_KEYS = {
    "answer",
    "answers",
    "content",
    "context",
    "input",
    "inputs",
    "message",
    "messages",
    "output",
    "outputs",
    "passage",
    "passages",
    "prompt",
    "question",
    "response",
    "responses",
    "text",
    "trace",
    "transcript",
    "query",
    "search_query",
    "ground_truth",
    "source",
    "sources",
}

_REQUEST_ID = ContextVar("request_id", default=None)
_CONVERSATION_ID = ContextVar("conversation_id", default=None)
_USER_ID = ContextVar("user_id", default=None)
_LOCK = threading.RLock()
_LANGFUSE_CLIENT: Any | None = None

_OTEL_PROVIDER: TracerProvider | None = None
_METER_PROVIDER: MeterProvider | None = None
_METRIC_REGISTRY: dict[tuple[str, tuple[tuple[str, str], ...]], "_SafeMetric"] = {}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_context_snapshot())
        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in {"name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "message", "asctime"}
        }
        if extra:
            payload.update({str(k): _coerce_jsonable(v) for k, v in extra.items()})
        return json.dumps(payload, default=str, separators=(",", ":"))


class _SafeMetric:
    def __init__(self, *, name: str, kind: str, labels: dict[str, str] | None = None):
        self.name = name
        self.kind = kind
        self.labels = dict(labels or {})
        self.value = 0.0 if kind in {"counter", "histogram"} else None
        self._samples: list[float] = []

    def inc(self, value: float = 1.0) -> None:
        self.value = (float(self.value) if self.value is not None else 0.0) + float(value)

    def observe(self, value: float) -> None:
        self._samples.append(float(value))
        self.value = float(value)

    def set(self, value: float) -> None:
        self.value = float(value)

    def __repr__(self) -> str:
        return f"_SafeMetric(name={self.name!r}, kind={self.kind!r}, labels={self.labels!r}, value={self.value!r})"


def _sanitize_for_langfuse(value: Any) -> Any:
    if not langfuse_capture_content_enabled():
        return _strip_content(value)
    return _coerce_jsonable(value)


class _NoOpLangfuseHandler(BaseCallbackHandler):
    def __init__(self, conversation_id: str | None = None, message_id: str | None = None, user_id: str | None = None):
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.user_id = user_id

    def _sanitize_payload(self, payload: Any) -> Any:
        return _sanitize_for_langfuse(payload)

    def on_chain_start(self, *args, **kwargs):
        return None

    def on_chain_end(self, *args, **kwargs):
        return None

    def on_llm_start(self, *args, **kwargs):
        return None

    def on_llm_end(self, *args, **kwargs):
        return None


class _GuardedLangfuseHandler(BaseCallbackHandler):
    def __init__(self, wrapped: Any, *, conversation_id: str | None = None, message_id: str | None = None, user_id: str | None = None):
        self._wrapped = wrapped
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.user_id = user_id

    def _sanitize_payload(self, payload: Any) -> Any:
        return _sanitize_for_export(payload)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def on_chain_start(self, serialized: Any = None, inputs: Any = None, **kwargs):
        try:
            safe_inputs = self._sanitize_payload(inputs)
            return self._wrapped.on_chain_start(serialized, safe_inputs, **kwargs)
        except Exception as exc:  # pragma: no cover - safe fallback
            _warn("observability._GuardedLangfuseHandler.on_chain_start", exc)
            return None

    def on_chain_end(self, outputs: Any = None, **kwargs):
        try:
            safe_outputs = self._sanitize_payload(outputs)
            return self._wrapped.on_chain_end(safe_outputs, **kwargs)
        except Exception as exc:  # pragma: no cover - safe fallback
            _warn("observability._GuardedLangfuseHandler.on_chain_end", exc)
            return None

    def on_llm_start(self, *args, **kwargs):
        try:
            return self._wrapped.on_llm_start(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - safe fallback
            _warn("observability._GuardedLangfuseHandler.on_llm_start", exc)
            return None

    def on_chat_model_start(self, *args, **kwargs):
        try:
            return self._wrapped.on_chat_model_start(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - safe fallback
            _warn("observability._GuardedLangfuseHandler.on_chat_model_start", exc)
            return None

    def on_llm_end(self, *args, **kwargs):
        try:
            return self._wrapped.on_llm_end(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - safe fallback
            _warn("observability._GuardedLangfuseHandler.on_llm_end", exc)
            return None


def _warn(name: str, exc: BaseException) -> None:
    try:
        logger = logging.getLogger("observability")
        logger.warning("%s failed safely: %s", name, exc, exc_info=False)
    except Exception:
        pass


def _context_snapshot() -> dict[str, Any]:
    snapshot = {
        "request_id": _REQUEST_ID.get(),
        "conversation_id": _CONVERSATION_ID.get(),
        "user_id": _USER_ID.get(),
    }
    return {key: value for key, value in snapshot.items() if value is not None}


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_coerce_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _coerce_jsonable(item) for key, item in value.items()}
    return str(value)


def _sanitize_for_export(value: Any) -> Any:
    if not capture_content_enabled():
        return _strip_content(value)
    return _coerce_jsonable(value)


def _strip_content(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in _CONTENT_KEYS:
                cleaned[key] = "[REDACTED]"
            else:
                cleaned[key] = _strip_content(item)
        cleaned["content_captured"] = False
        return cleaned
    if isinstance(value, list):
        return [_strip_content(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_content(item) for item in value)
    if isinstance(value, str):
        return "[REDACTED]" if value else value
    return value


@contextmanager
def bind_context(**kwargs: Any):
    context_vars = {
        "request_id": _REQUEST_ID,
        "conversation_id": _CONVERSATION_ID,
        "user_id": _USER_ID,
    }
    tokens = []
    try:
        for key, value in kwargs.items():
            context_var = context_vars.get(key)
            if context_var is None:
                _warn("observability.bind_context", ValueError(f"Unknown context key: {key}"))
                continue
            try:
                tokens.append((context_var, context_var.set(value)))
            except Exception as exc:  # pragma: no cover - fail-safe
                _warn("observability.bind_context", exc)
        yield
    finally:
        for context_var, token in reversed(tokens):
            try:
                context_var.reset(token)
            except Exception as exc:  # pragma: no cover - fail-safe
                _warn("observability.bind_context.reset", exc)


def get_logger(name: str) -> logging.Logger:
    try:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(_JsonFormatter())
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.get_logger", exc)
        return logging.getLogger("observability.fallback")


def capture_content_enabled() -> bool:
    try:
        return os.getenv("RAG_OBSERVABILITY_CAPTURE_CONTENT", "false").strip().casefold() in {"1", "true", "yes", "on", "enabled"}
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.capture_content_enabled", exc)
        return False


def should_capture_content() -> bool:
    return capture_content_enabled()


def _langfuse_enabled() -> bool:
    try:
        return os.getenv("LANGFUSE_ENABLED", "false").strip().casefold() in {"1", "true", "yes", "on", "enabled"}
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability._langfuse_enabled", exc)
        return False


def langfuse_capture_content_enabled() -> bool:
    try:
        return os.getenv("LANGFUSE_CAPTURE_CONTENT", "true").strip().casefold() not in {"0", "false", "no", "off", "disabled"}
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.langfuse_capture_content_enabled", exc)
        return True


def get_langfuse_client() -> Any | None:
    global _LANGFUSE_CLIENT
    try:
        if not _langfuse_enabled():
            return None
        if _LANGFUSE_CLIENT is not None:
            return _LANGFUSE_CLIENT
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        if not public_key or not secret_key:
            return None
        if Langfuse is None:
            return None
        host = os.getenv("LANGFUSE_HOST", _DEFAULT_LANGFUSE_HOST)
        _LANGFUSE_CLIENT = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        return _LANGFUSE_CLIENT
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.get_langfuse_client", exc)
        return None


def flush_langfuse() -> None:
    try:
        client = get_client() if get_client is not None else get_langfuse_client()
        if client is not None and hasattr(client, "flush"):
            client.flush()
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.flush_langfuse", exc)


@contextmanager
def langfuse_observation_context(*, conversation_id: str | None = None, user_id: str | None = None, tags: Iterable[str] | None = None):
    callback_cls = _load_langfuse_handler_class()
    handler = callback_cls() if callback_cls is not None else _NoOpLangfuseHandler(
        conversation_id=conversation_id,
        message_id=str(conversation_id or "default"),
        user_id=user_id,
    )
    if not _langfuse_enabled() or propagate_attributes is None:
        yield handler
        return

    attributes: dict[str, Any] = {}
    if user_id is not None:
        attributes["user_id"] = str(user_id)
    if conversation_id is not None:
        attributes["session_id"] = str(conversation_id)
    if tags:
        attributes["tags"] = [str(tag) for tag in tags]

    try:
        propagation_context = propagate_attributes(**attributes)
        propagation_context.__enter__()
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.langfuse_observation_context", exc)
        yield handler
        return

    try:
        yield handler
    except BaseException as exc:
        try:
            propagation_context.__exit__(type(exc), exc, exc.__traceback__)
        except Exception as exit_exc:  # pragma: no cover - fail-safe
            _warn("observability.langfuse_observation_context.exit", exit_exc)
        raise
    else:
        try:
            propagation_context.__exit__(None, None, None)
        except Exception as exc:  # pragma: no cover - fail-safe
            _warn("observability.langfuse_observation_context.exit", exc)


def _load_langfuse_handler_class() -> Any:
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability._load_langfuse_handler_class", exc)
        return None


def get_langfuse_handler(conversation_id: str | None, message_id: str | None, user_id: str | None = None) -> Any:
    try:
        if not _langfuse_enabled():
            return _NoOpLangfuseHandler(conversation_id=conversation_id, message_id=message_id, user_id=user_id)

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        if not public_key or not secret_key:
            return _NoOpLangfuseHandler(conversation_id=conversation_id, message_id=message_id, user_id=user_id)

        callback_cls = _load_langfuse_handler_class()
        if callback_cls is None:
            return _NoOpLangfuseHandler(conversation_id=conversation_id, message_id=message_id, user_id=user_id)

        wrapped = callback_cls()
        return _GuardedLangfuseHandler(
            wrapped,
            conversation_id=str(conversation_id or "default"),
            message_id=str(message_id or "default"),
            user_id=str(user_id) if user_id else None,
        )
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.get_langfuse_handler", exc)
        return _NoOpLangfuseHandler(conversation_id=conversation_id, message_id=message_id, user_id=user_id)


def attach_langfuse_callbacks(model: Any, *, conversation_id: str | None = None, message_id: str | None = None, user_id: str | None = None) -> Any:
    """Attach a guarded Langfuse callback to a raw LLM instance without changing runtime behavior."""
    try:
        if model is None:
            return None

        def _with_active_langfuse_observation(run: Any):
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                client = get_langfuse_client()
                if client is None or not hasattr(client, "start_as_current_observation"):
                    return run(*args, **kwargs)

                model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or "unknown"
                with client.start_as_current_observation(
                    name=f"llm.{conversation_id or message_id or 'request'}",
                    as_type="span",
                    metadata={
                        "conversation_id": conversation_id,
                        "message_id": message_id,
                        "user_id": user_id,
                    },
                    model=str(model_name),
                ):
                    return run(*args, **kwargs)

            return _wrapper

        callback = get_langfuse_handler(
            conversation_id or "default",
            message_id or str(conversation_id or "default"),
            user_id=user_id,
        )
        callbacks = list(getattr(model, "callbacks", []) or [])
        exists = any(
            getattr(existing, "conversation_id", None) == getattr(callback, "conversation_id", None)
            and getattr(existing, "message_id", None) == getattr(callback, "message_id", None)
            for existing in callbacks
        )
        if callback is not None and not exists:
            callbacks.append(callback)
        if hasattr(model, "callbacks"):
            model.callbacks = callbacks
        else:
            setattr(model, "callbacks", callbacks)

        original_invoke = getattr(model, "invoke", None)
        if callable(original_invoke):
            try:
                model.invoke = _with_active_langfuse_observation(original_invoke)
            except Exception:
                original_unbound_invoke = getattr(type(model), "invoke", None)
                if callable(original_unbound_invoke):

                    def _class_invoke_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                        client = get_langfuse_client()
                        if client is None or not hasattr(client, "start_as_current_observation"):
                            return original_unbound_invoke(self, *args, **kwargs)

                        with client.start_as_current_observation(
                            name=f"llm.{conversation_id or message_id or 'request'}",
                            as_type="span",
                            metadata={
                                "conversation_id": conversation_id,
                                "message_id": message_id,
                                "user_id": user_id,
                            },
                            model=str(getattr(self, "model_name", None) or getattr(self, "model", None) or "unknown"),
                        ):
                            return original_unbound_invoke(self, *args, **kwargs)

                    setattr(type(model), "invoke", _class_invoke_wrapper)

        original_ainvoke = getattr(model, "ainvoke", None)
        if callable(original_ainvoke):
            try:
                model.ainvoke = _with_active_langfuse_observation(original_ainvoke)
            except Exception:
                original_unbound_ainvoke = getattr(type(model), "ainvoke", None)
                if callable(original_unbound_ainvoke):

                    def _class_ainvoke_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                        client = get_langfuse_client()
                        if client is None or not hasattr(client, "start_as_current_observation"):
                            return original_unbound_ainvoke(self, *args, **kwargs)

                        with client.start_as_current_observation(
                            name=f"llm.{conversation_id or message_id or 'request'}",
                            as_type="span",
                            metadata={
                                "conversation_id": conversation_id,
                                "message_id": message_id,
                                "user_id": user_id,
                            },
                            model=str(getattr(self, "model_name", None) or getattr(self, "model", None) or "unknown"),
                        ):
                            return original_unbound_ainvoke(self, *args, **kwargs)

                    setattr(type(model), "ainvoke", _class_ainvoke_wrapper)

        return model
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.attach_langfuse_callbacks", exc)
        return model


def _ensure_otel_configuration() -> bool:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False

    global _OTEL_PROVIDER
    if _OTEL_PROVIDER is not None:
        return True

    try:
        resource = Resource.create({"service.name": "rag-application"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint) if OTLPSpanExporter is not None else None
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _OTEL_PROVIDER = provider
        return True
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability._ensure_otel_configuration", exc)
        return False


def get_tracer(name: str) -> Any:
    try:
        if not _ensure_otel_configuration():
            return trace.NoOpTracer()
        return trace.get_tracer(name)
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.get_tracer", exc)
        return trace.NoOpTracer()


def _ensure_metrics_configuration() -> bool:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False

    global _METER_PROVIDER
    if _METER_PROVIDER is not None:
        return True

    try:
        if OTLPMetricExporter is None:
            return False
        resource = Resource.create({"service.name": "rag-application"})
        exporter = OTLPMetricExporter(endpoint=endpoint)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        _METER_PROVIDER = provider
        return True
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability._ensure_metrics_configuration", exc)
        return False


def counter(name: str, **labels: str) -> Any:
    key = (name, tuple(sorted(labels.items())))
    try:
        with _LOCK:
            metric = _METRIC_REGISTRY.get(key)
            if metric is None:
                metric = _SafeMetric(name=name, kind="counter", labels=labels)
                _METRIC_REGISTRY[key] = metric
        if not _ensure_metrics_configuration():
            return metric
        meter = metrics.get_meter("rag.observability")
        meter.create_counter(name)
        return metric
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.counter", exc)
        return _METRIC_REGISTRY.get(key) or _SafeMetric(name=name, kind="counter", labels=labels)


def histogram(name: str) -> Any:
    key = (name, tuple())
    try:
        with _LOCK:
            metric = _METRIC_REGISTRY.get(key)
            if metric is None:
                metric = _SafeMetric(name=name, kind="histogram")
                _METRIC_REGISTRY[key] = metric
        if not _ensure_metrics_configuration():
            return metric
        meter = metrics.get_meter("rag.observability")
        meter.create_histogram(name)
        return metric
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.histogram", exc)
        return _METRIC_REGISTRY.get(key) or _SafeMetric(name=name, kind="histogram")


def gauge(name: str, **labels: str) -> Any:
    key = (name, tuple(sorted(labels.items())))
    try:
        with _LOCK:
            metric = _METRIC_REGISTRY.get(key)
            if metric is None:
                metric = _SafeMetric(name=name, kind="gauge", labels=labels)
                _METRIC_REGISTRY[key] = metric
        if not _ensure_metrics_configuration():
            return metric
        return metric
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.gauge", exc)
        return _METRIC_REGISTRY.get(key) or _SafeMetric(name=name, kind="gauge", labels=labels)


def record_langfuse_score(
    name: str,
    value: float | str,
    *,
    session_id: str | None = None,
    trace_id: str | None = None,
    observation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    comment: str | None = None,
) -> None:
    """Forward a numeric/categorical score to Langfuse without interrupting app behavior."""
    try:
        if not _langfuse_enabled():
            return
        client = get_client() if get_client is not None else get_langfuse_client()
        if client is None or not hasattr(client, "create_score"):
            return
        payload = {
            "name": str(name),
            "value": value,
            "session_id": session_id,
            "trace_id": trace_id,
            "observation_id": observation_id,
            "metadata": metadata or {},
            "comment": comment,
        }
        if payload["session_id"] is None:
            payload.pop("session_id")
        if payload["trace_id"] is None:
            payload.pop("trace_id")
        if payload["observation_id"] is None:
            payload.pop("observation_id")
        if payload["comment"] is None:
            payload.pop("comment")
        client.create_score(**{k: v for k, v in payload.items() if v is not None})
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.record_langfuse_score", exc)


def record_guardrail_triggered(guardrail: str, outcome: str, **labels: str) -> None:
    try:
        metric = counter("guardrail_triggered_total", guardrail=guardrail, outcome=outcome, **labels)
        metric.inc(1)
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.record_guardrail_triggered", exc)


def record_audio_error(source: str, error_type: str | None = None, **labels: str) -> None:
    try:
        metric = counter("audio_error_total", source=source, error_type=error_type or "unknown", **labels)
        metric.inc(1)
    except Exception as exc:  # pragma: no cover - fail-safe
        _warn("observability.record_audio_error", exc)


__all__ = [
    "attach_langfuse_callbacks",
    "bind_context",
    "capture_content_enabled",
    "counter",
    "gauge",
    "get_langfuse_handler",
    "get_logger",
    "get_tracer",
    "histogram",
    "record_audio_error",
    "record_guardrail_triggered",
    "record_langfuse_score",
    "should_capture_content",
]

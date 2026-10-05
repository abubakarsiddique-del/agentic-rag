from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from guardrails.harmful_content import inspect_harmful_output
from config import NO_INFO_PHRASE
from guardrails.output_schema import (
    validate_citations_against_sources,
    validate_rewritten_query,
    validate_trace_record,
)
from guardrails.sensitive_content import redact_persisted_value, redact_sensitive_text
from agentic_rag import RAGService


def test_rewritten_query_validation_rejects_empty_and_oversized_values():
    assert validate_rewritten_query("  policy approval process  ") == "policy approval process"
    with pytest.raises(ValidationError):
        validate_rewritten_query("  ")
    with pytest.raises(ValidationError):
        validate_rewritten_query("q" * 1001)


def test_trace_validation_preserves_additive_guardrail_fields():
    trace = validate_trace_record({
        "step": "retrieve",
        "detail": "found one passage",
        "guardrail": {"check": "passage_injection", "outcome": "passed"},
    })
    assert trace["guardrail"]["outcome"] == "passed"
    with pytest.raises(ValidationError):
        validate_trace_record({"detail": "missing step"})


def test_citation_schema_accepts_known_sources_and_rejects_unknown_ids():
    sources = [{"filename": "policy.pdf", "page": 3, "passage_id": "doc-1:3"}]
    valid = validate_citations_against_sources(
        "Policy requires review [policy.pdf, page 3] [doc:doc-1:3]",
        sources,
    )
    assert valid["valid"] is True
    invalid = validate_citations_against_sources("Claim [fake.pdf, page 9]", sources)
    assert invalid["valid"] is False
    assert invalid["invalid"][0]["source"] == "fake.pdf"


def test_sensitive_redaction_masks_common_identifiers_and_preserves_other_text():
    value = "Contact alex@example.com or 212-555-0100; SSN 123-45-6789."
    redacted = redact_sensitive_text(value)
    assert "alex@example.com" not in redacted
    assert "212-555-0100" not in redacted
    assert "123-45-6789" not in redacted
    assert "Contact" in redacted
    nested = redact_persisted_value({"answer": ["alex@example.com"]})
    assert "alex@example.com" not in nested["answer"][0]


def test_harmful_output_detector_is_high_confidence_and_toggleable():
    assert inspect_harmful_output("Deploy ransomware to encrypt the customer database.")["outcome"] == "blocked"
    assert inspect_harmful_output("This guide explains ransomware detection and recovery.")["outcome"] == "passed"
    assert inspect_harmful_output("steal user passwords", enabled=False)["outcome"] == "disabled"


def test_direct_graph_answer_blocks_harmful_output(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_HARMFUL_CONTENT_ENABLED", "1")
    service = RAGService.__new__(RAGService)
    service.direct_chain = SimpleNamespace(
        invoke=lambda _payload: "Deploy ransomware to encrypt the customer database."
    )

    result = service._direct_answer({"question": "Hello", "trace": []})

    assert result["answer"] == NO_INFO_PHRASE
    assert result["trace"][-1]["guardrail"]["outcome"] == "blocked"
    assert result["trace"][-1]["guardrail"]["check"] == "harmful_content"


def test_direct_stream_buffers_then_blocks_harmful_output(monkeypatch):
    monkeypatch.setenv("RAG_GUARDRAIL_HARMFUL_CONTENT_ENABLED", "1")
    service = RAGService.__new__(RAGService)
    service._scope_guard = lambda _state: {
        "guardrail_blocked": False,
        "scope_guardrail": {"outcome": "passed"},
    }
    service._contextualize = lambda state: {"trace": state["trace"]}
    service._route_question = lambda state: {"route": "direct", "trace": state["trace"]}
    service.route_chain = object()
    service.max_retries = 2
    service.direct_chain = SimpleNamespace(
        stream=lambda _payload: iter(("Deploy ransomware ", "to encrypt the customer database."))
    )

    stream, sources = service.ask_stream("Hello")

    assert list(stream) == [NO_INFO_PHRASE]
    assert sources == []
    assert service.last_trace[-2]["guardrail"]["outcome"] == "blocked"
    assert service.last_trace[-1]["step"] == "abstain"


def test_rewrite_chain_retries_malformed_output_then_accepts_valid_query():
    service = RAGService.__new__(RAGService)

    class RewriteChain:
        responses = iter(("   ", "targeted approval process query"))
        calls = 0

        def invoke(self, _payload):
            self.calls += 1
            return next(self.responses)

    service.rewrite_chain = RewriteChain()
    result = service._rewrite_query({
        "question": "Who approves this?",
        "search_query": "Who approves this?",
        "memory_hints": "",
        "trace": [],
    })
    assert result["search_query"] == "targeted approval process query"
    assert service.rewrite_chain.calls == 2
    assert result["trace"][-1]["guardrail"]["outcome"] == "passed_after_retry"


def test_rewrite_chain_falls_back_after_two_malformed_outputs():
    service = RAGService.__new__(RAGService)

    class RewriteChain:
        def __init__(self):
            self.calls = 0

        def invoke(self, _payload):
            self.calls += 1
            return " "

    service.rewrite_chain = RewriteChain()
    result = service._rewrite_query({
        "question": "Who approves this?",
        "search_query": "Who approves this?",
        "memory_hints": "",
        "trace": [],
    })
    assert service.rewrite_chain.calls == 2
    assert result["search_query"] == "Who approves this?"
    assert result["trace"][-1]["guardrail"]["outcome"] == "fallback"
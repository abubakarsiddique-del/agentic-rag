"""Live groundedness enforcement using the shared eval answer scorer."""

from __future__ import annotations

from typing import Any, Callable

from config import NO_INFO_PHRASE
from eval.answer_eval import create_groq_judge, score_answer
from guardrails.config import groundedness_threshold, guardrail_enabled
from guardrails.harmful_content import inspect_harmful_output
from guardrails.output_schema import validate_citations_against_sources
from observability import record_guardrail_triggered


def sources_from_documents(documents: list[Any]) -> list[dict[str, Any]]:
    sources = []
    for document in documents:
        metadata = dict(getattr(document, "metadata", {}) or {})
        source = metadata.get("source") or metadata.get("document_name") or ""
        item = {
            "filename": str(source),
            "page": metadata.get("page", "?"),
            "snippet": str(getattr(document, "page_content", "") or ""),
        }
        if metadata.get("passage_id"):
            item["passage_id"] = str(metadata["passage_id"])
        sources.append(item)
    return sources


def assess_groundedness(answer: str, scoring: dict[str, Any]) -> dict[str, Any]:
    claims = scoring.get("claims", [])
    if not claims:
        passed = answer.strip() == NO_INFO_PHRASE
        return {
            "enabled": True,
            "outcome": "passed" if passed else "failed",
            "faithfulness": None,
            "citation_precision": None,
            "citation_recall": None,
            "reason": "no_claims_detected" if not passed else None,
        }

    metrics = {
        name: scoring.get(name)
        for name in ("faithfulness", "citation_precision", "citation_recall")
    }
    failures = []
    for name, value in metrics.items():
        threshold = groundedness_threshold(name, 1.0)
        if value is None or value < threshold:
            failures.append(name)
    return {
        "enabled": True,
        "outcome": "failed" if failures else "passed",
        **metrics,
        "reason": ",".join(failures) if failures else None,
    }


def score_generated_answer(
    question: str,
    answer: str,
    documents: list[Any],
    *,
    judge: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Use eval.answer_eval's claim/citation scorer with retrieved passages only."""
    return score_answer(
        question,
        answer,
        sources_from_documents(documents),
        judge=judge or create_groq_judge(),
    )


def validate_generated_answer(
    question: str,
    answer: str,
    documents: list[Any],
    memory_hints: str,
    *,
    sufficient: bool | None = True,
    judge: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    regenerate: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Score once, optionally regenerate once, then return pass/block metadata."""
    groundedness_on = guardrail_enabled("groundedness")
    leakage_on = guardrail_enabled("memory_leakage")
    abstention_on = guardrail_enabled("abstention")
    needs_scoring = groundedness_on or leakage_on
    schema_on = guardrail_enabled("output_schema")
    harmful = inspect_harmful_output(
        answer,
        enabled=guardrail_enabled("harmful_content"),
    )
    citation_schema = (
        validate_citations_against_sources(answer, sources_from_documents(documents))
        if schema_on
        else {"valid": True, "citation_count": 0, "invalid": []}
    )
    schema_result = {
        "enabled": schema_on,
        "outcome": "failed" if not citation_schema["valid"] else ("passed" if schema_on else "disabled"),
        "invalid_citations": citation_schema["invalid"],
    }
    if not needs_scoring:
        passed = schema_result["outcome"] != "failed" and harmful["outcome"] != "blocked"
        return {
            "answer": answer if passed else NO_INFO_PHRASE,
            "passed": passed,
            "groundedness": {"enabled": False, "outcome": "disabled"},
            "memory_leakage": {"enabled": False, "outcome": "disabled"},
            "output_schema": schema_result,
            "harmful_content": harmful,
            "attempts": 0,
        }

    current_answer = answer
    try:
        judge_fn = judge or create_groq_judge()
        scoring = score_generated_answer(question, current_answer, documents, judge=judge_fn)
    except Exception as exc:
        if groundedness_on:
            record_guardrail_triggered("groundedness", "judge_error")
        return {
            "answer": NO_INFO_PHRASE,
            "passed": False,
            "groundedness": {
                "enabled": groundedness_on,
                "outcome": "failed" if groundedness_on else "disabled",
                "reason": "judge_error" if groundedness_on else None,
            },
            "memory_leakage": {
                "enabled": leakage_on,
                "outcome": "failed" if leakage_on else "disabled",
                "violations": ["judge_error"] if leakage_on else [],
            },
            "forced_abstain": False,
            "attempts": 1,
            "error_type": type(exc).__name__,
        }
    groundedness = (
        assess_groundedness(current_answer, scoring)
        if groundedness_on
        else {"enabled": False, "outcome": "disabled"}
    )
    if schema_result["outcome"] == "failed" and groundedness.get("outcome") != "failed":
        groundedness = {
            **groundedness,
            "outcome": "failed",
            "reason": "invalid_citations",
        }
        record_guardrail_triggered("groundedness", "invalid_citations")
    from guardrails.memory_leakage import check_memory_leakage

    leakage = (
        check_memory_leakage(scoring, documents, memory_hints)
        if leakage_on
        else {"enabled": False, "outcome": "disabled", "violations": []}
    )
    forced_abstain = (
        abstention_on
        and sufficient is False
        and groundedness.get("outcome") == "failed"
    )
    attempts = 1

    schema_retry_required = schema_result.get("outcome") == "failed"
    groundedness_retry_allowed = False
    if groundedness.get("outcome") == "failed":
        from guardrails.config import groundedness_action

        groundedness_retry_allowed = groundedness_action() == "regenerate_once"

    should_retry = schema_retry_required or groundedness_retry_allowed
    if (
        not forced_abstain
        and harmful.get("outcome") != "blocked"
        and leakage.get("outcome") != "failed"
        and groundedness.get("outcome") == "failed"
        and should_retry
        and regenerate is not None
    ):
        feedback = schema_result.get("invalid_citations") or groundedness.get("reason") or "Use supported claims and valid retrieved citations."
        current_answer = regenerate(str(feedback))
        attempts += 1
        try:
            scoring = score_generated_answer(question, current_answer, documents, judge=judge_fn)
        except Exception as exc:
            if groundedness_on:
                record_guardrail_triggered("groundedness", "judge_error")
            return {
                "answer": NO_INFO_PHRASE,
                "passed": False,
                "groundedness": {
                    "enabled": groundedness_on,
                    "outcome": "failed" if groundedness_on else "disabled",
                    "reason": "judge_error" if groundedness_on else None,
                },
                "memory_leakage": {
                    "enabled": leakage_on,
                    "outcome": "failed" if leakage_on else "disabled",
                    "violations": ["judge_error"] if leakage_on else [],
                },
                "forced_abstain": False,
                "attempts": attempts,
                "error_type": type(exc).__name__,
            }
        groundedness = assess_groundedness(current_answer, scoring) if groundedness_on else groundedness
        if schema_on:
            citation_schema = validate_citations_against_sources(
                current_answer,
                sources_from_documents(documents),
            )
            schema_result = {
                "enabled": True,
                "outcome": "passed" if citation_schema["valid"] else "failed",
                "invalid_citations": citation_schema["invalid"],
            }
            if schema_result["outcome"] == "failed" and groundedness.get("outcome") != "failed":
                groundedness = {**groundedness, "outcome": "failed", "reason": "invalid_citations"}
        harmful = inspect_harmful_output(
            current_answer,
            enabled=guardrail_enabled("harmful_content"),
        )
        leakage = (
            check_memory_leakage(scoring, documents, memory_hints)
            if leakage_on
            else leakage
        )
        forced_abstain = (
            abstention_on
            and sufficient is False
            and groundedness.get("outcome") == "failed"
        )

    passed = (
        groundedness.get("outcome") != "failed"
        and leakage.get("outcome") != "failed"
        and schema_result.get("outcome") != "failed"
        and harmful.get("outcome") != "blocked"
        and not forced_abstain
    )
    return {
        "answer": current_answer if passed else NO_INFO_PHRASE,
        "passed": passed,
        "groundedness": groundedness,
        "memory_leakage": leakage,
        "output_schema": schema_result,
        "harmful_content": harmful,
        "forced_abstain": forced_abstain,
        "attempts": attempts,
    }
"""Hard-check judge-reported evidence/citations against retrieved passages."""

from __future__ import annotations

from typing import Any


def _source_id(document: Any, index: int) -> str:
    metadata = dict(getattr(document, "metadata", {}) or {})
    if metadata.get("passage_id"):
        return str(metadata["passage_id"])
    filename = str(metadata.get("source") or metadata.get("document_name") or "")
    if filename:
        return f"{filename} page {metadata.get('page', '?')}"
    return f"source-{index}"


def _hinted_filenames(memory_hints: str) -> set[str]:
    names: set[str] = set()
    for line in (memory_hints or "").splitlines():
        if "prior files:" in line.casefold():
            value = line.split(":", 1)[1]
            names.update(name.strip().casefold() for name in value.split(",") if name.strip())
    return names


def check_memory_leakage(
    scoring: dict[str, Any],
    documents: list[Any],
    memory_hints: str,
) -> dict[str, Any]:
    allowed_ids = {_source_id(document, index) for index, document in enumerate(documents, start=1)}
    hint_names = _hinted_filenames(memory_hints)
    violations = []
    for claim_index, claim in enumerate(scoring.get("claims", [])):
        citation_ids = claim.get("citation_ids", [])
        evidence_ids = claim.get("evidence_ids", [])
        if claim.get("has_citation") and not citation_ids:
            violations.append({"claim_index": claim_index, "reason": "citation_has_no_source_id"})
        for kind, values in (("citation", citation_ids), ("evidence", evidence_ids)):
            for value in values:
                source_id = str(value)
                if source_id in allowed_ids:
                    continue
                is_memory_source = any(name in source_id.casefold() for name in hint_names)
                violations.append({
                    "claim_index": claim_index,
                    "reason": "memory_hint_used_as_evidence" if is_memory_source else f"{kind}_not_retrieved",
                })
    return {
        "enabled": True,
        "outcome": "failed" if violations else "passed",
        "violations": violations,
    }
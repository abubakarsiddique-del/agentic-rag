"""Pydantic validation for query, citation, and trace outputs."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class RewrittenQuery(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class CitationReference(BaseModel):
    source: str = Field(min_length=1, max_length=300)
    page: str | None = Field(default=None, max_length=40)


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: str = Field(min_length=1, max_length=80)
    detail: str | None = None


_CITATION_PATTERNS = (
    re.compile(r"^Source\s+\d+\s*:\s*Page\s+[^\]]+$", re.IGNORECASE),
    re.compile(r"^(?P<source>[^\[\]]+?)[, ]+page\s+(?P<page>[\w.-]+)$", re.IGNORECASE),
    re.compile(r"^(?:doc|passage|source)\s*[:#-]\s*[\w:.-]+$", re.IGNORECASE),
)


def validate_rewritten_query(value: Any) -> str:
    return RewrittenQuery(query=str(value or "").strip()).query


def validate_trace_record(value: dict[str, Any]) -> dict[str, Any]:
    return TraceRecord.model_validate(value).model_dump(exclude_unset=True)


def extract_citation_references(answer: str) -> list[CitationReference]:
    citations = []
    for match in re.findall(r"\[([^\]]+)\]", answer or ""):
        value = match.strip()
        if _CITATION_PATTERNS[0].match(value) or _CITATION_PATTERNS[2].match(value):
            citations.append(CitationReference(source=value))
            continue
        parsed = _CITATION_PATTERNS[1].match(value)
        if parsed:
            citations.append(CitationReference(source=parsed.group("source").strip(), page=parsed.group("page")))
    return citations


def validate_citations_against_sources(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    references = extract_citation_references(answer)
    known_sources = {
        str(source.get("passage_id") or source.get("filename") or source.get("source") or "").casefold()
        for source in sources
    }
    known_pages = {
        (str(source.get("filename") or source.get("source") or "").casefold(), str(source.get("page", "?")))
        for source in sources
    }
    invalid = []
    for reference in references:
        source_text = reference.source.casefold()
        normalized_source = re.sub(r"^(?:doc|passage|source)\s*[:#-]\s*", "", source_text)
        if source_text.startswith("source "):
            match = re.match(r"source\s+(\d+)", source_text)
            if match and int(match.group(1)) <= len(sources):
                continue
        if source_text in known_sources or normalized_source in known_sources:
            continue
        if reference.page is not None and (source_text, reference.page) in known_pages:
            continue
        invalid.append(reference.model_dump(exclude_none=True))
    return {
        "valid": not invalid,
        "citation_count": len(references),
        "invalid": invalid,
    }


__all__ = ["CitationReference", "RewrittenQuery", "TraceRecord", "ValidationError", "validate_citations_against_sources", "validate_rewritten_query", "validate_trace_record"]
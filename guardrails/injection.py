"""Pattern-based detection of high-confidence prompt injection in passages."""

from __future__ import annotations

import re
from typing import Any


_INJECTION_PATTERNS = {
    "instruction_override": re.compile(
        r"\b(?:ignore|disregard|override|forget)\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|system|developer)\s+instructions\b",
        re.IGNORECASE,
    ),
    "system_prompt_exfiltration": re.compile(
        r"\b(?:reveal|print|repeat|show|disclose|expose)\b.{0,50}\b(?:system|developer)\s+prompt\b|"
        r"\b(?:system|developer)\s+prompt\b.{0,50}\b(?:reveal|print|repeat|show|disclose|expose)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "unrestricted_persona": re.compile(
        r"\b(?:act|respond|behave|roleplay|role-play)\b.{0,50}\b(?:unrestricted|unfiltered|jailbroken|without\s+(?:any\s+)?restrictions|as\s+DAN)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "embedded_role_instruction": re.compile(
        r"(?:<\|(?:system|assistant|developer)\|>|^\s*#{1,3}\s*(?:system|developer)\b|\[\s*(?:system|developer)\s*\])",
        re.IGNORECASE | re.MULTILINE,
    ),
    "context_boundary_spoof": re.compile(
        r"</?untrusted_reference_material\b",
        re.IGNORECASE,
    ),
    "citation_fabrication": re.compile(
        r"\b(?:invent|fabricate|make\s+up|fake)\b.{0,40}\b(?:citations?|sources?)\b|"
        r"\b(?:cite|attribute)\b.{0,40}\b(?:fake|fabricated|made\s+up)\s+(?:source|citation)\b",
        re.IGNORECASE | re.DOTALL,
    ),
}


def passage_identifier(document: Any, index: int = 0) -> str:
    metadata = dict(getattr(document, "metadata", {}) or {})
    for key in ("passage_id", "id", "document_id"):
        if metadata.get(key):
            value = str(metadata[key])
            if key == "document_id" and metadata.get("page") is not None:
                value += f":{metadata['page']}"
            return value
    source = metadata.get("source") or metadata.get("document_name") or "unknown"
    page = metadata.get("page", "?")
    return f"{source}:{page}:{index}"


def scan_passage(text: str) -> list[str]:
    """Return matching pattern categories; never return/log the document text."""
    value = text or ""
    return [name for name, pattern in _INJECTION_PATTERNS.items() if pattern.search(value)]


def filter_passages(documents: list[Any], *, enabled: bool = True) -> tuple[list[Any], dict]:
    if not enabled:
        return list(documents), {
            "enabled": False,
            "outcome": "disabled",
            "checked_passages": 0,
            "excluded": [],
        }

    kept = []
    excluded = []
    for index, document in enumerate(documents):
        text = str(getattr(document, "page_content", "") or "")
        categories = scan_passage(text)
        if categories:
            excluded.append({
                "passage_id": passage_identifier(document, index),
                "categories": categories,
            })
        else:
            kept.append(document)
    return kept, {
        "enabled": True,
        "outcome": "flagged" if excluded else "passed",
        "checked_passages": len(documents),
        "excluded": excluded,
    }
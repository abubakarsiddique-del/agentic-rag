"""High-confidence off-scope and jailbreak checks for user questions."""

from __future__ import annotations

import re


SCOPE_BLOCK_MESSAGE = "I can only answer questions grounded in your uploaded documents."

_SCOPE_PATTERNS = {
    "ignore_document_grounding": re.compile(
        r"\b(?:ignore|disregard|bypass|forget)\b.{0,50}\b(?:uploaded documents?|files?|sources?|document rules?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "system_prompt_disclosure": re.compile(
        r"\b(?:reveal|show|print|repeat|disclose|tell me)\b.{0,45}\b(?:system|developer)\s+prompt\b|"
        r"\b(?:system|developer)\s+prompt\b.{0,45}\b(?:reveal|show|print|repeat|disclose)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "unrestricted_persona": re.compile(
        r"\b(?:act|respond|behave|roleplay|role-play)\b.{0,50}\b(?:unrestricted|unfiltered|jailbroken|without\s+(?:any\s+)?restrictions|as\s+DAN)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "off_topic_creative_request": re.compile(
        r"\b(?:tell|write|make|generate)\s+me\s+(?:a|an)\s+(?:joke|poem|song|short\s+story)\b|"
        r"\b(?:let's|let\s+us)\s+play\s+(?:a\s+)?game\b",
        re.IGNORECASE,
    ),
    "instruction_override": re.compile(
        r"\b(?:ignore|disregard|override|forget)\s+(?:all\s+)?(?:the\s+)?"
        r"(?:(?:previous|prior|above|system|developer)\s+|(?:your|my|these|those)\s+)?instructions\b",
        re.IGNORECASE,
    ),
}


def check_scope(question: str, *, enabled: bool = True) -> dict:
    if not enabled:
        return {"enabled": False, "outcome": "disabled", "categories": []}
    text = question or ""
    categories = [name for name, pattern in _SCOPE_PATTERNS.items() if pattern.search(text)]
    return {
        "enabled": True,
        "outcome": "blocked" if categories else "passed",
        "categories": categories,
    }
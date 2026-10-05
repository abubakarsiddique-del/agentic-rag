"""Minimal high-confidence harmful-output checks for business RAG answers."""

from __future__ import annotations

import re


_PATTERNS = {
    "credential_theft": re.compile(
        r"\b(?:steal|harvest|exfiltrate)\b.{0,60}\b(?:passwords?|credentials?|api\s+keys?|tokens?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "malware_deployment": re.compile(
        r"\b(?:deploy|write|create|execute)\b.{0,60}\b(?:ransomware|credential\s+stealer|keylogger|destructive\s+malware)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "targeted_phishing": re.compile(
        r"\b(?:send|write|craft)\b.{0,60}\b(?:phishing|credential\s+theft)\b.{0,60}\b(?:password|login|credentials?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
}


def inspect_harmful_output(text: str, *, enabled: bool = True) -> dict:
    if not enabled:
        return {"enabled": False, "outcome": "disabled", "categories": []}
    categories = [name for name, pattern in _PATTERNS.items() if pattern.search(text or "")]
    return {
        "enabled": True,
        "outcome": "blocked" if categories else "passed",
        "categories": categories,
    }
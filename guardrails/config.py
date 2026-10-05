"""Environment-backed guardrail toggles; checks are enabled by default."""

from __future__ import annotations

import os


def guardrail_enabled(name: str) -> bool:
    value = os.getenv(f"RAG_GUARDRAIL_{name.upper()}_ENABLED", "1").strip().casefold()
    return value not in {"0", "false", "no", "off", "disabled"}


def groundedness_action() -> str:
    value = os.getenv("RAG_GUARDRAIL_GROUNDEDNESS_ACTION", "abstain").strip().casefold()
    return value if value in {"abstain", "regenerate_once"} else "abstain"


def groundedness_threshold(name: str, default: float = 1.0) -> float:
    try:
        return min(1.0, max(0.0, float(os.getenv(f"RAG_GUARDRAIL_MIN_{name.upper()}", str(default)))))
    except (TypeError, ValueError):
        return default


def sensitive_content_enabled() -> bool:
    value = os.getenv("RAG_GUARDRAIL_SENSITIVE_CONTENT_ENABLED", "0").strip().casefold()
    return value in {"1", "true", "yes", "on", "enabled"}
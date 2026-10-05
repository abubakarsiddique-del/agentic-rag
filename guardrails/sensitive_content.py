"""Opt-in lightweight masking for text copied into persistence/logging."""

from __future__ import annotations

import re


_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,2}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}(?!\w)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _luhn_valid(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def redact_sensitive_text(value: str) -> str:
    text = value or ""
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _SSN.sub("[REDACTED_SSN]", text)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    return _CARD.sub(
        lambda match: "[REDACTED_PAYMENT_CARD]" if _luhn_valid(match.group(0)) else match.group(0),
        text,
    )


def redact_persisted_value(value):
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, list):
        return [redact_persisted_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_persisted_value(item) for item in value)
    if isinstance(value, dict):
        return {key: redact_persisted_value(item) for key, item in value.items()}
    return value
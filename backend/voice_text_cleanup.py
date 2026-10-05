from __future__ import annotations

import re

from guardrails.output_schema import extract_citation_references

_BRACKETED_TEXT = re.compile(r"\[[^\]]+\]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?P<suffix>(?:\s*\[[^\]]+\])*)\s+")


def strip_citations(text: str) -> str:
    def keep_non_citation(match: re.Match[str]) -> str:
        return "" if extract_citation_references(match.group(0)) else match.group(0)

    return _BRACKETED_TEXT.sub(keep_non_citation, text or "")


def clean_text_for_speech(text: str) -> str:
    text = strip_citations(text)
    text = re.sub(r"(?s)```[^\n]*\n?(.*?)```", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*_]\s*){3,}$", "", text)
    text = re.sub(r"\*\*|__|~~|(?<!\*)\*(?!\*)|`+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_speech_text(text: str, maximum_characters: int = 200) -> list[str]:
    remaining = clean_text_for_speech(text)
    segments = []
    while len(remaining) > maximum_characters:
        boundary = remaining.rfind(" ", 0, maximum_characters + 1)
        if boundary < 1:
            boundary = maximum_characters
        segments.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        segments.append(remaining)
    return segments


def split_completed_sentences(text: str) -> tuple[list[str], str]:
    sentences = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text or ""):
        markers = _BRACKETED_TEXT.findall(match.group("suffix"))
        if any(not extract_citation_references(marker) for marker in markers):
            continue
        sentence = (text or "")[start:match.start()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    return sentences, (text or "")[start:]
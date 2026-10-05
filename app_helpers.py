"""User-facing helpers for the Streamlit document Q&A app."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
SNIPPET_MAX_LEN = 400
NO_INFO_PHRASE = "I couldn't find that information in the uploaded documents."


@dataclass(frozen=True)
class SourcePreview:
    """A single citation shown under an assistant message."""

    filename: str
    page: int | str
    snippet: str
    document_name: str | None = None
    document_path: str | None = None
    score: float | None = None


def normalize_document_name(raw: str) -> str:
    """Normalize raw document references to a stable display name."""
    value = (raw or "").strip()
    if not value:
        return "Unknown file"
    return os.path.basename(value) or value


def serialize_sources(sources: Sequence[SourcePreview] | None) -> list[dict[str, Any]]:
    """Convert source previews to JSON-safe dicts for persistence or export."""
    if not sources:
        return []

    serialized: list[dict[str, Any]] = []
    for source in sources:
        record = {
            "filename": source.filename,
            "page": source.page,
            "snippet": source.snippet,
            "document_name": source.document_name or normalize_document_name(source.filename),
            "document_path": source.document_path,
            "score": source.score,
        }
        serialized.append(record)
    return serialized


def group_sources_by_document(sources: Sequence[SourcePreview]) -> dict[str, list[SourcePreview]]:
    """Group source previews by their normalized document name."""
    grouped: dict[str, list[SourcePreview]] = {}
    for source in sources:
        key = (source.document_name or source.filename or "Unknown file").strip() or "Unknown file"
        normalized_key = normalize_document_name(key)
        grouped.setdefault(normalized_key, []).append(source)
    return dict(sorted(grouped.items()))


def convert_source_dicts(sources: Sequence[Any] | None) -> list[SourcePreview]:
    """Coerce serialized source payloads into SourcePreview objects."""
    if not sources:
        return []

    converted: list[SourcePreview] = []
    for item in sources:
        if isinstance(item, SourcePreview):
            converted.append(item)
        elif isinstance(item, dict):
            converted.append(
                SourcePreview(
                    filename=str(item.get("filename") or item.get("document_name") or "Unknown file"),
                    page=item.get("page", "?"),
                    snippet=str(item.get("snippet") or ""),
                    document_name=item.get("document_name"),
                    document_path=item.get("document_path"),
                    score=item.get("score"),
                )
            )
    return converted


@dataclass
class ChatMessage:
    """One turn in the chat UI."""

    role: str  # "user" | "assistant"
    content: str
    sources: list[SourcePreview] | None = None
    is_error: bool = False
    no_sources: bool = False
    reasoning: str | None = None
    trace: list[str] | list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class RejectedFile:
    name: str
    reason: str


def groq_api_key_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def partition_uploaded_files(
    files: Sequence[Any],
) -> tuple[list[Any], list[RejectedFile]]:
    """Split uploads into files we can process vs rejected (wrong type)."""
    valid: list[Any] = []
    rejected: list[RejectedFile] = []
    for uploaded in files:
        name = getattr(uploaded, "name", "") or "Unknown file"
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            rejected.append(
                RejectedFile(
                    name=name,
                    reason="Only PDF (.pdf) and plain text (.txt) files are supported.",
                )
            )
        else:
            valid.append(uploaded)
    return valid, rejected


def validate_uploaded_files(files: Sequence[Any]) -> str | None:
    """
    Return a user-facing error message, or None if all files look acceptable.
    """
    if not files:
        return "Choose at least one PDF or text file to continue."

    _, rejected = partition_uploaded_files(files)
    if rejected:
        names = ", ".join(f'"{r.name}"' for r in rejected[:3])
        extra = f" (and {len(rejected) - 3} more)" if len(rejected) > 3 else ""
        return (
            f"These files cannot be used: {names}{extra}. "
            "Remove them or pick PDF/text files only."
        )
    return None


def format_index_success(stats: dict[str, Any], file_names: Iterable[str]) -> str:
    """Turn build_index() stats into plain English."""
    doc_pages = stats.get("documents", 0)
    sections = stats.get("chunks", 0)
    names = list(file_names)
    if len(names) == 1:
        file_part = f'"{names[0]}"'
    elif len(names) == 2:
        file_part = f'"{names[0]}" and "{names[1]}"'
    else:
        file_part = f"{len(names)} files"

    return (
        f"Ready to go. We read {file_part} "
        f"({doc_pages} page{'s' if doc_pages != 1 else ''} with text, "
        f"split into {sections} searchable section{'s' if sections != 1 else ''}). "
        "Ask a question whenever you like."
    )


def documents_to_sources(docs: Sequence[Document]) -> list[SourcePreview]:
    """Build source previews from retrieved documents."""
    previews: list[SourcePreview] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        filename = str(metadata.get("source", metadata.get("document_name", "Unknown file")))
        document_name = str(metadata.get("document_name", filename))
        document_path = str(metadata.get("document_path", filename))
        page = metadata.get("page", "?")
        text = (doc.page_content or "").strip()
        if len(text) > SNIPPET_MAX_LEN:
            text = text[: SNIPPET_MAX_LEN - 1].rstrip() + "…"
        previews.append(
            SourcePreview(
                filename=normalize_document_name(filename),
                page=page,
                snippet=text,
                document_name=normalize_document_name(document_name),
                document_path=document_path,
                score=metadata.get("score"),
            )
        )
    return previews


def sources_are_empty(sources: list[SourcePreview]) -> bool:
    return not sources or all(not (s.snippet or "").strip() for s in sources)


def answer_indicates_no_information(answer: str) -> bool:
    return NO_INFO_PHRASE.lower() in (answer or "").lower()


def summarize_answer_effort(trace: Sequence[dict[str, Any]] | None) -> str:
    """Return a one-line effort label for a single answer from the backend trace."""
    if not trace:
        return "Couldn't confirm"

    steps = [str(item.get("step", "")).lower() for item in trace if isinstance(item, dict)]
    if "direct_answer" in steps:
        return "Answered directly"

    retrieve_count = steps.count("retrieve")
    rewrite_count = steps.count("rewrite")
    if not retrieve_count:
        return "Couldn't confirm"

    retries = max(0, retrieve_count - 1)
    if retries == 0:
        return "Answered directly"
    if retries == 1:
        return "Answered after 1 retry"
    if retries == 2:
        return "Answered after 2 retries"
    return f"Answered after {min(retries, 2)} retries" if retries > 0 else "Answered directly"


def format_conversation_export(messages: Sequence[ChatMessage]) -> str:
    """Plain-text export of the visible chat."""
    lines: list[str] = ["Agentic RAG — chat export", ""]
    for msg in messages:
        if msg.role == "user":
            lines.append(f"You: {msg.content}")
        else:
            status = getattr(msg, "status", None)
            is_error = bool(getattr(msg, "is_error", False)) or status == "error"
            prefix = "Assistant (issue)" if is_error else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
            if status and status != "complete":
                lines.append(f"  Status: {status}")
            reasoning = getattr(msg, "reasoning", None)
            if reasoning:
                lines.append("  Reasoning:")
                lines.append(f"    {reasoning}")
            trace = getattr(msg, "trace", None) or []
            if isinstance(trace, dict):
                trace = [trace]
            if trace:
                lines.append("  Agent trace:")
                for item in trace:
                    if isinstance(item, dict):
                        step = str(item.get("step", "")).replace("_", " ").title()
                        detail = str(
                            item.get("detail")
                            or item.get("query")
                            or item.get("standalone_question")
                            or ""
                        ).strip()
                        label = f"{step}: {detail}" if detail else step
                    else:
                        label = str(item)
                    lines.append(f"    - {label}")
            sources = getattr(msg, "sources", None)
            if sources:
                if isinstance(sources, dict):
                    sources = [sources]
                lines.append("  Sources:")
                for src in sources:
                    if isinstance(src, dict):
                        filename = src.get("filename") or src.get("document_name") or "Document"
                        page = src.get("page", "?")
                        snippet = src.get("snippet", "")
                    else:
                        filename = src.filename
                        page = src.page
                        snippet = src.snippet
                    lines.append(f"    - {filename} (page {page}): {snippet}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def friendly_error(exc: BaseException) -> str:
    """Map backend exceptions to calm, specific copy—never a traceback."""
    if isinstance(exc, ValueError):
        msg = str(exc).strip()
        if "Could not connect to tenant" in msg or "Could not connect to a Chroma server" in msg:
            return (
                "We could not prepare your documents for search on this device. "
                "Click **Start over**, then **Process documents** again. "
                "If it keeps happening, restart the app."
            )
        if "GROQ_API_KEY" in msg:
            return (
                "This app is not set up yet on this computer. "
                "Ask your administrator to configure access, then refresh the page."
            )
        if msg.startswith("Unsupported file type"):
            ext = msg.split(":")[-1].strip() if ":" in msg else "that type"
            return (
                f"We cannot read {ext} files here. "
                "Use PDF (.pdf) or plain text (.txt) only."
            )
        if "No documents found" in msg:
            return (
                "We could not find any readable text in those files. "
                "Try a different PDF or text file."
            )
        return msg or "Something went wrong with those files. Please try again."

    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "process documents" in msg or "before asking" in msg:
            return "Upload and process your documents first, then ask your question."
        return "Please process your documents before asking a question."

    return (
        "Something unexpected happened while we worked on that. "
        "Try again in a moment, or use Start over and re-upload your files."
    )


def uploaded_file_names(files: Sequence[Any]) -> list[str]:
    return sorted(getattr(f, "name", "") or "Unknown file" for f in files)


def selection_differs_from_processed(
    uploaded: Sequence[Any] | None,
    processed_names: list[str],
) -> bool:
    if not processed_names or not uploaded:
        return False
    return uploaded_file_names(uploaded) != sorted(processed_names)


EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "What are the main topics in these documents?",
    "Summarize the key points in simple language.",
    "Are there any dates or deadlines I should know about?",
    "Who is this document written for?",
    "What steps or actions does it recommend?",
)

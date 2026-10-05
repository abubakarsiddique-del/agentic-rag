from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class Conversation:
    id: str
    title: str | None
    created_at: str
    updated_at: str
    message_count: int = 0
    document_count: int = 0
    user_id: str | None = None


@dataclass(slots=True)
class MessageRecord:
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str
    reasoning: str | None = None
    trace: list[dict[str, Any]] | dict[str, Any] | None = None
    sources: list[dict[str, Any]] | dict[str, Any] | None = None
    rating: int | None = None
    feedback: str | None = None
    status: Literal["complete", "stopped", "error"] = "complete"


@dataclass(slots=True)
class MemoryTurn:
    id: str
    conversation_id: str
    question: str
    summary: str
    document_names: list[str]
    created_at: str
    user_id: str | None = None


@dataclass(slots=True)
class DocumentRecord:
    id: str
    conversation_id: str
    filename: str
    sha256: str | None
    size_bytes: int
    pages: int
    chunks: int
    status: Literal["queued", "processing", "ready", "failed"]
    error_code: str | None
    error_message: str | None
    created_at: str
    user_id: str | None = None


@dataclass(slots=True)
class IndexedDocument:
    id: str
    conversation_id: str
    document_id: str
    filename: str
    source_path: str | None = None
    chunk_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    user_id: str | None = None

"""Session state initialization and RAGService lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from agentic_rag import RAGService
from persistence.models import IndexedDocument, MessageRecord
from persistence.store import SQLiteConversationStore

ProcessedSettings = tuple[int, int, int]  # chunk_size, chunk_overlap, top_k

SESSION_DEFAULTS: dict[str, Any] = {
    "rag_service": None,
    "conversation_store": None,
    "active_conversation_id": None,
    "messages": [],
    "messages_backup": [],
    "documents_processed": False,
    "processed_file_names": [],
    "processed_settings": None,
    "last_success_message": None,
    "advanced_chunk_size": 800,
    "advanced_chunk_overlap": 150,
    "advanced_top_k": 4,
    "next_question": None,
    "answer_in_progress": False,
    "_active_request": False,
    "pipeline_step": "upload",
    "pipeline_phase": "prepare",  # prepare | question
    "file_status": {},  # filename -> pending | processing | done | skipped | failed
    "confirm_start_over": False,
    "session_reset_notice": False,
}


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (
                value.copy() if isinstance(value, dict) else value
            )


def get_store() -> SQLiteConversationStore:
    store = st.session_state.get("conversation_store")
    if store is None:
        db_path = Path(__file__).resolve().parent.parent / "conversations.db"
        store = SQLiteConversationStore(db_path=db_path)
        st.session_state.conversation_store = store
    return store


def get_active_conversation_id() -> str:
    store = get_store()
    conv_id = st.session_state.get("active_conversation_id")
    if not conv_id:
        conversation = store.create_conversation(title="Default conversation")
        st.session_state.active_conversation_id = conversation.id
        return conversation.id

    if store.get_conversation(conv_id) is None:
        conversation = store.create_conversation(title="Default conversation")
        st.session_state.active_conversation_id = conversation.id
        return conversation.id

    return conv_id


def set_active_conversation(conv_id: str | None) -> str | None:
    if conv_id is None:
        st.session_state.active_conversation_id = None
        return None

    store = get_store()
    if store.get_conversation(conv_id) is None:
        raise KeyError(f"Conversation not found: {conv_id}")

    st.session_state.active_conversation_id = conv_id
    return conv_id


def load_messages_from_store(conv_id: str | None = None) -> list[MessageRecord]:
    target_id = conv_id or get_active_conversation_id()
    return get_store().get_messages(target_id)


def persist_uploaded_file_metadata(
    file_names: list[str],
    *,
    metadata: dict[str, Any] | None = None,
) -> list[IndexedDocument]:
    """Persist the processed document file list to the active conversation."""
    if not file_names:
        return []

    conversation_id = get_active_conversation_id()
    store = get_store()
    details: list[IndexedDocument] = []
    payload = metadata or {}

    for index, file_name in enumerate(file_names, start=1):
        details.append(
            IndexedDocument(
                id=f"doc_{conversation_id}_{index}",
                conversation_id=conversation_id,
                document_id=f"doc_{conversation_id}_{index}",
                filename=file_name,
                source_path=str(payload.get("source_path", file_name)),
                chunk_count=int(payload.get("chunk_count", 0) or 0),
                metadata={
                    **payload,
                    "filename": file_name,
                    "source": file_name,
                    "uploaded_at": payload.get("uploaded_at"),
                },
                created_at=payload.get("created_at") or __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            )
        )

    return store.register_documents(conversation_id, details)


def _service_has_documents(service: Any) -> bool:
    if service is None:
        return False

    method = getattr(service, "has_documents", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            pass

    chunks = getattr(service, "chunks", None)
    documents = getattr(service, "documents", None)
    return bool(chunks or documents)


def is_chat_ready() -> bool:
    service = st.session_state.get("rag_service")
    processed = bool(st.session_state.get("documents_processed", False))
    has_docs = _service_has_documents(service)

    if processed and service is not None:
        return True

    if service is not None and has_docs:
        st.session_state.documents_processed = True
        return True

    return False


def reset_session() -> None:
    # Call RAGService cleanup hooks before wiping session state
    service = st.session_state.get("rag_service")
    try:
        if service is not None and hasattr(service, "cleanup_chroma_store"):
            service.cleanup_chroma_store()
    except Exception:
        pass

    for key, value in SESSION_DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, dict) else value


def current_advanced_settings() -> ProcessedSettings:
    return (
        int(st.session_state.advanced_chunk_size),
        int(st.session_state.advanced_chunk_overlap),
        int(st.session_state.advanced_top_k),
    )


def advanced_settings_changed() -> bool:
    if not st.session_state.documents_processed:
        return False
    processed: ProcessedSettings | None = st.session_state.processed_settings
    if processed is None:
        return False
    return current_advanced_settings() != processed


def service_settings_match(service: RAGService) -> bool:
    chunk, overlap, top_k = current_advanced_settings()
    return (
        service.chunk_size == chunk
        and service.chunk_overlap == overlap
        and service.top_k == top_k
    )


def get_or_create_service(*, for_processing: bool = False) -> RAGService:
    """
    Return the session RAGService.

    When for_processing is False, never replace an indexed instance just because
    advanced widgets changed — chat gating handles that mismatch.
    """
    chunk, overlap, top_k = current_advanced_settings()
    service: RAGService | None = st.session_state.rag_service
    conversation_id = str(get_active_conversation_id())

    if service is None:
        service = RAGService(
            chunk_size=chunk,
            chunk_overlap=overlap,
            top_k=top_k,
            conversation_id=conversation_id,
        )
        st.session_state.rag_service = service
        return service

    if getattr(service, "conversation_id", None) != conversation_id:
        service = RAGService(
            chunk_size=chunk,
            chunk_overlap=overlap,
            top_k=top_k,
            conversation_id=conversation_id,
        )
        st.session_state.rag_service = service
        return service

    if for_processing and not service_settings_match(service):
        service = RAGService(
            chunk_size=chunk,
            chunk_overlap=overlap,
            top_k=top_k,
            conversation_id=conversation_id,
        )
        st.session_state.rag_service = service

    return service


def mark_documents_processed(file_names: list[str]) -> None:
    st.session_state.documents_processed = True
    st.session_state.processed_file_names = file_names
    st.session_state.processed_settings = current_advanced_settings()
    st.session_state.messages = []
    st.session_state.answer_in_progress = False
    st.session_state.pipeline_step = "ready"
    st.session_state.pipeline_phase = "prepare"

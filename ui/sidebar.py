"""Sidebar: uploads, processing, advanced settings, reset."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from app_helpers import (
    ChatMessage,
    RejectedFile,
    SourcePreview,
    format_index_success,
    friendly_error,
    groq_api_key_configured,
    partition_uploaded_files,
    selection_differs_from_processed,
)
from ui.state import (
    advanced_settings_changed,
    get_active_conversation_id,
    get_or_create_service,
    get_store,
    load_messages_from_store,
    mark_documents_processed,
    persist_uploaded_file_metadata,
    reset_session,
    set_active_conversation,
)


def _status_icon(status: str) -> str:
    return {
        "pending": "○",
        "processing": "⏳",
        "done": "✅",
        "skipped": "⊘",
        "failed": "✕",
    }.get(status, "○")


def _status_label(status: str) -> str:
    return {
        "pending": "Waiting",
        "processing": "Processing",
        "done": "Done",
        "skipped": "Skipped",
        "failed": "Failed",
    }.get(status, status.replace("_", " "))


def _render_file_status_list(
    valid_names: list[str],
    rejected: list[RejectedFile],
) -> None:
    status_map: dict[str, str] = st.session_state.get("file_status") or {}
    if not status_map and not rejected:
        return

    st.markdown("**File status**")
    for name in valid_names:
        status = status_map.get(name, "pending")
        st.markdown(f"{_status_icon(status)} {name} — {_status_label(status)}")
    for item in rejected:
        st.markdown(
            f"{_status_icon('skipped')} {item.name} — Skipped ({item.reason})"
        )


def _render_document_status(uploaded: list[Any] | None) -> None:
    if not st.session_state.documents_processed:
        return

    st.status("Documents ready", state="complete", expanded=False)
    if st.session_state.last_success_message:
        st.caption(st.session_state.last_success_message)

    st.markdown("**Documents loaded**")
    for name in st.session_state.processed_file_names:
        st.markdown(f"- {name}")

    if selection_differs_from_processed(
        uploaded, st.session_state.processed_file_names
    ):
        st.warning(
            "You selected different files in the uploader. "
            "Click **Process documents** to update what we search."
        )

    if advanced_settings_changed():
        st.warning(
            "Your advanced settings changed. "
            "Click **Process documents** again for them to take effect."
        )


def _relative_time(value: str | None) -> str:
    if not value:
        return "just now"

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "just now"

    now = datetime.now(timezone.utc)
    delta_seconds = max(int((now - dt).total_seconds()), 0)
    if delta_seconds < 60:
        return "just now"
    if delta_seconds < 3600:
        minutes = delta_seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if delta_seconds < 86400:
        hours = delta_seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = delta_seconds // 86400
    return f"{days} day{'s' if days != 1 else ''} ago"


def _to_ui_messages(records: list[Any]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for record in records:
        sources = None
        if isinstance(record.sources, list):
            sources = []
            for item in record.sources:
                if isinstance(item, SourcePreview):
                    sources.append(item)
                elif isinstance(item, dict):
                    sources.append(
                        SourcePreview(
                            filename=str(item.get("filename") or item.get("document_name") or "Unknown file"),
                            page=item.get("page", "?"),
                            snippet=str(item.get("snippet") or ""),
                            document_name=item.get("document_name"),
                            document_path=item.get("document_path"),
                            score=item.get("score"),
                        )
                    )

        trace = record.trace if isinstance(record.trace, list) else None
        messages.append(
            ChatMessage(
                role=record.role,
                content=record.content,
                sources=sources,
                reasoning=record.reasoning,
                trace=trace,
            )
        )
    return messages


def render_conversation_history() -> None:
    store = get_store()
    convs = store.list_conversations()
    query = st.text_input(
        "Search conversations",
        label_visibility="collapsed",
        placeholder="Search conversations",
        key="conversation_search",
    )
    if query:
        query_lower = query.lower()
        convs = [conv for conv in convs if query_lower in conv.title.lower()]

    if st.button("➕ New conversation", use_container_width=True):
        conversation = store.create_conversation(title="New conversation")
        set_active_conversation(conversation.id)
        st.session_state.messages = []
        st.session_state.documents_processed = False
        st.session_state.processed_file_names = []
        st.session_state.processed_settings = None
        st.session_state.last_success_message = None
        st.session_state.file_status = {}
        st.rerun()

    if not convs:
        st.caption("No conversations yet")
        return

    for conv in convs:
        active = conv.id == get_active_conversation_id()
        cols = st.columns([5, 1])
        with cols[0]:
            label = f"{'🟢 ' if active else ''}{conv.title}"
            if st.button(label, key=f"conv_{conv.id}", use_container_width=True):
                set_active_conversation(conv.id)
                st.session_state.messages = _to_ui_messages(load_messages_from_store(conv.id))
                documents = store.get_documents(conv.id)
                st.session_state.documents_processed = bool(documents)
                st.session_state.processed_file_names = [doc.filename for doc in documents]
                settings = documents[0].metadata if documents else {}
                if settings:
                    st.session_state.advanced_chunk_size = int(
                        settings.get("chunk_size", st.session_state.advanced_chunk_size)
                    )
                    st.session_state.advanced_chunk_overlap = int(
                        settings.get("chunk_overlap", st.session_state.advanced_chunk_overlap)
                    )
                    st.session_state.advanced_top_k = int(
                        settings.get("top_k", st.session_state.advanced_top_k)
                    )
                    st.session_state.processed_settings = (
                        st.session_state.advanced_chunk_size,
                        st.session_state.advanced_chunk_overlap,
                        st.session_state.advanced_top_k,
                    )
                else:
                    st.session_state.processed_settings = None
                st.session_state.rag_service = None
                st.rerun()
        with cols[1]:
            if st.button("🗑", key=f"delete_{conv.id}", use_container_width=True):
                deleted = store.delete_conversation(conv.id)
                if deleted:
                    chroma_dir = Path(__file__).resolve().parent.parent / ".chroma_store" / conv.id
                    if chroma_dir.exists():
                        shutil.rmtree(chroma_dir, ignore_errors=True)
                    current_id = st.session_state.get("active_conversation_id")
                    if current_id == conv.id:
                        remaining = store.list_conversations()
                        st.session_state.active_conversation_id = remaining[0].id if remaining else None
                        st.session_state.messages = []
                        st.session_state.documents_processed = False
                        st.session_state.processed_file_names = []
                        st.session_state.processed_settings = None
                    st.rerun()

        messages = load_messages_from_store(conv.id)
        st.caption(f"{len(messages)} messages · {_relative_time(conv.updated_at)}")


def _render_start_over() -> None:
    if st.session_state.get("confirm_start_over"):
        st.warning("This clears your uploaded session and chat history.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, start over", use_container_width=True, type="primary"):
                reset_session()
                st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_start_over = False
                st.rerun()
        return

    if st.button("Start over", use_container_width=True):
        st.session_state.confirm_start_over = True
        st.rerun()


def render_sidebar() -> list[Any] | None:
    with st.sidebar:
        render_conversation_history()

        st.header("Your documents")
        st.caption("PDF and plain text (.txt) only.")

        uploaded = st.file_uploader(
            "Add files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="visible",
            help="Select one or more PDF or text files.",
        )

        if uploaded and not st.session_state.documents_processed:
            st.session_state.pipeline_step = "upload"
            st.session_state.pipeline_phase = "prepare"

        if not groq_api_key_configured():
            st.error(
                "This app is not set up on this computer yet. "
                "Ask whoever manages this machine to configure the API key, "
                "then refresh the page."
            )
            with st.expander("For administrators"):
                st.caption("Set the GROQ_API_KEY environment variable, then restart the app.")

        if st.button(
            "Process documents",
            type="primary",
            use_container_width=True,
            disabled=not groq_api_key_configured(),
        ):
            all_files = list(uploaded or [])
            if not all_files:
                st.error("Choose at least one PDF or text file to continue.")
            else:
                valid, rejected = partition_uploaded_files(all_files)
                if rejected:
                    for item in rejected:
                        st.warning(f'"{item.name}": {item.reason}')
                if not valid:
                    st.error("No supported files to process. Add PDF or .txt files.")
                else:
                    try:
                        names = [f.name for f in valid]
                        st.session_state.file_status = {
                            n: "processing" for n in names
                        }
                        for item in rejected:
                            st.session_state.file_status[item.name] = "skipped"

                        service = get_or_create_service(for_processing=True)
                        st.session_state.pipeline_phase = "prepare"
                        st.session_state.pipeline_step = "load"

                        with st.spinner("Working on your files…"):
                            with st.status(
                                "Preparing your documents…", expanded=True
                            ) as status:
                                st.write("Reading your files…")
                                st.session_state.pipeline_step = "load"
                                st.session_state.pipeline_step = "split"
                                stats = service.build_index(valid)
                                st.session_state.pipeline_step = "embed"
                                st.session_state.pipeline_step = "store"
                                st.write("Getting everything searchable…")
                                st.session_state.pipeline_step = "ready"
                                status.update(
                                    label="Documents ready",
                                    state="complete",
                                    expanded=False,
                                )

                        persist_uploaded_file_metadata(
                            names,
                            metadata={
                                "chunk_count": stats.get("chunks", 0),
                                "chunk_size": st.session_state.advanced_chunk_size,
                                "chunk_overlap": st.session_state.advanced_chunk_overlap,
                                "top_k": st.session_state.advanced_top_k,
                            },
                        )
                        st.session_state.file_status = {n: "done" for n in names}
                        for item in rejected:
                            st.session_state.file_status[item.name] = "skipped"

                        st.session_state.last_success_message = format_index_success(
                            stats, names
                        )
                        mark_documents_processed(names)
                    except Exception as exc:  # noqa: BLE001
                        st.session_state.documents_processed = False
                        st.session_state.processed_file_names = []
                        st.session_state.processed_settings = None
                        st.session_state.pipeline_step = "upload"
                        for n in st.session_state.get("file_status", {}):
                            if st.session_state.file_status[n] == "processing":
                                st.session_state.file_status[n] = "failed"
                        st.error(friendly_error(exc))

        if uploaded:
            valid, rejected = partition_uploaded_files(uploaded)
            _render_file_status_list([f.name for f in valid], rejected)

        _render_document_status(uploaded)

        st.divider()

        with st.expander("Advanced settings", expanded=False):
            st.session_state.advanced_chunk_size = st.number_input(
                "Section size (characters)",
                min_value=200,
                max_value=4000,
                value=int(st.session_state.advanced_chunk_size),
                step=100,
                help="Larger values keep more context together; smaller values focus on tighter passages.",
            )
            st.session_state.advanced_chunk_overlap = st.number_input(
                "Section overlap (characters)",
                min_value=0,
                max_value=800,
                value=int(st.session_state.advanced_chunk_overlap),
                step=25,
                help="How much neighboring text overlaps between sections. Usually leave at the default.",
            )
            st.session_state.advanced_top_k = st.number_input(
                "Passages checked per question",
                min_value=1,
                max_value=12,
                value=int(st.session_state.advanced_top_k),
                step=1,
                help="How many parts of your documents to consult for each answer.",
            )
            st.caption("Changes apply the next time you click **Process documents**.")

        st.divider()
        _render_start_over()

    return uploaded

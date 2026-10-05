"""Streamlit frontend tailored for the agentic RAG backend."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app_helpers import (
    ChatMessage,
    SourcePreview,
    convert_source_dicts,
    partition_uploaded_files,
    documents_to_sources,
    friendly_error,
    serialize_sources,
    sources_are_empty,
    answer_indicates_no_information,
    summarize_answer_effort,
)
from persistence.models import MessageRecord
from ui.state import (
    advanced_settings_changed,
    get_active_conversation_id,
    get_or_create_service,
    get_store,
    init_session_state,
    is_chat_ready,
    load_messages_from_store,
    mark_documents_processed,
    reset_session,
    set_active_conversation,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

if st.session_state.get("answer_in_progress") and not st.session_state.get("_active_request", False):
    st.session_state.answer_in_progress = False

if st.session_state.get("documents_processed") and st.session_state.get("rag_service") is None:
    st.session_state.documents_processed = False
    st.session_state._active_request = False
    st.warning("Your session was reset. Please re-upload your documents.")

st.set_page_config(
    page_title="Agentic Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _status_badge(status: str) -> str:
    mapping = {
        "pending": "○ Waiting",
        "processing": "⏳ Processing",
        "done": "✅ Ready",
        "failed": "✕ Failed",
        "skipped": "⊘ Skipped",
    }
    return mapping.get(status, status)


def derive_title(question: str) -> str:
    cleaned = " ".join((question or "").strip().split())
    if not cleaned:
        return "New conversation"
    words = cleaned.split()
    title = " ".join(words[:6])
    if len(words) > 6:
        title += "…"
    return title.strip() or "New conversation"


def _hydrate_session_from_store() -> None:
    conv_id = get_active_conversation_id()
    records = load_messages_from_store(conv_id)
    st.session_state.messages = [
        ChatMessage(
            role=record.role,
            content=record.content,
            sources=convert_source_dicts(getattr(record, "sources", None)),
            is_error=bool(getattr(record, "is_error", False)),
            no_sources=bool(getattr(record, "no_sources", False)),
            reasoning=getattr(record, "reasoning", None),
            trace=getattr(record, "trace", None),
        )
        for record in records
    ]

    if not st.session_state.messages and st.session_state.get("messages_backup"):
        st.session_state.messages = list(st.session_state.messages_backup)

    store = get_store()
    documents = store.get_documents(conv_id)
    st.session_state.documents_processed = bool(documents)
    st.session_state.processed_file_names = [doc.filename for doc in documents]


def _personalized_suggestions(file_names: list[str]) -> list[str]:
    names = [f.replace(".", " ").replace("_", " ").strip() for f in file_names]
    primary = names[0] if names else "document"
    base = [
        f"Summarize the main points in {primary}.",
        f"What are the key takeaways from {primary}?",
        f"List the important dates, risks, or actions in {primary}.",
        f"Who is this document written for?",
        f"What decisions or recommendations does it make?",
    ]
    if len(file_names) > 1:
        return base[:4] + [f"Compare the main ideas across {len(file_names)} uploaded files."]
    return base


def _render_source_cards(sources: list, message_key: str, no_match: bool = False) -> None:
    label = "Sources used for this answer"
    if no_match:
        label = "Sources (no close match found)"

    with st.expander(label, expanded=no_match):
        if no_match or not sources:
            st.caption("No passages were confident enough to answer this.")
            return

        for idx, src in enumerate(sources, start=1):
            with st.container(border=True):
                st.markdown(f"**{src.filename}** · Page {src.page}")
                st.markdown(f"> {src.snippet or '(No preview available)'}")
                st.download_button(
                    label=f"Download excerpt {idx}",
                    data=src.snippet or "",
                    file_name=f"{src.filename}_p{src.page}_excerpt.txt",
                    mime="text/plain",
                    key=f"src_dl_{message_key}_{idx}",
                    use_container_width=True,
                )


def _conversation_summary_text(messages: list[ChatMessage]) -> str:
    if not messages:
        return "No conversation yet."

    lines = []
    for msg in messages:
        if msg.role == "user":
            lines.append(f"Question: {msg.content}")
        elif msg.role == "assistant" and not msg.is_error:
            lines.append(f"Answer: {msg.content}")

    if not lines:
        return "No conversation yet."

    questions = [m.content for m in messages if m.role == "user"]
    answers = [m.content for m in messages if m.role == "assistant" and not m.is_error]
    summary = (
        f"This conversation includes {len(questions)} question(s) and {len(answers)} answer(s). "
        f"Main topics discussed: {', '.join(questions[:3])}."
    )
    return summary


def _render_export_tools() -> None:
    if not st.session_state.messages:
        return

    export_text = "\n\n".join(
        (
            f"User: {msg.content}" if msg.role == "user" else f"Assistant: {msg.content}"
        )
        for msg in st.session_state.messages
    )

    summary_text = _conversation_summary_text(st.session_state.messages)

    with st.expander("Conversation tools", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download conversation (.txt)",
                data=export_text,
                file_name="agentic_rag_conversation.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="Download summary (.txt)",
                data=summary_text,
                file_name="agentic_rag_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.text_area("Conversation summary", value=summary_text, height=120, disabled=True)


def _render_history() -> None:
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "🧑" if msg.role == "user" else "📄"
        with st.chat_message(msg.role, avatar=avatar):
            if msg.role == "assistant" and msg.is_error:
                st.error(msg.content)
            elif msg.role == "assistant" and msg.no_sources:
                st.warning(msg.content)
            else:
                st.markdown(msg.content)

            if msg.role == "assistant" and not msg.is_error:
                effort = summarize_answer_effort(getattr(msg, "trace", None))
                st.caption(f"Effort: {effort}")

            reasoning = getattr(msg, "reasoning", None)
            if msg.role == "assistant" and reasoning:
                with st.expander("Reasoning summary", expanded=False):
                    st.markdown(reasoning)

            trace = getattr(msg, "trace", None)
            if msg.role == "assistant" and trace:
                with st.expander("Step-by-step trace", expanded=False):
                    for item in trace:
                        st.markdown(f"- {item}")

            if msg.role == "assistant" and msg.sources is not None:
                _render_source_cards(msg.sources, str(idx), no_match=getattr(msg, "no_sources", False))

            if msg.role == "assistant" and not msg.is_error:
                st.download_button(
                    label="Download answer (.txt)",
                    data=msg.content,
                    file_name=f"answer_{idx + 1}.txt",
                    mime="text/plain",
                    key=f"download_answer_{idx}",
                    use_container_width=True,
                )


def _render_onboarding() -> None:
    if st.session_state.documents_processed:
        return

    with st.container(border=True):
        st.markdown("### 🧭 Start with your documents")
        st.markdown(
            "Upload PDFs or text files, process them, and ask practical questions about the content. "
            "The agentic version decides when retrieval is needed and when it can answer directly."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Upload**\nChoose your PDF or .txt files")
        with col2:
            st.markdown("**2. Process**\nBuild searchable document chunks")
        with col3:
            st.markdown("**3. Ask**\nUse natural-language questions")

        st.write("")
        sample_questions = [
            "Summarize the main ideas in these documents.",
            "What are the key actions or recommendations?",
            "List the important dates, risks, and deadlines.",
        ]
        for q in sample_questions:
            if st.button(q, key=f"sample_{q[:12]}", use_container_width=True):
                st.session_state.next_question = q
                st.rerun()


def _render_sidebar() -> list | None:
    with st.sidebar:
        st.header("Your documents")
        st.caption("PDF and plain text (.txt) files only.")

        uploaded = st.file_uploader(
            "Add files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="visible",
        )

        if not st.session_state.get("GROQ_API_KEY_CHECKED"):
            st.session_state.GROQ_API_KEY_CHECKED = True

        if not st.session_state.get("rag_service") and st.session_state.documents_processed:
            st.session_state.documents_processed = False

        if not st.session_state.get("documents_processed"):
            st.caption("Waiting for documents to be processed.")

        if st.button(
            "Process documents",
            type="primary",
            use_container_width=True,
            disabled=not bool(__import__("os").getenv("GROQ_API_KEY", "").strip()),
        ):
            if not uploaded:
                st.error("Choose at least one PDF or text file before processing.")
            else:
                valid, rejected = partition_uploaded_files(uploaded)
                if rejected:
                    for item in rejected:
                        st.warning(f'"{item.name}": {item.reason}')
                if not valid:
                    st.error("No supported files were selected. Use PDF or .txt only.")
                else:
                    try:
                        names = [f.name for f in valid]
                        st.session_state.file_status = {n: "processing" for n in names}
                        for item in rejected:
                            st.session_state.file_status[item.name] = "skipped"

                        service = get_or_create_service(for_processing=True)
                        with st.status("Preparing your documents…", expanded=True) as status:
                            st.write("Reading your files…")
                            st.session_state.pipeline_step = "load"
                            stats = service.build_index(valid)
                            st.write("Chunking and indexing text…")
                            st.session_state.pipeline_step = "split"
                            st.write("Building the searchable vector store…")
                            st.session_state.pipeline_step = "store"
                            status.update(label="Documents ready", state="complete", expanded=False)

                        st.session_state.file_status = {n: "done" for n in names}
                        for item in rejected:
                            st.session_state.file_status[item.name] = "skipped"

                        mark_documents_processed(names)
                        st.session_state.last_success_message = (
                            f"Ready to go. We indexed {len(valid)} file(s) and created {stats.get('chunks', 0)} searchable chunks."
                        )
                        st.success(st.session_state.last_success_message)
                    except Exception as exc:  # noqa: BLE001
                        st.session_state.documents_processed = False
                        st.session_state.processed_file_names = []
                        st.session_state.processed_settings = None
                        st.session_state.pipeline_step = "upload"
                        for name, state in list(st.session_state.get("file_status", {}).items()):
                            if state == "processing":
                                st.session_state.file_status[name] = "failed"
                        st.error(friendly_error(exc))

        if uploaded:
            valid, rejected = partition_uploaded_files(uploaded)
            for item in valid:
                label = st.session_state.get("file_status", {}).get(item.name, "pending")
                st.markdown(f"{_status_badge(label)} · {item.name}")
            for item in rejected:
                st.markdown(f"{_status_badge('skipped')} · {item.name} (skipped)")

        with st.expander("Advanced settings", expanded=False):
            st.session_state.advanced_chunk_size = st.number_input(
                "Section size (characters)",
                min_value=200,
                max_value=4000,
                value=int(st.session_state.advanced_chunk_size),
                step=100,
            )
            st.session_state.advanced_chunk_overlap = st.number_input(
                "Section overlap (characters)",
                min_value=0,
                max_value=800,
                value=int(st.session_state.advanced_chunk_overlap),
                step=25,
            )
            st.session_state.advanced_top_k = st.number_input(
                "Passages checked per question",
                min_value=1,
                max_value=12,
                value=int(st.session_state.advanced_top_k),
                step=1,
            )

        st.divider()
        if st.button("Start over", use_container_width=True, type="secondary"):
            reset_session()
            st.rerun()

    return uploaded


def _selected_document_scope() -> list[str]:
    file_names = list(st.session_state.get("processed_file_names") or [])
    if not file_names:
        return []

    selected = st.session_state.get("document_scope")
    if selected is None:
        st.session_state.document_scope = list(file_names)
        return list(file_names)

    valid = [name for name in selected if name in file_names]
    if not valid:
        st.session_state.document_scope = list(file_names)
        return list(file_names)

    st.session_state.document_scope = valid
    return valid


def _filter_retrieved_docs_by_scope(retrieved_docs: list, selected_scope: list[str]) -> list:
    if not retrieved_docs:
        return []
    if not selected_scope:
        return list(retrieved_docs)

    normalized_scope = {os.path.basename(name) for name in selected_scope}
    filtered: list = []
    for doc in retrieved_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        sources = [
            metadata.get("source"),
            metadata.get("document_name"),
            getattr(doc, "source", None),
            getattr(doc, "document_name", None),
        ]
        if any(os.path.basename(str(value)) in normalized_scope for value in sources if value is not None):
            filtered.append(doc)
    return filtered if filtered else []


def _render_prompt_box(ready: bool) -> None:
    if not is_chat_ready():
        st.info("Upload and process a PDF or .txt file to unlock the chat experience.")
        return

    if not ready:
        st.caption("Re-process your documents after changing advanced settings.")

    file_names = st.session_state.get("processed_file_names") or []
    if file_names:
        scope = _selected_document_scope()
        st.caption("Focus the answer on specific files")
        selected = st.multiselect(
            "Files in scope",
            options=file_names,
            default=scope,
            key="document_scope_multiselect",
            help="Keep this narrow when your question only applies to one or two loaded files.",
        )
        st.session_state.document_scope = list(selected) if selected else list(file_names)

    question = st.session_state.get("next_question")
    if question:
        st.session_state.next_question = None
        _queue_question(question)
        return

    if prompt := st.chat_input("Ask a question about your documents…"):
        _queue_question(prompt)


def _queue_question(question: str) -> None:
    question = question.strip()
    if not question:
        st.warning("Type a question before sending.")
        return

    if not is_chat_ready():
        st.warning("Upload and process your documents first, then ask your question.")
        return

    if advanced_settings_changed():
        st.warning("Your advanced settings changed. Process documents again before asking.")
        return

    conv_id = get_active_conversation_id()
    store = get_store()
    conversation = store.get_conversation(conv_id)
    if conversation and conversation.title == "New conversation":
        store.rename_conversation(conv_id, derive_title(question))

    user_record = MessageRecord(
        id=f"msg_{conv_id}_{len(st.session_state.messages) + 1}",
        conversation_id=conv_id,
        role="user",
        content=question,
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        reasoning=None,
        trace=None,
        sources=None,
    )
    store.append_message(conv_id, user_record)
    st.session_state.messages.append(ChatMessage(role="user", content=question))
    st.session_state.messages_backup = list(st.session_state.messages)
    st.session_state.answer_in_progress = True
    st.session_state._active_request = True
    st.session_state.pipeline_phase = "question"
    st.session_state.pipeline_step = "retrieve"
    st.rerun()


def _stream_assistant_reply() -> None:
    question = st.session_state.messages[-1].content
    service = st.session_state.rag_service
    msg_key = f"live_{len(st.session_state.messages)}"
    conv_id = get_active_conversation_id()
    store = get_store()

    st.session_state.answer_in_progress = True
    st.session_state._active_request = True
    st.session_state.last_question_trace = []
    try:
        with st.chat_message("assistant", avatar="📄"):
            try:
                with st.spinner("Checking relevance and searching the uploaded documents…"):
                    answer, retrieved_docs, reasoning = service.ask_with_reasoning(question)

                selected_scope = _selected_document_scope()
                filtered_docs = _filter_retrieved_docs_by_scope(retrieved_docs, selected_scope)
                if filtered_docs:
                    retrieved_docs = filtered_docs

                with st.status("Generating answer…", expanded=False):
                    pass

                trace = getattr(service, "last_trace", []) or []
                st.session_state.last_question_trace = trace
                trace_labels = _trace_to_labels(trace)
                sources = documents_to_sources(retrieved_docs)
                no_match = sources_are_empty(sources) or answer_indicates_no_information(answer)

                if no_match and not answer_indicates_no_information(answer):
                    answer = (
                        "No clear match was found in the uploaded documents. "
                        "Try rephrasing the question or upload a more relevant file."
                    )

                if reasoning:
                    with st.expander("Reasoning summary", expanded=False):
                        st.markdown(reasoning)

                if trace_labels:
                    with st.expander("Step-by-step trace", expanded=False):
                        for item in trace_labels:
                            st.markdown(f"- {item}")

                st.caption(f"Effort: {summarize_answer_effort(trace)}")
                _render_source_cards(sources, msg_key, no_match=no_match)

                assistant_record = MessageRecord(
                    id=f"msg_{conv_id}_{len(st.session_state.messages) + 1}",
                    conversation_id=conv_id,
                    role="assistant",
                    content=answer,
                    created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    reasoning=reasoning,
                    trace=trace_labels,
                    sources=serialize_sources(sources),
                )
                store.append_message(conv_id, assistant_record)

                st.session_state.messages.append(
                    ChatMessage(
                        role="assistant",
                        content=answer,
                        sources=sources,
                        no_sources=no_match,
                        reasoning=reasoning,
                        trace=trace_labels,
                    )
                )
                st.session_state.messages_backup = list(st.session_state.messages)
            except Exception as exc:  # noqa: BLE001
                message = friendly_error(exc)
                st.error(message)
                try:
                    assistant_record = MessageRecord(
                        id=f"msg_{conv_id}_{len(st.session_state.messages) + 1}",
                        conversation_id=conv_id,
                        role="assistant",
                        content=message,
                        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                        reasoning=None,
                        trace=None,
                        sources=None,
                    )
                    store.append_message(conv_id, assistant_record)
                except Exception:
                    pass
                st.session_state.messages.append(
                    ChatMessage(role="assistant", content=message, sources=None, is_error=True)
                )
                st.session_state.messages_backup = list(st.session_state.messages)
    finally:
        st.session_state.answer_in_progress = False
        st.session_state._active_request = False
        st.rerun()


def _render_suggested_questions() -> None:
    if not st.session_state.documents_processed:
        return
    if advanced_settings_changed() or st.session_state.answer_in_progress:
        return

    file_names = st.session_state.processed_file_names
    suggestions = _personalized_suggestions(file_names)
    with st.expander("Suggested questions", expanded=not bool(st.session_state.messages)):
        st.caption("These suggestions are tailored to the files you uploaded.")
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{suggestion[:20]}", use_container_width=True):
                _queue_question(suggestion)


def _trace_to_labels(trace: list[dict] | None) -> list[str]:
    if not trace:
        return []

    labels: list[str] = []
    for item in trace:
        step = str(item.get("step", "")).lower()
        detail = str(item.get("detail", "")).strip()
        if step == "route":
            labels.append(f"Route: {detail}")
        elif step == "retrieve":
            labels.append(f"Retrieve: {detail}")
        elif step == "grade":
            labels.append(f"Grade: {detail}")
        elif step == "rewrite":
            labels.append(f"Rewrite: {detail}")
        elif step == "generate":
            labels.append(f"Generate: {detail}")
        elif step == "direct_answer":
            labels.append(f"Direct answer: {detail}")
    return labels


def _render_agent_workflow() -> None:
    steps = [
        ("1", "Upload", "Add PDF or .txt files"),
        ("2", "Route", "Decide if retrieval is needed"),
        ("3", "Retrieve", "Search the indexed passages"),
        ("4", "Grade", "Check if the match is strong enough"),
        ("5", "Answer", "Reply with sources and reasoning"),
    ]

    cols = st.columns(len(steps))
    for col, (number, title, detail) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div style="padding:0.8rem 0.7rem;border:1px solid rgba(255,255,255,0.08);border-radius:12px;background:rgba(255,255,255,0.02);min-height:110px;">
                    <div style="font-size:0.75rem;opacity:0.75;">Step {number}</div>
                    <div style="font-size:1.1rem;font-weight:700;margin-top:0.35rem;">{title}</div>
                    <div style="font-size:0.82rem;opacity:0.8;margin-top:0.3rem;">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_agent_trace_panel(trace: list[dict] | None = None, *, expanded: bool = True) -> None:
    trace_items = _trace_to_labels(trace)
    if not trace_items:
        return

    with st.expander("Agent trace", expanded=expanded):
        for item in trace_items:
            st.markdown(f"- {item}")


def _render_stepper() -> None:
    if not st.session_state.documents_processed:
        return

    trace = st.session_state.get("last_question_trace") or []
    step_counts = {
        "Upload": 1 if st.session_state.documents_processed else 0,
        "Route": sum(1 for item in trace if str(item.get("step", "")).lower() == "route"),
        "Retrieve": sum(1 for item in trace if str(item.get("step", "")).lower() == "retrieve"),
        "Verify": sum(1 for item in trace if str(item.get("step", "")).lower() == "grade"),
        "Answer": sum(1 for item in trace if str(item.get("step", "")).lower() in {"generate", "direct_answer"}),
    }
    stages = [
        ("Upload", step_counts["Upload"] > 0),
        ("Route", step_counts["Route"] > 0),
        ("Retrieve", step_counts["Retrieve"] > 0),
        ("Verify", step_counts["Verify"] > 0),
        ("Answer", step_counts["Answer"] > 0),
    ]

    cols = st.columns(len(stages))
    for col, (label, ready) in zip(cols, stages):
        with col:
            badge = ""
            count = step_counts.get(label, 0)
            if label in {"Retrieve", "Verify"} and count > 1:
                badge = f' <span style="font-size:0.75rem;opacity:0.9;">×{count}</span>'
            marker = "✅" if ready else "○"
            bg = "#1f2a38" if ready else "#2d2d2d"
            st.markdown(
                f'<div style="padding:0.7rem 0.8rem;border-radius:12px;background:{bg};text-align:center;font-weight:600;">{marker} {label}{badge}</div>',
                unsafe_allow_html=True,
            )


def _detect_session_desync() -> bool:
    missing = []
    service = st.session_state.get("rag_service")
    if st.session_state.get("documents_processed") is True and service is None:
        missing.append("rag_service")
    if st.session_state.get("documents_processed") is True and getattr(service, "chunks", None) in (None, []) and getattr(service, "documents", None) in (None, []):
        missing.append("service_has_no_documents")

    if not missing:
        return False

    print("[session reset] missing keys:", missing)
    st.session_state.documents_processed = False
    st.session_state.session_reset_notice = True
    st.session_state.messages = []
    st.session_state.processed_file_names = []
    st.session_state.processed_settings = None
    st.session_state.answer_in_progress = False
    st.session_state.pipeline_step = "upload"
    st.session_state.pipeline_phase = "prepare"
    st.session_state.rag_service = None
    return True


def main() -> None:
    init_session_state()
    _hydrate_session_from_store()

    if not __import__("os").getenv("GROQ_API_KEY", "").strip():
        st.title("Agentic Document Q&A")
        st.warning(
            "GROQ_API_KEY is missing or not loaded from the project .env file. "
            "Update the key in the RAG/.env file and restart the app."
        )
        st.stop()

    _detect_session_desync()

    service = st.session_state.get("rag_service")
    if service is not None and getattr(service, "documents", None) and not st.session_state.get("documents_processed"):
        st.session_state.documents_processed = True

    st.title("Agentic Document Q&A")
    st.caption("A smarter document Q&A experience with routing, retries, and source visibility.")

    if st.session_state.get("session_reset_notice"):
        st.warning("🔄 Session was reset — re-upload your files.")
        st.session_state.session_reset_notice = False

    _render_sidebar()
    _render_stepper()
    if not is_chat_ready():
        _render_agent_workflow()
        _render_onboarding()
        st.info("Upload and process a PDF or .txt file to unlock the chat experience.")

    _render_export_tools()
    _render_suggested_questions()
    _render_history()

    if st.session_state.answer_in_progress and st.session_state.messages and st.session_state.messages[-1].role == "user":
        _stream_assistant_reply()
        return

    if is_chat_ready():
        _render_prompt_box(ready=True)
    else:
        st.info("Upload and process a PDF or .txt file to unlock the chat experience.")


main()

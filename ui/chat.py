"""Main chat pane: history, suggested questions, streaming answers."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import streamlit as st

from app_helpers import (
    ChatMessage,
    SourcePreview,
    answer_indicates_no_information,
    convert_source_dicts,
    documents_to_sources,
    format_conversation_export,
    friendly_error,
    group_sources_by_document,
    serialize_sources,
    sources_are_empty,
    summarize_answer_effort,
)
from persistence.models import MessageRecord
from ui.state import (
    advanced_settings_changed,
    get_active_conversation_id,
    get_or_create_service,
    get_store,
    load_messages_from_store,
)


def _record_to_chat_message(record: object) -> ChatMessage:
    sources = convert_source_dicts(getattr(record, "sources", None))
    trace = getattr(record, "trace", None)
    return ChatMessage(
        role=getattr(record, "role", "user"),
        content=getattr(record, "content", ""),
        sources=sources,
        is_error=bool(getattr(record, "is_error", False)),
        no_sources=bool(getattr(record, "no_sources", False)),
        reasoning=getattr(record, "reasoning", None),
        trace=trace if isinstance(trace, list) else None,
    )


def _trace_to_labels(trace: list[dict] | None) -> list[str]:
    if not trace:
        return []

    labels = []
    label_names = {
        "route": "Route",
        "retrieve": "Retrieve",
        "grade": "Grade",
        "rewrite": "Rewrite",
        "generate": "Generate",
        "direct_answer": "Direct answer",
        "abstain": "Abstain",
    }
    for item in trace:
        if not isinstance(item, dict):
            labels.append(str(item))
            continue
        step = str(item.get("step", "")).lower()
        label = label_names.get(step)
        detail = str(item.get("detail", "")).strip()
        if label:
            labels.append(f"{label}: {detail}" if detail else label)
    return labels


def _derive_title(question: str) -> str:
    words = " ".join((question or "").strip().split()).split()
    if not words:
        return "New conversation"
    title = " ".join(words[:6])
    return f"{title}…" if len(words) > 6 else title


def _persist_message(
    role: str,
    content: str,
    *,
    reasoning: str | None = None,
    trace: list[dict] | None = None,
    sources: list[SourcePreview] | None = None,
) -> None:
    conversation_id = get_active_conversation_id()
    record = MessageRecord(
        id=f"msg_{uuid4().hex}",
        conversation_id=conversation_id,
        role=role,
        content=content,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        reasoning=reasoning,
        trace=trace,
        sources=serialize_sources(sources),
    )
    get_store().append_message(conversation_id, record)


def _personalized_suggestions(file_names: list[str]) -> list[str]:
    names = [name.rsplit(".", 1)[0].replace("_", " ").strip() for name in file_names]
    primary = names[0] if names else "document"
    suggestions = [
        f"Summarize the main points in {primary}.",
        f"What are the key takeaways from {primary}?",
        f"List the important dates, risks, or actions in {primary}.",
        "Who is this document written for?",
        "What decisions or recommendations does it make?",
    ]
    if len(file_names) > 1:
        return suggestions[:4] + [
            f"Compare the main ideas across {len(file_names)} uploaded files."
        ]
    return suggestions


def _conversation_summary_text(messages: list[ChatMessage]) -> str:
    questions = [msg.content for msg in messages if msg.role == "user"]
    answers = [
        msg.content
        for msg in messages
        if msg.role == "assistant" and not msg.is_error
    ]
    if not questions and not answers:
        return "No conversation yet."
    topics = ", ".join(questions[:3]) or "No questions recorded"
    return (
        f"This conversation includes {len(questions)} question(s) and "
        f"{len(answers)} answer(s). Main topics discussed: {topics}."
    )


def _render_document_scope() -> list[str]:
    file_names = list(st.session_state.get("processed_file_names") or [])
    if not file_names:
        return []

    conversation_id = get_active_conversation_id()
    key = f"document_scope_{conversation_id}"
    if key not in st.session_state:
        st.session_state[key] = list(file_names)
    else:
        st.session_state[key] = [
            name for name in st.session_state[key] if name in file_names
        ] or list(file_names)

    selected = st.multiselect(
        "Files in scope",
        options=file_names,
        key=key,
        help="Limit retrieval and answer generation to selected files.",
    )
    return list(selected) or file_names


def _render_source_cards(
    sources: list[SourcePreview],
    *,
    message_key: str,
    no_match: bool = False,
) -> None:
    label = "Sources used for this answer"
    if no_match:
        label = "Sources (none closely matched your question)"

    grouped = group_sources_by_document(sources)
    source_names = list(grouped.keys())
    if source_names:
        st.caption(f"📚 Sources: {', '.join(source_names)}")

    with st.expander(label, expanded=no_match):
        if no_match or not sources:
            st.warning(
                "We could not find passages in your files that clearly match this question. "
                "Try rephrasing, or check that the right documents were processed."
            )
            return

        for doc_name, grouped_sources in grouped.items():
            st.markdown(f"**{doc_name}**")
            for idx, src in enumerate(grouped_sources, start=1):
                with st.container(border=True):
                    st.markdown(f"Page {src.page} · Excerpt {idx}")
                    st.markdown(f"> {src.snippet or '(No preview available)'}")
                    st.download_button(
                        label=f"Download excerpt {idx}",
                        data=src.snippet or "",
                        file_name=f"{doc_name}_p{src.page}_excerpt_{idx}.txt",
                        mime="text/plain",
                        key=f"src_dl_{message_key}_{doc_name}_{idx}_{hash(str(src.page) + str(src.snippet))}",
                        use_container_width=True,
                    )


def _render_export_controls() -> None:
    conv_id = get_active_conversation_id()
    records = load_messages_from_store(conv_id)
    if not records:
        return
    export_messages = [_record_to_chat_message(record) for record in records]
    export_text = format_conversation_export(export_messages)
    summary_text = _conversation_summary_text(export_messages)
    with st.expander("Conversation tools", expanded=False):
        st.caption(
            "This export is for your records. The assistant still treats each new "
            "question independently—it does not remember earlier turns."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download conversation (.txt)",
                data=export_text,
                file_name="document_qa_conversation.txt",
                mime="text/plain",
                key="export_conversation",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="Download summary (.txt)",
                data=summary_text,
                file_name="document_qa_summary.txt",
                mime="text/plain",
                key="export_summary",
                width="stretch",
            )
        st.text_area(
            "Conversation summary",
            value=summary_text,
            height=120,
            disabled=True,
        )


def render_chat_history() -> None:
    conv_id = get_active_conversation_id()
    records = load_messages_from_store(conv_id)
    for msg_idx, record in enumerate(records):
        msg = _record_to_chat_message(record)
        avatar = "🧑" if msg.role == "user" else "📄"
        with st.chat_message(msg.role, avatar=avatar):
            if msg.role == "assistant" and msg.is_error:
                st.error(msg.content)
            elif msg.role == "assistant" and msg.no_sources:
                st.warning(msg.content)
            else:
                st.markdown(msg.content)

            if msg.role == "assistant" and msg.sources is not None:
                _render_source_cards(
                    msg.sources,
                    message_key=f"{conv_id}_{msg_idx}",
                    no_match=msg.no_sources,
                )

            if msg.role == "assistant" and not msg.is_error:
                raw_trace = msg.trace if msg.trace and isinstance(msg.trace[0], dict) else None
                st.caption(f"Effort: {summarize_answer_effort(raw_trace)}")

            if msg.role == "assistant" and msg.reasoning:
                with st.expander("Reasoning", expanded=False):
                    st.markdown(msg.reasoning)

            if msg.role == "assistant" and msg.trace:
                trace_items = _trace_to_labels(msg.trace)
                if trace_items:
                    with st.expander("Agent trace", expanded=False):
                        for item in trace_items:
                            st.markdown(f"- {item}")

            if msg.role == "assistant" and not msg.is_error:
                st.download_button(
                    label="Download answer (.txt)",
                    data=msg.content,
                    file_name=f"answer_{msg_idx + 1}.txt",
                    mime="text/plain",
                    key=f"download_answer_{conv_id}_{msg_idx}",
                    width="stretch",
                )


def _awaiting_assistant_reply() -> bool:
    messages = st.session_state.messages
    return bool(messages) and messages[-1].role == "user"


def _stream_assistant_reply() -> None:
    question = st.session_state.messages[-1].content
    service = get_or_create_service()
    msg_key = f"live_{len(st.session_state.messages)}"

    st.session_state.pipeline_phase = "question"
    st.session_state.pipeline_step = "retrieve"

    with st.chat_message("assistant", avatar="📄"):
        try:
            # Note: ask_stream currently returns the full answer as a single
            # chunk for compatibility with legacy frontends. Make the UI
            # messaging explicit so users are not misled about token-by-token
            # streaming.
            with st.spinner("Checking relevance and searching the uploaded documents…"):
                stream_gen, retrieved = service.ask_stream(
                    question,
                    document_scope=st.session_state.get("active_document_scope") or None,
                )
            st.session_state.pipeline_step = "answer"

            # Normalize to an iterator: backend may return a callable generator
            stream_iter = stream_gen() if callable(stream_gen) else stream_gen

            # Prepare placeholders for incremental updates
            answer_placeholder = st.empty()
            progress_placeholder = st.empty()
            stop_stream = st.button("Stop stream", key=f"stop_{msg_key}")

            sources = documents_to_sources(retrieved)
            source_placeholders = []
            with st.expander("Sources (progressive)", expanded=True):
                if not sources:
                    st.caption("No retrieved passages available yet.")
                else:
                    for idx, src in enumerate(sources):
                        ph = st.empty()
                        source_placeholders.append(ph)

            # Stream tokens / chunks and update UI progressively
            accumulated = ""
            sentence_buffer = ""
            import re

            sentence_end = re.compile(r"([.!?]+)\s+")

            try:
                for chunk in stream_iter:
                    if stop_stream:
                        break

                    text = str(chunk)
                    accumulated += text

                    # Buffer and flush on sentence boundaries for smoother updates
                    sentence_buffer += text
                    parts = sentence_end.split(sentence_buffer)
                    flushed = ""
                    # Reconstruct segments where sentence_end splits into groups
                    i = 0
                    while i + 2 < len(parts):
                        flushed += parts[i] + parts[i + 1]
                        i += 2

                    # Remaining buffer is whatever wasn't a full sentence yet
                    sentence_buffer = "".join(parts[i:])

                    if flushed:
                        answer_placeholder.markdown(accumulated)
                        progress_placeholder.caption(f"Streaming… {len(accumulated)} characters received")

                    # Reveal source previews progressively as more content arrives
                    reveal_count = min(len(source_placeholders), 1 + len(accumulated) // 200)
                    for i in range(reveal_count):
                        src = sources[i]
                        source_placeholders[i].markdown(f"**{src.filename}** · Page {src.page}\n> {src.snippet}")

                # flush any remaining buffered text
                if sentence_buffer:
                    answer_placeholder.markdown(accumulated)
                    progress_placeholder.caption(f"Streaming… {len(accumulated)} characters received")

            except Exception:  # noqa: BLE001
                raise

            # Finalize answer and compute no_match
            answer = accumulated
            sources = documents_to_sources(retrieved)
            trace = getattr(service, "last_trace", []) or []
            reasoning = getattr(service, "last_reasoning", None)
            no_match = sources_are_empty(sources) or answer_indicates_no_information(
                answer
            )

            if no_match and not answer_indicates_no_information(answer):
                answer = (
                    "We found some text in your files, but nothing that clearly answers "
                    "this question. Try asking in a different way or upload a more relevant document."
                )

            if reasoning:
                with st.expander("Reasoning summary", expanded=False):
                    st.markdown(reasoning)

            trace_labels = _trace_to_labels(trace)
            if trace_labels:
                with st.expander("Step-by-step trace", expanded=False):
                    for item in trace_labels:
                        st.markdown(f"- {item}")
            st.caption(f"Effort: {summarize_answer_effort(trace)}")

            _render_source_cards(
                sources,
                message_key=msg_key,
                no_match=no_match,
            )
            _persist_message(
                "assistant",
                answer,
                reasoning=reasoning,
                trace=trace,
                sources=sources,
            )
            st.session_state.messages.append(
                ChatMessage(
                    role="assistant",
                    content=answer,
                    sources=sources,
                    no_sources=no_match,
                    reasoning=reasoning,
                    trace=trace,
                ),
            )
            st.session_state.pipeline_step = "answer"
        except Exception as exc:  # noqa: BLE001
            message = friendly_error(exc)
            st.error(message)
            try:
                _persist_message("assistant", message)
            except Exception:
                pass
            st.session_state.messages.append(
                ChatMessage(
                    role="assistant",
                    content=message,
                    sources=None,
                    is_error=True,
                ),
            )
            st.session_state.pipeline_phase = "prepare"

    st.session_state.answer_in_progress = False
    st.rerun(scope="fragment")


def _queue_question(question: str) -> None:
    question = question.strip()
    if not question:
        st.warning("Type a question before sending.")
        return

    if not st.session_state.documents_processed:
        st.warning("Upload and process your documents first, then ask your question.")
        return

    if advanced_settings_changed():
        st.warning(
            "Your advanced settings changed. "
            "Process documents again in the sidebar before asking."
        )
        return

    conversation_id = get_active_conversation_id()
    store = get_store()
    conversation = store.get_conversation(conversation_id)
    if conversation and conversation.title in {"New conversation", "Default conversation"}:
        store.rename_conversation(conversation_id, _derive_title(question))
    try:
        _persist_message("user", question)
    except Exception as exc:  # noqa: BLE001
        st.error(friendly_error(exc))
        return

    st.session_state.messages.append(ChatMessage(role="user", content=question))
    st.session_state.answer_in_progress = True
    st.session_state.pipeline_phase = "question"
    st.session_state.pipeline_step = "retrieve"
    st.rerun(scope="fragment")


def _render_suggested_questions() -> None:
    if not st.session_state.documents_processed:
        return
    if advanced_settings_changed() or st.session_state.get("answer_in_progress"):
        return

    has_messages = bool(st.session_state.messages)
    file_names = list(st.session_state.get("processed_file_names") or [])
    with st.expander(
        "Suggested questions",
        expanded=not has_messages,
    ):
        st.caption(
            "Each answer is based on your files only. Earlier messages in this chat "
            "are shown for your reference—the assistant does not remember them."
        )
        for idx, example in enumerate(_personalized_suggestions(file_names)):
            if st.button(
                example,
            key=f"suggest_{get_active_conversation_id()}_{idx}",
                use_container_width=True,
            ):
                _queue_question(example)


@st.fragment
def render_chat_panel() -> None:
    if "answer_in_progress" not in st.session_state:
        st.session_state.answer_in_progress = False

    _render_export_controls()
    render_chat_history()

    scope = _render_document_scope() if st.session_state.documents_processed else []
    st.session_state.active_document_scope = scope

    if st.session_state.answer_in_progress and _awaiting_assistant_reply():
        _stream_assistant_reply()
        return

    _render_suggested_questions()

    ready = (
        st.session_state.documents_processed
        and not advanced_settings_changed()
        and not st.session_state.answer_in_progress
    )
    if not st.session_state.documents_processed:
        st.caption("Process your documents in the sidebar to unlock the question box.")
    elif advanced_settings_changed():
        st.caption(
            "Re-process your documents in the sidebar after changing advanced settings."
        )

    question = st.session_state.next_question
    if question:
        st.session_state.next_question = None
        _queue_question(question)
        return

    if prompt := st.chat_input(
        "Ask about your documents…",
        disabled=not ready,
    ):
        _queue_question(prompt)


def render_stepper() -> None:
    ready = st.session_state.documents_processed
    asked = bool(st.session_state.messages)
    step_one = "① Upload & process" + (" ✓" if ready else "")
    step_two = "② Ask questions" + (" ✓" if asked else "")

    col1, col2 = st.columns(2)
    with col1:
        css_one = "step-done" if ready else "step-todo"
        st.markdown(f'<p class="{css_one}">{step_one}</p>', unsafe_allow_html=True)
    with col2:
        css_two = "step-done" if ready else "step-todo"
        st.markdown(f'<p class="{css_two}">{step_two}</p>', unsafe_allow_html=True)

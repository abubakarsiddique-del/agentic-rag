"""
Unified Streamlit frontend combining the original shell and the agentic app UI.

This single entrypoint preserves the behavior of the previous `agentic_app.py` while
maintaining the page metadata and basic error handling from the original `app.py`.
"""

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
from ui.chat import render_chat_panel, render_stepper
from ui.pipeline_viz import render_pipeline_visualization
from ui.sidebar import render_sidebar
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


st.set_page_config(
    page_title="Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Document Q&A** lets you upload PDF or text files and ask questions "
            "in everyday language. Each answer uses your uploaded files only—not "
            "earlier chat messages."
        ),
        "Report a bug": "mailto:support@example.com?subject=Document%20Q%26A%20issue",
    },
)


def _render_app_shell() -> None:
    init_session_state()
    uploaded = render_sidebar()

    st.title("Document Q&A")
    st.markdown(
        "Upload your PDFs or text files, then ask questions in everyday language. "
        "Each answer is generated fresh from your files—not from earlier messages in this chat."
    )

    render_stepper()
    render_pipeline_visualization(uploaded)

    if not st.session_state.documents_processed:
        st.info(
            "**Get started:** use the sidebar to add files and click **Process documents**. "
            "When step ① shows a checkmark, you can ask questions below."
        )

    render_chat_panel()


def main() -> None:
    try:
        _render_app_shell()
    except Exception:  # noqa: BLE001 — never show raw tracebacks to end users
        st.error(
            "Something went wrong while loading this page. "
            "Try refreshing, or click **Start over** in the sidebar and upload your files again."
        )


if __name__ == "__main__":
    main()

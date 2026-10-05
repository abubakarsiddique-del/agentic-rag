"""Pipeline diagram and step highlights tied to session state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import streamlit as st

from app_helpers import selection_differs_from_processed

PREPARE_STEPS: tuple[tuple[str, str, str], ...] = (
    ("upload", "Your files added", "You chose PDF or text files to work with."),
    ("load", "Reading text", "We pull readable text out of each file."),
    ("split", "Organizing sections", "Long documents are split into smaller passages."),
    ("embed", "Preparing search", "Each passage is indexed so we can look things up quickly."),
    ("store", "Saving for lookup", "Everything is stored in memory for this session only."),
    ("ready", "Ready for questions", "You can ask questions in plain language below."),
)

QUESTION_STEPS: tuple[tuple[str, str, str], ...] = (
    ("retrieve", "Finding passages", "We look for parts of your files that match your question."),
    ("answer", "Writing the answer", "We draft a reply using only those passages—not earlier chat messages."),
)

STEP_ORDER = [key for key, _, _ in PREPARE_STEPS]


def architecture_image_path() -> Path | None:
    base = Path(__file__).resolve().parent.parent
    for candidate in (
        base / "assets" / "rag_architecture.png",
        base / "RAG Architecture.png",
        base.parent / "RAG Architecture.png",
    ):
        if candidate.is_file():
            return candidate
    return None


def _step_index(step_key: str, order: list[str]) -> int:
    try:
        return order.index(step_key)
    except ValueError:
        return 0


def infer_prepare_step(uploaded: Sequence[Any] | None) -> str:
    explicit = st.session_state.get("pipeline_step")
    if explicit and explicit in STEP_ORDER:
        return explicit

    if st.session_state.get("documents_processed"):
        if selection_differs_from_processed(
            uploaded, st.session_state.get("processed_file_names") or []
        ):
            return "upload"
        return "ready"

    if uploaded:
        return "upload"
    return "upload"


def _render_step_list(
    steps: tuple[tuple[str, str, str], ...],
    active_key: str,
) -> None:
    order = [k for k, _, _ in steps]
    active_idx = _step_index(active_key, order)
    for key, title, _help in steps:
        idx = _step_index(key, order)
        if idx < active_idx:
            st.markdown(f"✅ **{title}**")
        elif idx == active_idx:
            st.markdown(f"▶ **{title}**")
        else:
            st.markdown(f"○ {title}")


def _active_help_text(
    steps: tuple[tuple[str, str, str], ...],
    active_key: str,
) -> str:
    for key, _, help_text in steps:
        if key == active_key:
            return help_text
    return ""


def render_pipeline_visualization(uploaded: Sequence[Any] | None) -> None:
    has_upload = bool(uploaded)
    has_processed = bool(st.session_state.get("documents_processed"))
    if not has_upload and not has_processed:
        return

    phase = st.session_state.get("pipeline_phase", "prepare")
    prepare_step = infer_prepare_step(uploaded)
    image_path = architecture_image_path()

    with st.container(border=True):
        st.subheader("What happens to your documents")
        st.caption(
            "This diagram shows the journey from your files to answers. "
            "Each step updates while you work."
        )

        if image_path:
            st.image(
                str(image_path),
                use_container_width=True,
                caption="Overview: files → preparation → your questions → answers",
            )

        if phase == "question":
            q_step = st.session_state.get("pipeline_step", "retrieve")
            if q_step not in {s[0] for s in QUESTION_STEPS}:
                q_step = "retrieve"
            st.markdown("**Your last question**")
            _render_step_list(QUESTION_STEPS, q_step)
            st.info(_active_help_text(QUESTION_STEPS, q_step))
        else:
            st.markdown("**Preparing your files**")
            _render_step_list(PREPARE_STEPS, prepare_step)
            help_text = _active_help_text(PREPARE_STEPS, prepare_step)
            if help_text:
                st.caption(help_text)

            if prepare_step == "ready":
                st.success("Your files are ready. Ask a question in the chat below.")
            elif has_upload and not has_processed:
                st.caption("Click **Process documents** in the sidebar to continue.")
            elif prepare_step in {"load", "split", "embed", "store"}:
                st.caption("Please wait—we are still preparing your files…")

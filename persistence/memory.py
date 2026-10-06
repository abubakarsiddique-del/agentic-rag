"""Global, low-trust cross-session memory backed by SQLite and user-sharded Chroma Cloud."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.documents import Document

from .models import MemoryTurn
from .store import SQLiteConversationStore

logger = logging.getLogger(__name__)


def summarize_turn_answer(answer: str, *, max_chars: int = 600) -> str:
    normalized = " ".join((answer or "").split())
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", normalized) if part]
    summary = " ".join(sentences[:3]) or normalized
    if len(summary) <= max_chars:
        return summary
    excerpt = summary[:max_chars]
    boundary = max(excerpt.rfind("."), excerpt.rfind("!"), excerpt.rfind("?"))
    if boundary >= max_chars // 2:
        return excerpt[: boundary + 1]
    return excerpt.rsplit(" ", 1)[0].rstrip(" ,;:") + "..."


class ConversationMemory:
    def __init__(self, store: SQLiteConversationStore, *, vector_store=None) -> None:
        self.store = store
        self._vector_store = vector_store
        self._vector_stores: dict[str, object] = {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            from agentic_rag import _create_isolated_chroma_client

            self._client = _create_isolated_chroma_client("_memory")
        return self._client

    def _get_vector_store(self, owner_id: str | None = None):
        if self._vector_store is not None:
            return self._vector_store
        key = owner_id or "legacy"
        if key not in self._vector_stores:
            from chroma_cloud import (
                ChromaCloudVectorStore,
                get_or_create_collection,
                memory_collection_name,
            )

            client = self._get_client()
            collection = get_or_create_collection(client, memory_collection_name(owner_id))
            self._vector_stores[key] = ChromaCloudVectorStore(
                client,
                collection,
                group_by_document=False,
            )
        return self._vector_stores[key]

    def record_turn(
        self,
        conversation_id: str,
        question: str,
        summary: str,
        document_names: list[str],
    ) -> MemoryTurn | None:
        conversation = self.store.get_conversation(conversation_id)
        owner_id = conversation.user_id if conversation else None
        if not self.store.get_global_memory_enabled(owner_id):
            return None
        if not self.store.get_conversation_memory_enabled(conversation_id):
            return None
        clean_question = " ".join((question or "").split())[:1000]
        clean_summary = summarize_turn_answer(summary)
        if not clean_question or not clean_summary:
            return None

        turn = MemoryTurn(
            id=str(uuid4()),
            conversation_id=conversation_id,
            question=clean_question,
            summary=clean_summary,
            document_names=list(dict.fromkeys(str(name) for name in document_names if name)),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            user_id=owner_id,
        )
        self.store.append_memory_turn(turn)
        metadata = {
            "memory_id": turn.id,
            "conversation_id": conversation_id,
            "title": conversation.title if conversation else "Past conversation",
            "question": turn.question,
            "summary": turn.summary,
            "document_names": json.dumps(turn.document_names, ensure_ascii=False),
            "created_at": turn.created_at,
        }
        if owner_id is not None:
            metadata["user_id"] = owner_id
        try:
            self._get_vector_store(owner_id).add_documents(
                [Document(
                    page_content=f"Question: {turn.question}\nAnswer summary: {turn.summary}",
                    metadata=metadata,
                )],
                ids=[turn.id],
            )
        except Exception:  # noqa: BLE001 - memory indexing must not fail a completed answer
            logger.exception("Could not index memory turn %s", turn.id)
        return turn

    def search(
        self,
        query: str,
        *,
        exclude_conversation_id: str | None = None,
        limit: int = 3,
        owner_id: str | None = None,
    ) -> list[dict]:
        if not self.store.get_global_memory_enabled(owner_id) or limit < 1:
            return []
        allowed_conversations = (
            self.store.list_conversation_ids_for_owner(owner_id)
            if owner_id is not None
            else None
        )
        vector_store = self._get_vector_store(owner_id)
        search_options = {"k": max(limit * 10, limit)}
        if owner_id is not None:
            search_options["filter"] = {"user_id": owner_id}
        matches = vector_store.similarity_search(query, **search_options)
        hits = []
        seen_turns = set()
        for document in matches:
            metadata = dict(document.metadata or {})
            conversation_id = str(metadata.get("conversation_id") or "")
            turn_id = str(metadata.get("memory_id") or "")
            if not conversation_id or conversation_id == exclude_conversation_id or turn_id in seen_turns:
                continue
            if allowed_conversations is not None and conversation_id not in allowed_conversations:
                continue
            if not self.store.get_conversation_memory_enabled(conversation_id):
                continue
            seen_turns.add(turn_id)
            try:
                document_names = json.loads(metadata.get("document_names", "[]"))
            except (TypeError, json.JSONDecodeError):
                document_names = []
            hits.append({
                "memory_id": turn_id,
                "conversation_id": conversation_id,
                "title": str(metadata.get("title") or "Past conversation"),
                "question": str(metadata.get("question") or ""),
                "summary": str(metadata.get("summary") or ""),
                "document_names": document_names if isinstance(document_names, list) else [],
                "created_at": str(metadata.get("created_at") or ""),
            })
            if len(hits) >= limit:
                break
        return hits

    def delete_conversation(self, conversation_id: str) -> int:
        conversation = self.store.get_conversation(conversation_id)
        owner_id = conversation.user_id if conversation else None
        ids = self.store.delete_memory_turns(conversation_id)
        if not ids:
            return 0
        try:
            self._get_vector_store(owner_id).delete(where={"conversation_id": conversation_id})
        except Exception:  # noqa: BLE001 - cleanup should continue with SQLite as source of truth
            logger.exception("Could not delete memory vectors for conversation %s", conversation_id)
        return len(ids)

    def clear_all(self, *, owner_id: str | None = None) -> int:
        ids = self.store.delete_memory_turns(owner_id=owner_id)
        if owner_id is not None:
            if ids:
                try:
                    self._get_vector_store(owner_id).delete(ids=ids)
                except Exception:  # noqa: BLE001 - SQLite remains the source of truth
                    logger.exception("Could not clear memory vectors for user %s", owner_id)
            return len(ids)
        try:
            if self._vector_store is not None and hasattr(self._vector_store, "clear"):
                self._vector_store.clear()
            elif self._vector_store is not None and ids:
                self._vector_store.delete(ids=ids)
            else:
                client = self._get_client()
                for collection in client.list_collections():
                    if collection.name.startswith("memory_"):
                        client.delete_collection(name=collection.name)
                self._vector_stores.clear()
        except Exception:  # noqa: BLE001 - do not fail settings actions on stale vectors
            logger.exception("Could not clear global memory vectors")
            self._vector_store = None
            self._vector_stores.clear()
        return len(ids)


def format_memory_hints(hits: list[dict], *, max_chars: int = 3000) -> str:
    if not hits:
        return "No related past-chat hints."
    lines = [
        "UNTRUSTED PAST-CHAT HINTS (for resolving references only; not evidence, never cite):"
    ]
    for index, hit in enumerate(hits, start=1):
        line = (
            f"{index}. Prior question: {hit.get('question', '')}\n"
            f"   Short summary: {hit.get('summary', '')}\n"
            f"   Prior files: {', '.join(hit.get('document_names') or [])}"
        )
        lines.append(line)
    return "\n".join(lines)[:max_chars]
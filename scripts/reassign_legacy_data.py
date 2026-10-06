"""Assign pre-authentication rows and memory vectors to one existing admin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from persistence.store import SQLiteConversationStore


def _update_memory_vector_owners(store: SQLiteConversationStore, user_id: str) -> int:
    import chromadb
    from langchain_core.documents import Document

    from chroma_cloud import (
        ChromaCloudVectorStore,
        create_cloud_client,
        get_or_create_collection,
        memory_collection_name,
    )

    client = create_cloud_client()
    try:
        source = client.get_collection(name=memory_collection_name(None))
    except chromadb.errors.NotFoundError:
        return 0
    turns = store.list_memory_turns(owner_id=user_id, limit=1_000_000)
    turn_ids = [turn.id for turn in turns]
    if not turn_ids:
        return 0

    target = ChromaCloudVectorStore(
        client,
        get_or_create_collection(client, memory_collection_name(user_id)),
        group_by_document=False,
    )
    updated = 0
    for offset in range(0, len(turn_ids), 500):
        result = source.get(
            ids=turn_ids[offset : offset + 500],
            include=["documents", "metadatas"],
        )
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        if not ids:
            continue
        documents = [
            Document(
                page_content=str(text or ""),
                metadata={**dict(metadata or {}), "user_id": user_id},
            )
            for text, metadata in zip(result.get("documents") or [], metadatas)
        ]
        target.upsert_documents(documents, ids=ids)
        source.delete(ids=ids)
        updated += len(ids)
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="existing admin account to receive ownerless data")
    parser.add_argument("--yes", action="store_true", help="confirm reassignment without prompting")
    args = parser.parse_args(argv)

    store = SQLiteConversationStore()
    user = store.get_user_by_email(args.email)
    if user is None or not user.get("is_admin"):
        parser.error("the target email must belong to an existing administrator")
    if not args.yes:
        try:
            answer = input(f"Assign all ownerless legacy rows to {user['email']}? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().casefold() not in {"y", "yes"}:
            print("No rows changed.")
            return 1

    counts = store.reassign_legacy_data(user["id"])
    counts["memory_vectors"] = _update_memory_vector_owners(store, user["id"])
    print(json.dumps({"owner_email": user["email"], "reassigned": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
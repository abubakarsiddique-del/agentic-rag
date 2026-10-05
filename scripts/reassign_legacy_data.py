"""Assign pre-authentication rows and memory vectors to one existing admin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persistence.store import SQLiteConversationStore


def _update_memory_vector_owners(store: SQLiteConversationStore, user_id: str) -> int:
    memory_dir = Path(__file__).resolve().parent.parent / ".chroma_store" / "_memory"
    if not (memory_dir / "chroma.sqlite3").is_file():
        return 0

    from agentic_rag import _create_isolated_chroma_client

    collection = _create_isolated_chroma_client("_memory").get_collection(name="global_memory")
    turns = store.list_memory_turns(owner_id=user_id, limit=1_000_000)
    turn_ids = [turn.id for turn in turns]
    if not turn_ids:
        return 0

    updated = 0
    for offset in range(0, len(turn_ids), 500):
        result = collection.get(ids=turn_ids[offset : offset + 500], include=["metadatas"])
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        if not ids:
            continue
        collection.update(
            ids=ids,
            metadatas=[{**dict(metadata or {}), "user_id": user_id} for metadata in metadatas],
        )
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
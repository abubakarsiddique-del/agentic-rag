"""Remove saved conversations and their Chroma Cloud collections when requested."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from persistence.store import SQLiteConversationStore
from chroma_cloud import delete_conversation_collection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the listed conversations and matching Chroma Cloud collections",
    )
    args = parser.parse_args()

    store = SQLiteConversationStore(db_path=PROJECT_ROOT / ".rag_history.db")
    conversations = store.list_conversations()

    print(f"Conversations to remove: {len(conversations)}")
    print(f"Chroma Cloud conversation collections to remove: {len(conversations)}")
    if not args.apply:
        print("Dry run only. Pass --apply to delete these records and Cloud collections.")
        return 0

    for conversation in conversations:
        delete_conversation_collection(conversation.id)
        store.delete_conversation(conversation.id)
    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

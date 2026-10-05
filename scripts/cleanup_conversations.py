"""Remove saved conversations and their Chroma folders when explicitly requested."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from persistence.store import SQLiteConversationStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the listed conversations and matching Chroma directories",
    )
    args = parser.parse_args()

    store = SQLiteConversationStore(db_path=PROJECT_ROOT / ".rag_history.db")
    conversations = store.list_conversations()
    chroma_root = PROJECT_ROOT / ".chroma_store"
    directories = [
        chroma_root / conversation.id
        for conversation in conversations
        if (chroma_root / conversation.id).is_dir()
    ]

    print(f"Conversations to remove: {len(conversations)}")
    print(f"Chroma directories to remove: {len(directories)}")
    if not args.apply:
        print("Dry run only. Pass --apply to delete these records and directories.")
        return 0

    for conversation in conversations:
        store.delete_conversation(conversation.id)
    for directory in directories:
        shutil.rmtree(directory)
    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

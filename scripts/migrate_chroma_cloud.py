"""Copy local Chroma records into user/conversation-sharded Chroma Cloud collections."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import chromadb
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chroma_cloud import (  # noqa: E402
    ChromaCloudVectorStore,
    collection_name,
    create_cloud_client,
    get_or_create_collection,
    memory_collection_name,
    split_documents_for_cloud,
)


def _metadata_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _migrate_collection(
    source_collection,
    *,
    source_name: str,
    destination_for_owner,
    batch_size: int,
    apply: bool,
) -> tuple[int, int]:
    total_records = source_collection.count()
    next_chunk_index: dict[tuple[str, str], int] = defaultdict(int)
    destination_stores: dict[str, ChromaCloudVectorStore] = {}
    client = create_cloud_client() if apply else None
    migrated_records = 0
    migrated_chunks = 0

    for offset in range(0, total_records, batch_size):
        batch = source_collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        destination_batches: dict[str, tuple[list[Document], list[str]]] = {}
        for old_id, text, old_metadata in zip(
            batch["ids"],
            batch.get("documents") or [],
            batch.get("metadatas") or [],
        ):
            if not text or not text.strip():
                continue
            old_metadata = dict(old_metadata or {})
            source = str(old_metadata.get("source") or old_metadata.get("document_name") or "legacy")
            document_id = str(
                old_metadata.get("source_document_id")
                or old_metadata.get("document_id")
                or uuid5(NAMESPACE_URL, f"{source_name}:{source}")
            )
            owner_id = old_metadata.get("user_id") if source_name == "global_memory" else None
            destination_name = destination_for_owner(owner_id)
            metadata = {
                key: _metadata_value(value)
                for key, value in old_metadata.items()
                if key not in {"sparse_embedding", "embedding"}
            }
            metadata.update({
                "document_id": document_id,
                "source_document_id": document_id,
                "source": source,
            })
            document = Document(page_content=text, metadata=metadata)
            pieces, _ = split_documents_for_cloud(
                [document],
                chunk_size=12 * 1024,
                chunk_overlap=0,
            )
            output_documents, output_ids = destination_batches.setdefault(
                destination_name,
                ([], []),
            )
            for piece_index, piece in enumerate(pieces, start=1):
                counter_key = (destination_name, document_id)
                next_chunk_index[counter_key] += 1
                chunk_index = next_chunk_index[counter_key]
                piece.metadata["chunk_index"] = chunk_index
                piece.metadata["passage_id"] = f"{document_id}:{chunk_index}"
                stable_id = str(uuid5(
                    NAMESPACE_URL,
                    f"{source_name}:{old_id}:{piece_index}",
                ))
                output_documents.append(piece)
                output_ids.append(stable_id)
            migrated_records += 1
            migrated_chunks += len(pieces)

        if apply:
            for destination_name, (documents, ids) in destination_batches.items():
                store = destination_stores.get(destination_name)
                if store is None:
                    if client is None:
                        raise RuntimeError("Chroma Cloud client was not initialized for an apply migration.")
                    collection = get_or_create_collection(client, destination_name)
                    store = ChromaCloudVectorStore(
                        client,
                        collection,
                        group_by_document=source_name != "global_memory",
                    )
                    destination_stores[destination_name] = store
                store.upsert_documents(documents, ids=ids)

    return migrated_records, migrated_chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=PROJECT_ROOT / ".chroma_store",
        help="existing local Chroma directory (default: .chroma_store)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="number of local records read per batch",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write/re-embed records in Chroma Cloud; without this, only print a dry run",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not args.local_root.is_dir():
        parser.error(f"local Chroma directory does not exist: {args.local_root}")

    report = []
    with tempfile.TemporaryDirectory(prefix="chroma-cloud-migration-") as temp_dir:
        local_collections = []
        for directory in sorted(path for path in args.local_root.iterdir() if path.is_dir()):
            legacy_collection_name = "global_memory" if directory.name == "_memory" else collection_name(directory.name)
            copied_directory = Path(temp_dir) / directory.name
            shutil.copytree(directory, copied_directory)
            local_client = chromadb.PersistentClient(path=str(copied_directory))
            known_names = {
                getattr(item, "name", item)
                for item in local_client.list_collections()
            }
            if legacy_collection_name in known_names:
                local_collections.append((directory, copied_directory, legacy_collection_name))

        if not local_collections:
            print(f"No local Chroma collections found under {args.local_root}.")
            return 0

        for directory, copied_directory, name in local_collections:
            local_client = chromadb.PersistentClient(path=str(copied_directory))
            source = local_client.get_collection(name=name)
            if name == "global_memory":
                destination = memory_collection_name
            else:
                destination = lambda _owner, conversation_id=directory.name: collection_name(conversation_id)
            records, chunks = _migrate_collection(
                source,
                source_name=name,
                destination_for_owner=destination,
                batch_size=args.batch_size,
                apply=args.apply,
            )
            report.append({
                "source": str(directory),
                "collection": name,
                "records": records,
                "cloud_chunks": chunks,
            })

    print(json.dumps({
        "mode": "apply" if args.apply else "dry-run",
        "local_data_preserved": True,
        "collections": report,
    }, indent=2))
    if not args.apply:
        print("Dry run only. Local indexes were read from temporary copies and left unchanged.")
        print("Back up your local data, then re-run with --apply to copy and re-embed it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

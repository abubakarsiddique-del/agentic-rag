"""Compare vector-order and cross-encoder retrieval using labeled page pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag import _create_conversation_vector_store
from reranking import rerank_documents


def _relevant_keys(row: dict) -> set[tuple[str, int]]:
    return {
        (str(item["filename"]), int(item["page"]))
        for item in row["relevant_pages"]
    }


def _key(document) -> tuple[str, int]:
    metadata = document.metadata or {}
    return str(metadata.get("source", "")), int(metadata.get("page", -1))


def _metrics(ranked, relevant: set[tuple[str, int]], top_n: int) -> tuple[float, float]:
    top = ranked[:top_n]
    hit = float(any(_key(document) in relevant for document in top))
    reciprocal_rank = next(
        (1.0 / rank for rank, document in enumerate(ranked, start=1) if _key(document) in relevant),
        0.0,
    )
    return hit, reciprocal_rank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation-id", required=True, help="Conversation containing the indexed evaluation document")
    parser.add_argument("--pairs", required=True, type=Path, help="JSON list with question and relevant_pages labels")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    if args.candidates < 1 or args.top_n < 1 or args.top_n > args.candidates:
        parser.error("require candidates >= top-n >= 1")

    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        parser.error("pairs JSON must be a non-empty list")

    vector_store = _create_conversation_vector_store(args.conversation_id)
    totals = {"vector": [0.0, 0.0], "reranked": [0.0, 0.0]}
    for row in pairs:
        question = str(row["question"])
        relevant = _relevant_keys(row)
        candidates = vector_store.similarity_search(question, k=args.candidates)
        reranked = rerank_documents(question, candidates, args.top_n)
        for name, ranked in (
            ("vector", candidates),
            ("reranked", reranked.documents),
        ):
            hit, reciprocal_rank = _metrics(ranked, relevant, args.top_n)
            totals[name][0] += hit
            totals[name][1] += reciprocal_rank

    query_count = len(pairs)
    print(json.dumps({
        name: {
            f"hit@{args.top_n}": round(values[0] / query_count, 4),
            "mrr": round(values[1] / query_count, 4),
        }
        for name, values in totals.items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
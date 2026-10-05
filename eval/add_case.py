"""Add a reviewed production-evaluation case to a local-only JSON dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DATA = Path(__file__).parent / "data" / "production_cases.local.json"


def validate_case(case: dict[str, Any]) -> None:
    required = (
        "id", "question", "conversation_id", "document_ids", "gold_passages",
        "expected_route", "expected_sufficiency", "strata",
    )
    missing = [field for field in required if field not in case]
    if missing:
        raise ValueError(f"case is missing required fields: {', '.join(missing)}")
    if not str(case["question"]).strip() or not case["document_ids"]:
        raise ValueError("question and at least one document ID are required")
    if case["expected_route"] not in {"local", "broad"}:
        raise ValueError("expected_route must be local or broad")
    if not isinstance(case["expected_sufficiency"], bool):
        raise ValueError("expected_sufficiency must be a boolean")
    if case["expected_sufficiency"] and not case["gold_passages"]:
        raise ValueError("answerable cases require at least one gold passage ID")
    if not case["expected_sufficiency"] and case["gold_passages"]:
        raise ValueError("unanswerable cases must not list a gold passage")


def append_case(path: Path, case: dict[str, Any]) -> None:
    validate_case(case)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = payload.get("cases") if isinstance(payload, dict) else payload
        if not isinstance(cases, list):
            raise ValueError("dataset must contain a cases list")
    else:
        cases = []
    if any(existing.get("id") == case["id"] for existing in cases):
        raise ValueError(f"case ID already exists: {case['id']}")
    cases.append(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--id", required=True, help="opaque case ID; do not use a person's name")
    parser.add_argument("--question", required=True)
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--document-id", action="append", required=True, dest="document_ids")
    parser.add_argument("--gold-passage", action="append", default=[], dest="gold_passages")
    parser.add_argument("--expected-route", choices=("local", "broad"), required=True)
    parser.add_argument("--answerable", choices=("yes", "no"), required=True)
    parser.add_argument("--language", required=True, help="language tag, such as en or es")
    parser.add_argument("--document-profile", required=True,
                        help="short descriptor such as short, long, structured, or multilingual")
    args = parser.parse_args(argv)

    answerable = args.answerable == "yes"
    case = {
        "id": args.id,
        "question": args.question,
        "conversation_id": args.conversation_id,
        "document_ids": list(dict.fromkeys(args.document_ids)),
        "gold_passages": list(dict.fromkeys(args.gold_passages)),
        "expected_route": args.expected_route,
        "expected_sufficiency": answerable,
        "strata": {
            "scope": "broad" if args.expected_route == "broad" else "narrow",
            "document_count": "multi" if len(set(args.document_ids)) > 1 else "single",
            "answerability": "answerable" if answerable else "unanswerable",
            "language": args.language,
            "document_profile": args.document_profile,
        },
    }
    try:
        append_case(args.data, case)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Added case {case['id']} to {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
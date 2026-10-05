"""Compare memory-on/off follow-up resolution and hard-fail evidence leakage."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DATA = Path(__file__).parent / "data" / "memory_threads.json"


def _normalized_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").casefold()))


def load_threads(path: Path = DEFAULT_DATA) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    threads = payload.get("threads") if isinstance(payload, dict) else payload
    if not isinstance(threads, list) or not threads:
        raise ValueError("memory fixture must contain a non-empty threads list")
    for thread in threads:
        required = ("id", "history", "follow_up", "expected_standalone_question", "memory_hints", "memory_on", "memory_off")
        if not all(key in thread for key in required):
            raise ValueError("each memory thread is missing a required field")
        if not isinstance(thread["history"], list) or not isinstance(thread["memory_hints"], list):
            raise ValueError(f"thread {thread['id']} history and memory_hints must be lists")
        for mode in ("memory_on", "memory_off"):
            if not isinstance(thread[mode], dict) or not isinstance(thread[mode].get("claims"), list):
                raise ValueError(f"thread {thread['id']} needs {mode} resolution and claims")
    return threads


def _check_evidence_provenance(
    thread: dict[str, Any],
    mode: str,
    outcome: dict[str, Any],
) -> list[dict[str, str]]:
    hint_ids = {str(hint.get("id")) for hint in thread["memory_hints"] if hint.get("id")}
    retrieved = {str(value) for value in outcome.get("retrieved_passage_ids", [])}
    failures = []
    for index, claim in enumerate(outcome["claims"]):
        evidence_ids = {str(value) for value in claim.get("evidence_ids", [])}
        citation_ids = {str(value) for value in claim.get("citation_ids", [])}
        if claim.get("factual", True) and not evidence_ids:
            failures.append({"thread_id": thread["id"], "mode": mode, "reason": f"claim_{index}_has_no_retrieved_evidence"})
        if evidence_ids & hint_ids:
            failures.append({"thread_id": thread["id"], "mode": mode, "reason": f"claim_{index}_uses_memory_hint_as_evidence"})
        if citation_ids & hint_ids:
            failures.append({"thread_id": thread["id"], "mode": mode, "reason": f"claim_{index}_cites_memory_hint"})
        if evidence_ids - retrieved:
            failures.append({"thread_id": thread["id"], "mode": mode, "reason": f"claim_{index}_evidence_not_retrieved"})
        if citation_ids - retrieved:
            failures.append({"thread_id": thread["id"], "mode": mode, "reason": f"claim_{index}_citation_not_retrieved"})
    return failures


def evaluate_memory_threads(threads: list[dict[str, Any]]) -> dict[str, Any]:
    correct = {"memory_on": 0, "memory_off": 0}
    failures = []
    case_results = []
    expected_count = len(threads)
    for thread in threads:
        result = {"thread_id": str(thread["id"]), "resolution": {}}
        expected = _normalized_question(thread["expected_standalone_question"])
        for mode in ("memory_on", "memory_off"):
            outcome = thread[mode]
            resolved = _normalized_question(str(outcome.get("resolved_question", "")))
            is_correct = bool(resolved) and resolved == expected
            correct[mode] += int(is_correct)
            result["resolution"][mode] = is_correct
            failures.extend(_check_evidence_provenance(thread, mode, outcome))
        case_results.append(result)

    accuracy_on = correct["memory_on"] / expected_count if expected_count else None
    accuracy_off = correct["memory_off"] / expected_count if expected_count else None
    return {
        "thread_count": expected_count,
        "resolution_accuracy": {"memory_on": accuracy_on, "memory_off": accuracy_off},
        "memory_resolution_delta": accuracy_on - accuracy_off if accuracy_on is not None and accuracy_off is not None else None,
        "leakage_pass": not failures,
        "leakage_failures": failures,
        "passed": not failures,
        "case_results": case_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_memory_threads(load_threads(args.data))
    except Exception as exc:
        parser.error(f"memory evaluation failed ({type(exc).__name__}); fixture content was not logged")
    print(f"Memory evaluation ({report['thread_count']} threads)")
    print(f"Resolution accuracy: on={report['resolution_accuracy']['memory_on']:.3f}, "
          f"off={report['resolution_accuracy']['memory_off']:.3f}")
    print(f"Memory resolution delta: {report['memory_resolution_delta']:.3f}")
    print(f"Leakage guard: {'PASS' if report['leakage_pass'] else 'FAIL'}")
    for failure in report["leakage_failures"]:
        print(f"  - {failure['thread_id']} ({failure['mode']}): {failure['reason']}")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
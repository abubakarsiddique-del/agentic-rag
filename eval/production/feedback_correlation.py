"""Correlate automated answer scores with persisted thumbs feedback, read-only."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from eval.answer_eval import create_groq_judge, score_answer
from eval.metrics import pearson_correlation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = PROJECT_ROOT / ".rag_history.db"
QUALITY_METRICS = (
    "faithfulness", "citation_precision", "citation_recall",
    "judge_correctness", "judge_completeness", "judge_conciseness",
)


def _load_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def load_rated_turns(database: Path = DEFAULT_DATABASE) -> list[dict[str, Any]]:
    if not database.is_file():
        raise FileNotFoundError("feedback database not found")
    uri = f"file:{database.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT conversation_id, role, content, COALESCE(trace_json, trace) AS trace_value,
                COALESCE(sources_json, sources) AS sources_value, rating, status, rowid
            FROM messages ORDER BY conversation_id, rowid
            """
        ).fetchall()
    finally:
        connection.close()

    turns: list[dict[str, Any]] = []
    latest_question: dict[str, str] = {}
    for row in rows:
        conversation_id = str(row["conversation_id"])
        if row["role"] == "user":
            latest_question[conversation_id] = str(row["content"] or "")
            continue
        rating = row["rating"]
        if row["role"] != "assistant" or rating not in (-1, 1) or row["status"] != "complete":
            continue
        question = latest_question.get(conversation_id)
        if not question:
            continue
        sources = _load_json(row["sources_value"], [])
        trace = _load_json(row["trace_value"], [])
        turns.append({
            "message_id": str(row["rowid"]),
            "question": question,
            "answer": str(row["content"] or ""),
            "sources": sources if isinstance(sources, list) else [],
            "trace": trace if isinstance(trace, list) else [],
            "rating": int(rating),
        })
    return turns


def analyze_feedback(
    turns: list[dict[str, Any]],
    *,
    scorer: Callable[[str, str, list[dict[str, Any]]], dict[str, Any]],
    min_samples: int = 20,
    weak_correlation_threshold: float = 0.2,
) -> dict[str, Any]:
    if not turns:
        return {
            "sample_count": 0,
            "thumbs_up_rate": None,
            "correlations": {},
            "stage_feedback": {},
            "warning": "no rated completed assistant messages; no judge calls were made",
        }

    observations: dict[str, list[tuple[float, float]]] = {metric: [] for metric in QUALITY_METRICS}
    stage_totals: dict[str, list[int]] = {}
    thumbs_up = 0
    for turn in turns:
        positive = float(turn["rating"] == 1)
        thumbs_up += int(positive)
        scores = scorer(turn["question"], turn["answer"], turn["sources"])
        judge_scores = scores.get("judge_scores", {})
        values = {
            "faithfulness": scores.get("faithfulness"),
            "citation_precision": scores.get("citation_precision"),
            "citation_recall": scores.get("citation_recall"),
            "judge_correctness": judge_scores.get("correctness"),
            "judge_completeness": judge_scores.get("completeness"),
            "judge_conciseness": judge_scores.get("conciseness"),
        }
        for metric, value in values.items():
            if isinstance(value, (int, float)):
                observations[metric].append((float(value), positive))
        stages = {
            str(item.get("step"))
            for item in turn.get("trace", [])
            if isinstance(item, dict) and item.get("step")
        }
        for stage in stages:
            counts = stage_totals.setdefault(stage, [0, 0])
            counts[0] += int(positive)
            counts[1] += 1

    correlations = {}
    for metric, pairs in observations.items():
        coefficient = pearson_correlation(
            [item[0] for item in pairs],
            [item[1] for item in pairs],
        )
        insufficient = len(pairs) < min_samples
        predicts = not insufficient and coefficient is not None and coefficient >= weak_correlation_threshold
        correlations[metric] = {
            "n": len(pairs),
            "point_biserial_r": coefficient,
            "predictive": predicts,
            "warning": (
                "insufficient rated samples" if insufficient
                else "weak or unavailable positive correlation with thumbs-up" if not predicts
                else None
            ),
        }
    return {
        "sample_count": len(turns),
        "thumbs_up_rate": thumbs_up / len(turns),
        "correlations": correlations,
        "stage_feedback": {
            stage: {"count": count, "thumbs_up_rate": positives / count}
            for stage, (positives, count) in sorted(stage_totals.items())
        },
        "minimum_samples": min_samples,
        "weak_correlation_threshold": weak_correlation_threshold,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--weak-correlation-threshold", type=float, default=0.2)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    try:
        turns = load_rated_turns(args.database)[-max(0, args.limit):]
    except Exception as exc:
        parser.error(f"feedback could not be read ({type(exc).__name__}); message content was not logged")

    if not turns:
        report = analyze_feedback([], scorer=lambda *_: {})
    else:
        judge = create_groq_judge()

        def scorer(question: str, answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
            return score_answer(question, answer, sources, judge=judge)

        try:
            report = analyze_feedback(
                turns,
                scorer=scorer,
                min_samples=args.min_samples,
                weak_correlation_threshold=args.weak_correlation_threshold,
            )
        except Exception as exc:
            parser.error(f"feedback scoring failed ({type(exc).__name__}); message content was not logged")

    print(f"Feedback correlation ({report['sample_count']} rated answers)")
    if not report["sample_count"]:
        print(report["warning"])
    else:
        for metric, values in report["correlations"].items():
            coefficient = values["point_biserial_r"]
            label = "n/a" if coefficient is None else f"{coefficient:.3f}"
            state = "OK" if values["predictive"] else "FLAG"
            print(f"{metric:<22} r={label:>5}  n={values['n']:>3}  {state}")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
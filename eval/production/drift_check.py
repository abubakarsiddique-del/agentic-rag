"""Run the reviewed variance set and append timestamped snapshots.

Example weekly cron entry (replace the path with this checkout):
0 4 * * 1 cd /path/to/RAG && .venv/bin/python -m eval.production.drift_check --data eval/data/production_cases.local.json --history eval/reports/drift_history.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from eval.production.variance_eval import DEFAULT_DATA, load_cases, run_evaluation
from eval.pipeline_eval import render_variance_report


DEFAULT_HISTORY = Path(__file__).parents[1] / "reports" / "drift_history.jsonl"


def _latest_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def run_drift_check(
    cases: list[dict[str, Any]],
    history_path: Path,
    *,
    evaluation_settings: dict[str, Any] | None = None,
    evaluator: Callable[..., dict[str, Any]] = run_evaluation,
    timestamp: str | None = None,
) -> dict[str, Any]:
    previous = _latest_snapshot(history_path)
    report = evaluator(cases, **(evaluation_settings or {}))
    snapshot = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "settings": evaluation_settings or {},
        "report": report,
        "movement_vs_previous": {},
    }
    previous_metrics = (previous or {}).get("report", {}).get("metrics", {})
    for metric, current in report.get("metrics", {}).items():
        old_mean = previous_metrics.get(metric, {}).get("mean")
        new_mean = current.get("mean")
        if isinstance(old_mean, (int, float)) and isinstance(new_mean, (int, float)):
            snapshot["movement_vs_previous"][metric] = new_mean - old_mean
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rerank-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--rerank-top-n", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--map-reduce-mode", choices=("auto", "off", "force"), default="auto")
    args = parser.parse_args(argv)
    if not args.data.is_file():
        parser.error(f"reviewed case file not found: {args.data}; add reviewed cases with python -m eval.add_case")
    try:
        cases = load_cases(args.data)
        snapshot = run_drift_check(
            cases,
            args.history,
            evaluation_settings={
                "runs": args.runs,
                "k": args.k,
                "top_k": args.top_k,
                "rerank_enabled": args.rerank_enabled,
                "rerank_candidates": args.rerank_candidates,
                "rerank_top_n": args.rerank_top_n,
                "max_retries": args.max_retries,
                "map_reduce_mode": args.map_reduce_mode,
            },
        )
    except Exception as exc:
        parser.error(f"drift check failed ({type(exc).__name__}); case content was not logged")
    print(f"Snapshot {snapshot['timestamp']} saved to {args.history}")
    print(render_variance_report(snapshot["report"]))
    print("Movement vs previous:", json.dumps(snapshot["movement_vs_previous"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
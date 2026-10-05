"""Compare two named RAG settings against the same reviewed question set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from eval.production.variance_eval import DEFAULT_DATA, load_cases, run_evaluation


DEFAULT_CONFIGS = Path(__file__).parents[1] / "data" / "shadow_configs.example.json"
_ALLOWED_SETTINGS = {
    "runs", "k", "top_k", "rerank_enabled", "rerank_candidates", "rerank_top_n",
    "max_retries", "map_reduce_mode", "high_variance_std", "min_answer_agreement",
}


def load_named_settings(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    settings = payload.get(name) if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"named settings not found: {name}")
    unknown = set(settings) - _ALLOWED_SETTINGS
    if unknown:
        raise ValueError(f"unsupported settings: {', '.join(sorted(unknown))}")
    return settings


def compare_reports(
    left_name: str,
    left: dict[str, Any],
    right_name: str,
    right: dict[str, Any],
) -> dict[str, Any]:
    quality = {}
    for metric in sorted(set(left.get("metrics", {})) & set(right.get("metrics", {}))):
        left_value = left["metrics"][metric].get("mean")
        right_value = right["metrics"][metric].get("mean")
        quality[metric] = {
            left_name: left_value,
            right_name: right_value,
            "delta_right_minus_left": (
                right_value - left_value
                if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float))
                else None
            ),
        }

    latency = {}
    left_latency = left.get("latency_ms", {})
    right_latency = right.get("latency_ms", {})
    left_stages = {"end_to_end": left_latency.get("end_to_end", {}), **left_latency.get("stages", {})}
    right_stages = {"end_to_end": right_latency.get("end_to_end", {}), **right_latency.get("stages", {})}
    for stage in sorted(set(left_stages) | set(right_stages)):
        left_stage = left_stages.get(stage, {})
        right_stage = right_stages.get(stage, {})
        latency[stage] = {
            left_name: {key: left_stage.get(key) for key in ("p50_ms", "p95_ms")},
            right_name: {key: right_stage.get(key) for key in ("p50_ms", "p95_ms")},
            "delta_right_minus_left": {
                key: right_stage.get(key) - left_stage.get(key)
                if isinstance(left_stage.get(key), (int, float)) and isinstance(right_stage.get(key), (int, float))
                else None
                for key in ("p50_ms", "p95_ms")
            },
        }

    left_calls = left.get("llm_calls", {}).get("total_per_run", {}).get("mean")
    right_calls = right.get("llm_calls", {}).get("total_per_run", {}).get("mean")
    return {
        "reports": {left_name: left, right_name: right},
        "quality": quality,
        "latency_ms": latency,
        "cost_proxy_llm_calls_per_run": {
            left_name: left_calls,
            right_name: right_calls,
            "delta_right_minus_left": right_calls - left_calls
            if isinstance(left_calls, (int, float)) and isinstance(right_calls, (int, float))
            else None,
        },
    }


def run_shadow_compare(
    cases: list[dict[str, Any]],
    left_name: str,
    left_settings: dict[str, Any],
    right_name: str,
    right_settings: dict[str, Any],
    *,
    evaluator: Callable[..., dict[str, Any]] = run_evaluation,
) -> dict[str, Any]:
    left = evaluator(cases, **left_settings)
    right = evaluator(cases, **right_settings)
    comparison = compare_reports(left_name, left, right_name, right)
    comparison["settings"] = {left_name: left_settings, right_name: right_settings}
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--left", required=True, help="left settings name in the config JSON")
    parser.add_argument("--right", required=True, help="right settings name in the config JSON")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if not args.data.is_file():
        parser.error(f"reviewed case file not found: {args.data}; add reviewed cases with python -m eval.add_case")
    try:
        cases = load_cases(args.data)
        left_settings = load_named_settings(args.configs, args.left)
        right_settings = load_named_settings(args.configs, args.right)
        report = run_shadow_compare(
            cases, args.left, left_settings, args.right, right_settings
        )
    except Exception as exc:
        parser.error(f"shadow comparison failed ({type(exc).__name__}); case content was not logged")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
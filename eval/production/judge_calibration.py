"""Calibrate the existing Groq answer judge against manual 1-5 ratings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from eval.answer_eval import (
    DIMENSIONS,
    confirm_live_call_count,
    create_groq_judge,
    create_stub_judge,
    score_answer,
)
from eval.metrics import spearman_correlation
from eval.production.variance_eval import run_repeated_cases, summarize


DEFAULT_CASES = Path(__file__).parents[1] / "data" / "calibration_cases.local.json"
DEFAULT_RATINGS = Path(__file__).parents[1] / "data" / "human_ratings.local.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not 20 <= len(cases) <= 30:
        raise ValueError("judge calibration requires 20-30 cases")
    seen: set[str] = set()
    for case in cases:
        if not all(key in case for key in ("id", "question", "answer", "sources")):
            raise ValueError("each calibration case needs id, question, answer, and sources")
        case_id = str(case["id"])
        if case_id in seen:
            raise ValueError(f"duplicate calibration case ID: {case_id}")
        seen.add(case_id)
    return cases


def rating_template(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ratings": [
            {"case_id": str(case["id"]), **{dimension: None for dimension in DIMENSIONS}}
            for case in cases
        ]
    }


def load_ratings(path: Path, cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("ratings") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("human ratings file must contain a ratings list")
    ratings: dict[str, dict[str, int]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id in ratings:
            raise ValueError(f"duplicate human rating for case: {case_id}")
        values = {}
        for dimension in DIMENSIONS:
            score = row.get(dimension)
            if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
                raise ValueError(f"{case_id}: {dimension} must be an integer from 1 to 5")
            values[dimension] = score
        ratings[case_id] = values
    expected_ids = {str(case["id"]) for case in cases}
    if set(ratings) != expected_ids:
        raise ValueError("human ratings must cover exactly the case IDs in the calibration dataset")
    return ratings


def run_calibration(
    cases: list[dict[str, Any]],
    ratings: dict[str, dict[str, int]],
    *,
    judge: Callable[[dict[str, Any]], dict[str, Any]],
    runs: int = 1,
    mode: str = "STUB",
    weak_agreement_threshold: float = 0.3,
) -> dict[str, Any]:
    if not 20 <= len(cases) <= 30:
        raise ValueError("judge calibration requires 20-30 cases")
    if runs < 1:
        raise ValueError("runs must be at least 1")
    for case in cases:
        case_id = str(case["id"])
        if case_id not in ratings:
            raise ValueError(f"human rating missing for case: {case_id}")
    case_scores_by_run = run_repeated_cases(
        cases,
        runs,
        lambda case: {
            "case_id": str(case["id"]),
            "judge": score_answer(case["question"], case["answer"], case["sources"], judge=judge)["judge_scores"],
            "human": ratings[str(case["id"])],
        },
    )

    per_run_agreement: dict[str, list[float | None]] = {dimension: [] for dimension in DIMENSIONS}
    for run_scores in case_scores_by_run:
        for dimension in DIMENSIONS:
            correlation = spearman_correlation(
                [float(item["judge"][dimension]) for item in run_scores],
                [float(item["human"][dimension]) for item in run_scores],
            )
            per_run_agreement[dimension].append(correlation)

    agreement = {}
    for dimension, values in per_run_agreement.items():
        summary = summarize(values)
        mean = summary["mean"]
        weak = mean is None or mean < weak_agreement_threshold
        agreement[dimension] = {
            "n": len(cases),
            "per_run_spearman_rho": values,
            "spearman_rho": mean,
            "mean": mean,
            "std_dev": summary["std_dev"],
            "min": summary["min"],
            "max": summary["max"],
            "weak": weak,
            "warning": "calibration agreement is weak; do not trust this judge dimension" if weak else None,
        }
    return {
        "mode": mode.upper(),
        "case_count": len(cases),
        "runs": runs,
        "agreement": agreement,
        "case_scores_by_run": case_scores_by_run,
        "weak_agreement_threshold": weak_agreement_threshold,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--ratings-template-out", type=Path,
                        help="write blank 1-5 ratings for these cases and exit without calling the judge")
    parser.add_argument("--live", action="store_true", help="opt into real Groq judge calls")
    parser.add_argument("--yes", action="store_true", help="confirm the displayed live call count without prompting")
    parser.add_argument("--runs", type=int, default=3, help="judge repeats per case (default: 3)")
    parser.add_argument("--weak-agreement-threshold", type=float, default=0.3)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if args.yes and not args.live:
        parser.error("--yes is only valid with --live")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if not args.cases.is_file():
        parser.error(f"case file not found: {args.cases}; provide 20-30 reviewed question/answer/source cases")
    try:
        cases = load_cases(args.cases)
        if args.ratings_template_out:
            args.ratings_template_out.parent.mkdir(parents=True, exist_ok=True)
            args.ratings_template_out.write_text(
                json.dumps(rating_template(cases), indent=2) + "\n", encoding="utf-8"
            )
            print(f"Wrote ratings template for {len(cases)} cases to {args.ratings_template_out}")
            return 0
        if not args.ratings.is_file():
            parser.error(
                f"human ratings not found: {args.ratings}; first use --ratings-template-out, "
                "then fill in 1-5 ratings"
            )
        ratings = load_ratings(args.ratings, cases)
        expected_calls = len(cases) * args.runs if args.live else 0
        if args.live and not confirm_live_call_count(expected_calls, yes=args.yes):
            parser.error("live calibration cancelled; no Groq calls were made")
        judge = create_groq_judge() if args.live else create_stub_judge()
        result = run_calibration(
            cases,
            ratings,
            judge=judge,
            runs=args.runs,
            mode="LIVE" if args.live else "STUB",
            weak_agreement_threshold=args.weak_agreement_threshold,
        )
        result["expected_groq_calls"] = expected_calls
        result["groq_call_attempts"] = getattr(judge, "call_count", 0) if args.live else 0
    except Exception as exc:
        reason = str(exc) if "rate limit" in str(exc).casefold() else type(exc).__name__
        parser.error(f"calibration could not run ({reason}); case content was not logged")

    print(f"[{result['mode']}] Judge calibration ({result['case_count']} cases x {result['runs']} runs; Spearman rank correlation)")
    if args.live:
        print(f"Groq calls: {result['groq_call_attempts']} attempts (estimate {result['expected_groq_calls']}; retries may increase this)")
    for dimension, values in result["agreement"].items():
        state = "WEAK" if values["weak"] else "OK"
        rho = "n/a" if values["spearman_rho"] is None else f"{values['spearman_rho']:.3f}"
        std_dev = "n/a" if values["std_dev"] is None else f"{values['std_dev']:.3f}"
        print(f"{dimension:<14} mean rho={rho:>5} +/- {std_dev:<5} runs={values['per_run_spearman_rho']}  {state}")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
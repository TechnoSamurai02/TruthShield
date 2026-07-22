from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


AUTHENTIC_LABELS = {"authentic", "real", "real_camera"}
GENERATED_LABELS = {"generated", "ai_generated"}
MANIPULATED_LABELS = {"manipulated", "ai_manipulated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a manipulation score and localized-area rule on tuning predictions "
            "under fixed precision and false-warning constraints."
        )
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    parser.add_argument("--authentic-false-warning-limit", type=float, default=0.01)
    parser.add_argument("--generated-false-warning-limit", type=float, default=0.01)
    parser.add_argument("--max-view-range", type=float, default=0.18)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        values = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for value in values:
        label = str(value.get("label") or "").strip().lower()
        try:
            score = float(value["manipulation_score"])
            area = float(value["predicted_region_area_ratio"])
            view_range = float(value["view_score_range"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(
            {
                "label": label,
                "score": score,
                "area": area,
                "view_range": view_range,
                "localized_support": _boolean(value.get("localized_or_persistent_support")),
                "stable": _boolean(value.get("stable_across_views")),
            }
        )
    return rows


def optimize(
    rows: list[dict[str, Any]],
    *,
    minimum_precision: float,
    authentic_false_warning_limit: float,
    generated_false_warning_limit: float,
    max_view_range: float,
) -> dict[str, Any]:
    counts = {
        "authentic": sum(row["label"] in AUTHENTIC_LABELS for row in rows),
        "generated": sum(row["label"] in GENERATED_LABELS for row in rows),
        "manipulated": sum(row["label"] in MANIPULATED_LABELS for row in rows),
    }
    if not all(counts.values()):
        raise ValueError("Optimization requires authentic, generated, and manipulated tuning rows.")
    area_candidates = sorted({0.0, *(float(row["area"]) for row in rows)})
    best: dict[str, Any] | None = None
    feasible_rules = 0
    for area_threshold in area_candidates:
        eligible = [
            row
            for row in rows
            if row["localized_support"]
            and row["stable"]
            and float(row["view_range"]) <= max_view_range
            and float(row["area"]) >= area_threshold
        ]
        eligible.sort(key=lambda row: float(row["score"]), reverse=True)
        true_manipulated = false_authentic = false_generated = 0
        for index, row in enumerate(eligible):
            label = row["label"]
            true_manipulated += int(label in MANIPULATED_LABELS)
            false_authentic += int(label in AUTHENTIC_LABELS)
            false_generated += int(label in GENERATED_LABELS)
            next_score = float(eligible[index + 1]["score"]) if index + 1 < len(eligible) else None
            threshold = float(row["score"])
            if next_score is not None and next_score == threshold:
                continue
            predicted = index + 1
            precision = true_manipulated / predicted
            authentic_rate = false_authentic / counts["authentic"]
            generated_rate = false_generated / counts["generated"]
            if (
                precision < minimum_precision
                or authentic_rate > authentic_false_warning_limit
                or generated_rate > generated_false_warning_limit
            ):
                continue
            feasible_rules += 1
            candidate = {
                "manipulation_score_threshold": round(threshold, 8),
                "minimum_localized_area_ratio": round(area_threshold, 8),
                "max_view_score_range": max_view_range,
                "predicted_manipulated": predicted,
                "true_manipulated": true_manipulated,
                "false_manipulation_warnings": false_authentic + false_generated,
                "precision": precision,
                "recall": true_manipulated / counts["manipulated"],
                "authentic_false_warning_rate": authentic_rate,
                "generated_false_manipulation_rate": generated_rate,
            }
            if best is None or _rank(candidate) > _rank(best):
                best = candidate
    return {
        "record_count": len(rows),
        "class_counts": counts,
        "constraints": {
            "minimum_precision": minimum_precision,
            "authentic_false_warning_limit": authentic_false_warning_limit,
            "generated_false_warning_limit": generated_false_warning_limit,
            "max_view_score_range": max_view_range,
        },
        "feasible_rule_count": feasible_rules,
        "best_rule": best,
        "status": "candidate_requires_calibration" if best else "no_rule_meets_constraints",
        "warning": (
            "This rule was selected on tuning data. Freeze it, evaluate it on a separate "
            "calibration split, and do not promote it from this report alone."
        ),
    }


def _rank(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(candidate["recall"]),
        float(candidate["precision"]),
        -float(candidate["false_manipulation_warnings"]),
        -float(candidate["minimum_localized_area_ratio"]),
    )


def _boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    args = parse_args()
    report = optimize(
        read_rows(args.predictions),
        minimum_precision=args.minimum_precision,
        authentic_false_warning_limit=args.authentic_false_warning_limit,
        generated_false_warning_limit=args.generated_false_warning_limit,
        max_view_range=args.max_view_range,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["best_rule"] is None:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

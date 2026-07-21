from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


POSITIVE_GENERATION_LABELS = {"generated", "ai_generated", "likely_ai_generated"}
POSITIVE_MANIPULATION_LABELS = {"manipulated", "ai_manipulated", "likely_ai_manipulated"}
AUTHENTIC_LABELS = {"authentic", "real", "real_camera", "likely_authentic"}


def read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                **row,
                "label": str(row.get("label") or "").strip().lower(),
                "generation_score": _probability(row.get("generation_score")),
                "manipulation_score": _probability(row.get("manipulation_score")),
            }
        )
    return [row for row in parsed if row["label"] and row["generation_score"] is not None]


def calibrate(
    rows: list[dict[str, Any]],
    *,
    media_type: str,
    minimum_precision: float,
    authentic_false_warning_limit: float,
    false_authentic_limit: float,
) -> dict[str, Any]:
    authentic = [row for row in rows if row["label"] in AUTHENTIC_LABELS]
    generated = [row for row in rows if row["label"] in POSITIVE_GENERATION_LABELS]
    manipulated = [row for row in rows if row["label"] in POSITIVE_MANIPULATION_LABELS]
    if not authentic or not generated:
        raise ValueError("Calibration requires authentic and generated examples.")

    generation_upper = _upper_threshold(
        rows,
        score_key="generation_score",
        positive_labels=POSITIVE_GENERATION_LABELS,
        authentic_count=len(authentic),
        minimum_precision=minimum_precision,
        false_warning_limit=authentic_false_warning_limit,
    )
    manipulation_rows = [row for row in rows if row.get("manipulation_score") is not None]
    manipulation_upper = _upper_threshold(
        manipulation_rows,
        score_key="manipulation_score",
        positive_labels=POSITIVE_MANIPULATION_LABELS,
        authentic_count=len(authentic),
        minimum_precision=minimum_precision,
        false_warning_limit=authentic_false_warning_limit,
    ) if manipulated else None
    generation_lower, manipulation_lower = _lower_thresholds(
        rows,
        false_authentic_limit=false_authentic_limit,
    )
    return {
        "policy_version": "truthshield-media-policy-v4.0.0",
        "calibration_id": f"{media_type}-calibration-v4",
        "calibration_status": "calibrated_on_generator_isolated_split",
        "source_population": f"{len(rows)} {media_type} calibration records",
        "generation": {
            "lower_threshold": generation_lower,
            "upper_threshold": generation_upper if generation_upper is not None else 1.0,
            "enabled": generation_upper is not None,
            "minimum_decisive_precision": minimum_precision,
            "max_authentic_false_warning_rate": authentic_false_warning_limit,
        },
        "manipulation": {
            "lower_threshold": manipulation_lower,
            "upper_threshold": manipulation_upper if manipulation_upper is not None else 1.0,
            "enabled": manipulation_upper is not None,
            "minimum_decisive_precision": minimum_precision,
            "max_authentic_false_warning_rate": authentic_false_warning_limit,
            "require_localized_or_persistent_support": True,
        },
        "stability": {"max_view_score_range": 0.18, "max_window_score_range": 0.30},
        "quality": {"minimum_short_side": 224, "minimum_video_frames": 8},
        "requirements": {
            "generation_for_decisive_verdict": True,
            "manipulation_for_authentic_verdict": True,
        },
        "calibration_counts": {
            "authentic": len(authentic),
            "generated": len(generated),
            "manipulated": len(manipulated),
        },
    }


def _upper_threshold(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    positive_labels: set[str],
    authentic_count: int,
    minimum_precision: float,
    false_warning_limit: float,
) -> float | None:
    candidates = sorted({float(row[score_key]) for row in rows if row.get(score_key) is not None})
    valid: list[tuple[int, float]] = []
    for threshold in candidates:
        predicted = [row for row in rows if float(row[score_key]) >= threshold]
        if not predicted:
            continue
        true_positive = sum(row["label"] in positive_labels for row in predicted)
        false_authentic = sum(row["label"] in AUTHENTIC_LABELS for row in predicted)
        precision = true_positive / len(predicted)
        false_warning_rate = false_authentic / max(1, authentic_count)
        if precision >= minimum_precision and false_warning_rate <= false_warning_limit:
            valid.append((true_positive, threshold))
    return min(valid, key=lambda item: (-item[0], item[1]))[1] if valid else None


def _lower_thresholds(
    rows: list[dict[str, Any]],
    *,
    false_authentic_limit: float,
) -> tuple[float, float]:
    synthetic = [
        row for row in rows
        if row["label"] in POSITIVE_GENERATION_LABELS | POSITIVE_MANIPULATION_LABELS
    ]
    generation_candidates = sorted({float(row["generation_score"]) for row in rows})
    manipulation_candidates = sorted(
        {float(row["manipulation_score"]) for row in rows if row.get("manipulation_score") is not None}
    ) or [0.0]
    best = (0.0, 0.0, -1)
    for generation_lower in generation_candidates:
        for manipulation_lower in manipulation_candidates:
            false_authentic = sum(
                float(row["generation_score"]) <= generation_lower
                and row.get("manipulation_score") is not None
                and float(row["manipulation_score"]) <= manipulation_lower
                for row in synthetic
            )
            rate = false_authentic / max(1, len(synthetic))
            coverage = sum(
                float(row["generation_score"]) <= generation_lower
                and row.get("manipulation_score") is not None
                and float(row["manipulation_score"]) <= manipulation_lower
                for row in rows
            )
            if rate <= false_authentic_limit and coverage > best[2]:
                best = (generation_lower, manipulation_lower, coverage)
    return round(best[0], 6), round(best[1], 6)


def _probability(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit TruthShield v4 abstention thresholds on calibration predictions.")
    parser.add_argument("predictions", type=Path, help="CSV with label, generation_score, manipulation_score.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-type", choices=("image", "video"), required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    parser.add_argument("--authentic-false-warning-limit", type=float)
    parser.add_argument("--false-authentic-limit", type=float)
    args = parser.parse_args()
    false_warning = args.authentic_false_warning_limit
    if false_warning is None:
        false_warning = 0.01 if args.media_type == "image" else 0.02
    false_authentic = args.false_authentic_limit
    if false_authentic is None:
        false_authentic = 0.02 if args.media_type == "image" else 0.05
    policy = calibrate(
        read_predictions(args.predictions),
        media_type=args.media_type,
        minimum_precision=args.minimum_precision,
        authentic_false_warning_limit=false_warning,
        false_authentic_limit=false_authentic,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(policy, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

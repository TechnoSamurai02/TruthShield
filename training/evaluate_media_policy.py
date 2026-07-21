from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.calibrate_media_policy import (
    AUTHENTIC_LABELS,
    POSITIVE_GENERATION_LABELS,
    POSITIVE_MANIPULATION_LABELS,
    read_predictions,
)


def evaluate(rows: list[dict[str, Any]], policy: dict[str, Any], media_type: str, bootstrap_samples: int) -> dict[str, Any]:
    classified = [_classify(row, policy) for row in rows]
    decisive = [item for item in classified if item["verdict"] != "inconclusive"]
    total = len(classified)
    counts = Counter((item["label"], item["verdict"]) for item in classified)
    per_class = {}
    for label in ("authentic", "generated", "manipulated"):
        population = [item for item in classified if item["normalized_label"] == label]
        expected_verdict = _expected_verdict(label)
        correct = sum(item["verdict"] == expected_verdict for item in population)
        predicted = sum(item["verdict"] == expected_verdict for item in classified)
        negatives = [item for item in classified if item["normalized_label"] != label]
        false_positive = sum(item["verdict"] == expected_verdict for item in negatives)
        per_class[label] = {
            "count": len(population),
            "precision": correct / max(1, predicted),
            "recall": correct / max(1, len(population)),
            "specificity": 1.0 - false_positive / max(1, len(negatives)),
            "decisive_coverage": sum(item["verdict"] != "inconclusive" for item in population) / max(1, len(population)),
        }
    ai_verdicts = [item for item in classified if item["verdict"] in {"likely_ai_generated", "likely_ai_manipulated"}]
    ai_correct = sum(
        (item["verdict"] == "likely_ai_generated" and item["normalized_label"] == "generated")
        or (item["verdict"] == "likely_ai_manipulated" and item["normalized_label"] == "manipulated")
        for item in ai_verdicts
    )
    authentic = [item for item in classified if item["normalized_label"] == "authentic"]
    synthetic = [item for item in classified if item["normalized_label"] in {"generated", "manipulated"}]
    generated_verdicts = [item for item in classified if item["verdict"] == "likely_ai_generated"]
    manipulation_verdicts = [item for item in classified if item["verdict"] == "likely_ai_manipulated"]
    metrics = {
        "decisive_coverage": len(decisive) / max(1, total),
        "decisive_ai_precision": ai_correct / max(1, len(ai_verdicts)),
        "generated_verdict_precision": sum(item["normalized_label"] == "generated" for item in generated_verdicts) / max(1, len(generated_verdicts)),
        "manipulated_verdict_precision": sum(item["normalized_label"] == "manipulated" for item in manipulation_verdicts) / max(1, len(manipulation_verdicts)),
        "authentic_false_ai_rate": sum(item["verdict"].startswith("likely_ai_") for item in authentic) / max(1, len(authentic)),
        "synthetic_false_authentic_rate": sum(item["verdict"] == "likely_authentic" for item in synthetic) / max(1, len(synthetic)),
        "generation_roc_auc": _auc(rows, "generation_score", POSITIVE_GENERATION_LABELS),
        "manipulation_roc_auc": _auc(rows, "manipulation_score", POSITIVE_MANIPULATION_LABELS),
        "generation_calibration_error": _ece(rows, "generation_score", POSITIVE_GENERATION_LABELS),
        "manipulation_calibration_error": _ece(rows, "manipulation_score", POSITIVE_MANIPULATION_LABELS),
    }
    metrics["bootstrap_95_percent_intervals"] = _bootstrap(classified, bootstrap_samples)
    hard_real_categories = _group_rates(
        [item for item in classified if item["normalized_label"] == "authentic"],
        "semantic_category",
        lambda item: item["verdict"].startswith("likely_ai_"),
    )
    transformations = _group_rates(
        classified,
        "transformation",
        lambda item: (
            item["normalized_label"] == "authentic" and item["verdict"].startswith("likely_ai_")
        ) or (
            item["normalized_label"] in {"generated", "manipulated"} and item["verdict"] == "likely_authentic"
        ),
        include_label=True,
    )
    gates = _promotion_gates(
        metrics,
        media_type,
        has_manipulation_examples=bool(per_class["manipulated"]["count"]),
        hard_real_categories=hard_real_categories,
        transformations=transformations,
    )
    per_generator: dict[str, dict[str, float | int]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in classified:
        grouped[
            f"{item['normalized_label']}:{str(item.get('generator_or_editor') or 'unknown')}"
        ].append(item)
    for generator, items in sorted(grouped.items()):
        expected = _expected_verdict(items[0]["normalized_label"])
        item_ids = {id(item) for item in items}
        outside = [item for item in classified if id(item) not in item_ids]
        true_positive = sum(item["verdict"] == expected for item in items)
        false_positive = sum(item["verdict"] == expected for item in outside)
        per_generator[generator] = {
            "count": len(items),
            "decisive_coverage": sum(item["verdict"] != "inconclusive" for item in items) / len(items),
            "precision": true_positive / max(1, true_positive + false_positive),
            "recall": true_positive / len(items),
            "specificity": 1.0 - false_positive / max(1, len(outside)),
            "decisive_accuracy": sum(
                item["verdict"] == _expected_verdict(item["normalized_label"])
                for item in items if item["verdict"] != "inconclusive"
            ) / max(1, sum(item["verdict"] != "inconclusive" for item in items)),
        }
    return {
        "media_type": media_type,
        "record_count": total,
        "metrics": metrics,
        "per_class": per_class,
        "per_generator": per_generator,
        "per_hard_real_category": hard_real_categories,
        "per_transformation": transformations,
        "confusion_counts": {f"{label}->{verdict}": count for (label, verdict), count in sorted(counts.items())},
        "promotion_gates": gates,
        "promoted": all(gates.values()),
    }


def _classify(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    generation = float(row["generation_score"])
    manipulation = row.get("manipulation_score")
    manipulation = float(manipulation) if manipulation is not None else None
    generation_cfg = policy["generation"]
    manipulation_cfg = policy["manipulation"]
    if generation_cfg.get("enabled", True) and generation >= float(generation_cfg["upper_threshold"]):
        verdict = "likely_ai_generated"
    elif (
        manipulation is not None
        and manipulation_cfg.get("enabled", True)
        and generation < float(generation_cfg["upper_threshold"])
        and manipulation >= float(manipulation_cfg["upper_threshold"])
        and str(row.get("localized_or_persistent_support", "true")).lower() in {"1", "true", "yes"}
    ):
        verdict = "likely_ai_manipulated"
    elif (
        manipulation is not None
        and generation <= float(generation_cfg["lower_threshold"])
        and manipulation <= float(manipulation_cfg["lower_threshold"])
    ):
        verdict = "likely_authentic"
    else:
        verdict = "inconclusive"
    return {**row, "normalized_label": _normalized_label(row["label"]), "verdict": verdict}


def _normalized_label(label: str) -> str:
    if label in AUTHENTIC_LABELS:
        return "authentic"
    if label in POSITIVE_MANIPULATION_LABELS:
        return "manipulated"
    return "generated"


def _expected_verdict(label: str) -> str:
    return {
        "authentic": "likely_authentic",
        "generated": "likely_ai_generated",
        "manipulated": "likely_ai_manipulated",
    }[label]


def _auc(rows: list[dict[str, Any]], score_key: str, positive_labels: set[str]) -> float | None:
    usable = [(float(row[score_key]), row["label"] in positive_labels) for row in rows if row.get(score_key) is not None]
    positive = [item for item in usable if item[1]]
    negative = [item for item in usable if not item[1]]
    if not positive or not negative:
        return None
    wins = sum((p[0] > n[0]) + 0.5 * (p[0] == n[0]) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def _ece(rows: list[dict[str, Any]], score_key: str, positive_labels: set[str], bins: int = 10) -> float | None:
    usable = [(float(row[score_key]), row["label"] in positive_labels) for row in rows if row.get(score_key) is not None]
    if not usable:
        return None
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        bucket = [item for item in usable if lower <= item[0] <= upper if index == bins - 1 or item[0] < upper]
        if bucket:
            confidence = sum(item[0] for item in bucket) / len(bucket)
            accuracy = sum(item[1] for item in bucket) / len(bucket)
            error += len(bucket) / len(usable) * abs(confidence - accuracy)
    return error


def _bootstrap(rows: list[dict[str, Any]], samples: int) -> dict[str, list[float]]:
    if not rows or samples <= 0:
        return {}
    randomizer = random.Random(404)
    coverage: list[float] = []
    authentic_false_ai: list[float] = []
    for _ in range(samples):
        sample = [rows[randomizer.randrange(len(rows))] for _ in rows]
        coverage.append(sum(item["verdict"] != "inconclusive" for item in sample) / len(sample))
        authentic = [item for item in sample if item["normalized_label"] == "authentic"]
        authentic_false_ai.append(
            sum(item["verdict"].startswith("likely_ai_") for item in authentic) / max(1, len(authentic))
        )
    return {
        "decisive_coverage": [_quantile(coverage, 0.025), _quantile(coverage, 0.975)],
        "authentic_false_ai_rate": [_quantile(authentic_false_ai, 0.025), _quantile(authentic_false_ai, 0.975)],
    }


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def _group_rates(
    rows: list[dict[str, Any]],
    key: str,
    is_error: Any,
    include_label: bool = False,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_name = str(row.get(key) or "unspecified")
        if include_label:
            group_name = f"{row['normalized_label']}:{group_name}"
        grouped[group_name].append(row)
    return {
        name: {"count": len(items), "error_rate": sum(bool(is_error(item)) for item in items) / len(items)}
        for name, items in sorted(grouped.items())
    }


def _promotion_gates(
    metrics: dict[str, Any],
    media_type: str,
    *,
    has_manipulation_examples: bool,
    hard_real_categories: dict[str, dict[str, float | int]],
    transformations: dict[str, dict[str, float | int]],
) -> dict[str, bool]:
    false_warning_limit = 0.01 if media_type == "image" else 0.02
    synthetic_limit = 0.02 if media_type == "image" else 0.05
    return {
        "generated_verdict_precision_at_least_95_percent": metrics["generated_verdict_precision"] >= 0.95,
        "manipulated_verdict_precision_at_least_95_percent": (
            not has_manipulation_examples or metrics["manipulated_verdict_precision"] >= 0.95
        ),
        "authentic_false_ai_rate_within_limit": metrics["authentic_false_ai_rate"] <= false_warning_limit,
        "synthetic_false_authentic_rate_within_limit": metrics["synthetic_false_authentic_rate"] <= synthetic_limit,
        "decisive_coverage_within_limit": metrics["decisive_coverage"] >= (0.70 if media_type == "image" else 0.60),
        "no_hard_real_category_regression": all(
            float(value["error_rate"]) <= false_warning_limit for value in hard_real_categories.values()
        ),
        "no_supported_transformation_regression": all(
            float(value["error_rate"]) <= (false_warning_limit if name.startswith("authentic:") else synthetic_limit)
            for name, value in transformations.items()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TruthShield v4 policy on a locked prediction file.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--media-type", choices=("image", "video"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    report = evaluate(read_predictions(args.predictions), policy, args.media_type, args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["promoted"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(REPO_ROOT))

from training.evaluate_image_detector import (  # noqa: E402
    _collect_samples,
    _exclude_audit_leakage,
    _sample_per_class,
)


LABEL_TO_ID = {
    "ai_generated": 0,
    "real_camera": 1,
    "real_edited_or_captioned": 2,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the complete production image-analysis path.")
    parser.add_argument("--data-dir", default="training/data/defactify_sample")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", default="training/evaluation/full_pipeline_report_test_300.json")
    parser.add_argument("--audit-report", default="training/data/image_dataset_audit.json")
    return parser.parse_args()


def main() -> None:
    # A benchmark must not depend on paid provider availability. The resulting
    # report still exercises and records the production neutral fallback paths.
    os.environ["BRAVE_SEARCH_API_KEY"] = ""
    os.environ["GOOGLE_VISION_API_KEY"] = ""
    os.environ["GOOGLE_CLOUD_VISION_API_KEY"] = ""
    os.environ["ENABLE_LOCAL_AI_MODELS"] = "true"

    from analyzers.image_analyzer import analyze_image_bytes
    from models.schemas import AnalysisResponse

    args = parse_args()
    data_dir = Path(args.data_dir)
    split_dir = data_dir / args.split
    samples = _collect_samples(split_dir, LABEL_TO_ID)
    samples, excluded = _exclude_audit_leakage(
        samples,
        data_dir=data_dir,
        split=args.split,
        audit_path=Path(args.audit_report),
    )
    samples = _sample_per_class(samples, max_per_class=max(1, args.max_per_class), seed=args.seed)

    outcomes: Counter[str] = Counter()
    per_class: Dict[str, Counter[str]] = defaultdict(Counter)
    model_statuses: Counter[str] = Counter()
    provenance_statuses: Counter[str] = Counter()
    web_statuses: Counter[str] = Counter()
    legacy_likely_ai: Counter[str] = Counter()
    examples: List[Dict[str, Any]] = []

    for index, (path, label_id) in enumerate(samples, start=1):
        result = analyze_image_bytes(path.read_bytes(), path.name)
        AnalysisResponse(**result)
        expected = ID_TO_LABEL[label_id]
        assessment = result.get("assessment") or {}
        verdict = str(assessment.get("verdict") or "inconclusive")
        detector_score = assessment.get("detector_score")
        if isinstance(detector_score, (int, float)) and float(detector_score) >= 0.70:
            legacy_likely_ai["true_ai" if expected == "ai_generated" else "false_ai_alarm"] += 1
        outcomes[verdict] += 1
        per_class[expected][verdict] += 1
        summary = (result.get("technical_details") or {}).get("ai_detector_summary") or {}
        model_statuses["available" if summary.get("learned_model_available") else "unavailable"] += 1
        provenance_statuses[str((result.get("provenance") or {}).get("status") or "none")] += 1
        web_statuses[str((result.get("web_research") or {}).get("status") or "none")] += 1
        if len(examples) < 40 and _is_error(expected, verdict):
            examples.append(
                {
                    "path": str(path),
                    "expected": expected,
                    "verdict": verdict,
                    "detector_score": assessment.get("detector_score"),
                    "reason": assessment.get("reason"),
                }
            )
        if index % 25 == 0 or index == len(samples):
            print(f"  {index}/{len(samples)}", flush=True)

    ai_count = sum(per_class["ai_generated"].values())
    non_ai_count = len(samples) - ai_count
    true_ai = per_class["ai_generated"]["likely_ai_generated_or_manipulated"]
    false_ai = sum(
        per_class[label]["likely_ai_generated_or_manipulated"]
        for label in ("real_camera", "real_edited_or_captioned")
    )
    ai_as_authentic = per_class["ai_generated"]["likely_authentic"]
    true_authentic = sum(
        per_class[label]["likely_authentic"]
        for label in ("real_camera", "real_edited_or_captioned")
    )
    inconclusive = outcomes["inconclusive"]
    decisive = len(samples) - inconclusive
    report = {
        "data_split": str(split_dir.resolve()),
        "sample_count": len(samples),
        "sampling": {"max_per_class": args.max_per_class, "seed": args.seed},
        "audit_leakage_excluded_count": len(excluded),
        "outcomes": dict(outcomes),
        "per_class_outcomes": {label: dict(counts) for label, counts in per_class.items()},
        "metrics": {
            "false_positive_rate": round(false_ai / max(1, non_ai_count), 6),
            "false_negative_rate_as_authentic": round(ai_as_authentic / max(1, ai_count), 6),
            "ai_recall": round(true_ai / max(1, ai_count), 6),
            "authentic_recall": round(true_authentic / max(1, non_ai_count), 6),
            "inconclusive_rate": round(inconclusive / max(1, len(samples)), 6),
            "decisive_coverage": round(decisive / max(1, len(samples)), 6),
            "decisive_accuracy": round((true_ai + true_authentic) / max(1, decisive), 6),
            "likely_ai_precision": round(true_ai / max(1, true_ai + false_ai), 6),
            "likely_authentic_precision": round(true_authentic / max(1, true_authentic + ai_as_authentic), 6),
        },
        "legacy_frontend_likely_ai_decision": {
            "threshold": 0.70,
            "true_ai": legacy_likely_ai["true_ai"],
            "false_ai_alarm": legacy_likely_ai["false_ai_alarm"],
            "ai_recall": round(legacy_likely_ai["true_ai"] / max(1, ai_count), 6),
            "false_positive_rate": round(legacy_likely_ai["false_ai_alarm"] / max(1, non_ai_count), 6),
        },
        "model_availability": dict(model_statuses),
        "provenance_statuses": dict(provenance_statuses),
        "web_statuses": dict(web_statuses),
        "error_examples": examples,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"outcomes": report["outcomes"], "metrics": report["metrics"]}, indent=2))
    print(f"Full report: {output_path.resolve()}", flush=True)


def _is_error(expected: str, verdict: str) -> bool:
    if expected == "ai_generated":
        return verdict == "likely_authentic"
    return verdict == "likely_ai_generated_or_manipulated"


if __name__ == "__main__":
    main()

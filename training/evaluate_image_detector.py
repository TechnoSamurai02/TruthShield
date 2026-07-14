from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analyzers.image_decision import AI_DETECTOR_MIN, AUTHENTIC_DETECTOR_MAX  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TruthShield image model on an untouched split.")
    parser.add_argument("--model-dir", default="training/models/truthshield-image-detector-v2")
    parser.add_argument("--data-dir", default="training/data/defactify_sample")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Deterministically sample at most this many images from each class (preferred for balanced reports).",
    )
    parser.add_argument("--max-misclassified", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    parser.add_argument("--audit-report", default="training/data/image_dataset_audit.json")
    parser.add_argument("--include-audit-leakage", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    args = parse_args()
    model_dir = Path(args.model_dir)
    split_dir = Path(args.data_dir) / args.split
    if not model_dir.exists():
        raise SystemExit(f"Model folder not found: {model_dir}")
    if not split_dir.exists():
        raise SystemExit(f"Dataset split not found: {split_dir}")

    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = AutoModelForImageClassification.from_pretrained(str(model_dir))
    model.eval()
    label2id = {str(label): int(index) for label, index in model.config.label2id.items()}
    id2label = {int(index): str(label) for index, label in model.config.id2label.items()}
    samples = _collect_samples(split_dir, label2id)
    excluded_paths: List[str] = []
    if not args.include_audit_leakage:
        samples, excluded_paths = _exclude_audit_leakage(
            samples,
            data_dir=Path(args.data_dir),
            split=args.split,
            audit_path=Path(args.audit_report),
        )
    if args.max_per_class > 0:
        samples = _sample_per_class(samples, max_per_class=args.max_per_class, seed=args.seed)
    if args.max_samples > 0 and len(samples) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        chosen = sorted(rng.choice(len(samples), size=args.max_samples, replace=False).tolist())
        samples = [samples[index] for index in chosen]
    if not samples:
        raise SystemExit("No supported images were found in the requested split.")

    true_labels: List[int] = []
    predicted_labels: List[int] = []
    probabilities: List[List[float]] = []
    misclassified: List[Dict[str, Any]] = []
    print(f"Evaluating {len(samples)} images from {split_dir}...", flush=True)
    with torch.inference_mode():
        for start in range(0, len(samples), max(1, args.batch_size)):
            batch = samples[start : start + max(1, args.batch_size)]
            images = []
            for path, _ in batch:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            encoded = processor(images=images, return_tensors="pt")
            logits = model(**encoded).logits
            batch_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_predictions = np.argmax(batch_probabilities, axis=1)
            for (path, true_label), prediction, scores in zip(batch, batch_predictions, batch_probabilities):
                true_labels.append(true_label)
                predicted_labels.append(int(prediction))
                probabilities.append([float(value) for value in scores])
                if int(prediction) != true_label and len(misclassified) < max(0, args.max_misclassified):
                    misclassified.append(
                        {
                            "path": str(path),
                            "expected": id2label[true_label],
                            "predicted": id2label[int(prediction)],
                            "confidence": round(float(scores[int(prediction)]), 6),
                            "scores": {
                                id2label[index]: round(float(score), 6)
                                for index, score in enumerate(scores)
                            },
                        }
                    )
            completed = min(start + len(batch), len(samples))
            if completed % 500 == 0 or completed == len(samples):
                print(f"  {completed}/{len(samples)}", flush=True)

    report = build_report(
        np.asarray(true_labels, dtype=np.int64),
        np.asarray(predicted_labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        id2label,
    )
    report.update(
        {
            "model_dir": str(model_dir.resolve()),
            "data_split": str(split_dir.resolve()),
            "sample_count": len(samples),
            "audit_leakage_excluded_count": len(excluded_paths),
            "audit_leakage_excluded_paths": excluded_paths,
            "misclassified_examples": misclassified,
            "sampling": {
                "seed": args.seed,
                "max_samples": args.max_samples,
                "max_per_class": args.max_per_class,
            },
        }
    )
    output_path = Path(args.output) if args.output else model_dir / f"truthshield_{args.split}_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "legacy_frontend_likely_ai_decision",
                    "three_way_detection",
                )
            },
            indent=2,
        )
    )
    print(f"Full report: {output_path.resolve()}", flush=True)


def _collect_samples(split_dir: Path, label2id: Dict[str, int]) -> List[tuple[Path, int]]:
    samples: List[tuple[Path, int]] = []
    for label, label_id in sorted(label2id.items(), key=lambda item: item[1]):
        folder = split_dir / label
        if not folder.exists():
            print(f"Warning: missing label folder {folder}", flush=True)
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label_id))
    return samples


def _sample_per_class(
    samples: List[tuple[Path, int]],
    max_per_class: int,
    seed: int,
) -> List[tuple[Path, int]]:
    rng = np.random.default_rng(seed)
    selected: List[tuple[Path, int]] = []
    labels = sorted({label for _, label in samples})
    for label in labels:
        members = [sample for sample in samples if sample[1] == label]
        if len(members) > max_per_class:
            indices = sorted(rng.choice(len(members), size=max_per_class, replace=False).tolist())
            members = [members[index] for index in indices]
        selected.extend(members)
    return selected


def _exclude_audit_leakage(
    samples: List[tuple[Path, int]],
    data_dir: Path,
    split: str,
    audit_path: Path,
) -> tuple[List[tuple[Path, int]], List[str]]:
    if not audit_path.exists():
        print(f"Warning: audit report not found; no leakage exclusions applied: {audit_path}", flush=True)
        return samples, []
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    excluded_relative = set()
    group_keys = (
        "cross_split_exact_duplicates",
        "cross_split_normalized_pixel_duplicates",
        "cross_split_perceptual_hash_collisions",
        "cross_split_perceptual_near_matches",
    )
    for key in group_keys:
        for group in audit.get(key, []):
            paths = [str(path).replace("/", "\\") for path in group.get("paths", [])]
            if len({_path_split(path) for path in paths}) <= 1:
                continue
            for path in paths:
                if _path_split(path) == split:
                    excluded_relative.add(path.lower())
    kept = []
    excluded = []
    for path, label in samples:
        relative = str(path.relative_to(data_dir)).replace("/", "\\")
        if relative.lower() in excluded_relative:
            excluded.append(relative)
        else:
            kept.append((path, label))
    if excluded:
        print(f"Excluded {len(excluded)} {split} images flagged as cross-split perceptual leakage.", flush=True)
    return kept, sorted(excluded)


def _path_split(path: str) -> str:
    normalized = path.replace("/", "\\")
    return normalized.split("\\", 1)[0] if normalized else "unknown"


def build_report(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    label_ids = sorted(id2label)
    confusion = np.zeros((len(label_ids), len(label_ids)), dtype=np.int64)
    for expected, predicted in zip(true_labels, predictions):
        confusion[int(expected), int(predicted)] += 1

    classes: Dict[str, Any] = {}
    recalls = []
    f1_scores = []
    for label_id in label_ids:
        true_positive = int(confusion[label_id, label_id])
        false_positive = int(confusion[:, label_id].sum() - true_positive)
        false_negative = int(confusion[label_id, :].sum() - true_positive)
        support = int(confusion[label_id, :].sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall)
        f1_scores.append(f1)
        classes[id2label[label_id]] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    ai_label_id = next((index for index, label in id2label.items() if label == "ai_generated"), None)
    binary: Dict[str, Any] = {"available": False}
    if ai_label_id is not None:
        ai_truth = true_labels == ai_label_id
        ai_scores = probabilities[:, ai_label_id]
        ai_predictions = ai_scores >= 0.5
        true_positive = int(np.sum(ai_predictions & ai_truth))
        false_positive = int(np.sum(ai_predictions & ~ai_truth))
        true_negative = int(np.sum(~ai_predictions & ~ai_truth))
        false_negative = int(np.sum(~ai_predictions & ai_truth))
        binary = {
            "available": True,
            "threshold": 0.5,
            "accuracy": round((true_positive + true_negative) / max(1, len(ai_truth)), 6),
            "precision": round(true_positive / max(1, true_positive + false_positive), 6),
            "recall": round(true_positive / max(1, true_positive + false_negative), 6),
            "specificity": round(true_negative / max(1, true_negative + false_positive), 6),
            "false_positive_rate": round(false_positive / max(1, true_negative + false_positive), 6),
            "false_negative_rate": round(false_negative / max(1, true_positive + false_negative), 6),
            "roc_auc": round(_binary_auc(ai_truth.astype(np.int64), ai_scores), 6),
            "expected_calibration_error": round(_expected_calibration_error(ai_truth, ai_scores), 6),
            "confusion_matrix": {
                "true_ai": true_positive,
                "false_ai_alarm": false_positive,
                "true_non_ai": true_negative,
                "missed_ai": false_negative,
            },
        }
        legacy = _likely_ai_threshold_metrics(ai_truth, ai_scores, threshold=0.70)
        three_way = _three_way_metrics(
            ai_truth,
            ai_scores,
            authentic_max=AUTHENTIC_DETECTOR_MAX,
            ai_min=AI_DETECTOR_MIN,
        )
        threshold_analysis = [
            _likely_ai_threshold_metrics(ai_truth, ai_scores, threshold=threshold)
            for threshold in (0.50, 0.70, 0.80, 0.85, 0.90, 0.95)
        ]
    else:
        legacy = {"available": False}
        three_way = {"available": False}
        threshold_analysis = []

    return {
        "accuracy": round(float(np.mean(predictions == true_labels)), 6),
        "balanced_accuracy": round(float(np.mean(recalls)), 6),
        "macro_f1": round(float(np.mean(f1_scores)), 6),
        "classes": classes,
        "labels": [id2label[index] for index in label_ids],
        "confusion_matrix_rows_expected_columns_predicted": confusion.tolist(),
        "binary_ai_detection": binary,
        "legacy_frontend_likely_ai_decision": legacy,
        "three_way_detection": three_way,
        "likely_ai_threshold_analysis": threshold_analysis,
    }


def _likely_ai_threshold_metrics(ai_truth: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    predicted_ai = scores >= threshold
    true_ai = int(np.sum(predicted_ai & ai_truth))
    false_ai = int(np.sum(predicted_ai & ~ai_truth))
    ai_count = int(np.sum(ai_truth))
    non_ai_count = int(np.sum(~ai_truth))
    return {
        "available": True,
        "threshold": threshold,
        "true_ai": true_ai,
        "false_ai_alarm": false_ai,
        "ai_count": ai_count,
        "non_ai_count": non_ai_count,
        "ai_recall": round(true_ai / max(1, ai_count), 6),
        "false_positive_rate": round(false_ai / max(1, non_ai_count), 6),
        "precision": round(true_ai / max(1, true_ai + false_ai), 6),
    }


def _three_way_metrics(
    ai_truth: np.ndarray,
    scores: np.ndarray,
    authentic_max: float,
    ai_min: float,
) -> Dict[str, Any]:
    likely_ai = scores >= ai_min
    likely_authentic = scores <= authentic_max
    inconclusive = ~(likely_ai | likely_authentic)
    ai_count = int(np.sum(ai_truth))
    non_ai_count = int(np.sum(~ai_truth))
    true_ai = int(np.sum(likely_ai & ai_truth))
    false_ai = int(np.sum(likely_ai & ~ai_truth))
    true_authentic = int(np.sum(likely_authentic & ~ai_truth))
    ai_mislabeled_authentic = int(np.sum(likely_authentic & ai_truth))
    decisive = int(np.sum(likely_ai | likely_authentic))
    decisive_correct = true_ai + true_authentic
    return {
        "available": True,
        "authentic_max": authentic_max,
        "ai_min": ai_min,
        "counts": {
            "likely_ai": int(np.sum(likely_ai)),
            "likely_authentic": int(np.sum(likely_authentic)),
            "inconclusive": int(np.sum(inconclusive)),
            "true_ai": true_ai,
            "false_ai_alarm": false_ai,
            "true_authentic": true_authentic,
            "ai_mislabeled_authentic": ai_mislabeled_authentic,
            "ai_inconclusive": int(np.sum(inconclusive & ai_truth)),
            "non_ai_inconclusive": int(np.sum(inconclusive & ~ai_truth)),
        },
        "false_positive_rate": round(false_ai / max(1, non_ai_count), 6),
        "false_negative_rate_as_authentic": round(ai_mislabeled_authentic / max(1, ai_count), 6),
        "ai_recall": round(true_ai / max(1, ai_count), 6),
        "authentic_recall": round(true_authentic / max(1, non_ai_count), 6),
        "inconclusive_rate": round(float(np.mean(inconclusive)), 6),
        "decisive_coverage": round(decisive / max(1, len(ai_truth)), 6),
        "decisive_accuracy": round(decisive_correct / max(1, decisive), 6),
        "likely_ai_precision": round(true_ai / max(1, true_ai + false_ai), 6),
        "likely_authentic_precision": round(true_authentic / max(1, true_authentic + ai_mislabeled_authentic), 6),
    }


def _binary_auc(truth: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[truth == 1]
    negatives = scores[truth == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    ordered = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[ordered] = np.arange(1, len(scores) + 1, dtype=np.float64)
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    del unique_scores
    for group_id, count in enumerate(counts):
        if count > 1:
            members = inverse == group_id
            ranks[members] = float(np.mean(ranks[members]))
    positive_rank_sum = float(np.sum(ranks[truth == 1]))
    return (positive_rank_sum - len(positives) * (len(positives) + 1) / 2.0) / (len(positives) * len(negatives))


def _expected_calibration_error(truth: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        members = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
        if not np.any(members):
            continue
        error += float(np.mean(members)) * abs(float(np.mean(scores[members])) - float(np.mean(truth[members])))
    return error


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train_manipulation_localizer import (
    ManifestSegmentationDataset,
    _image_score,
)


AUTHENTIC_LABELS = {"authentic", "real", "real_camera"}
GENERATED_LABELS = {"generated", "ai_generated"}
MANIPULATED_LABELS = {"manipulated", "ai_manipulated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a TruthShield pixel-mask manipulation localizer."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("tuning", "calibration", "locked_test"),
        default="tuning",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--pixel-threshold", type=float, default=0.5)
    parser.add_argument("--max-view-range", type=float, default=0.18)
    parser.add_argument("--allow-locked-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "locked_test" and not args.allow_locked_test:
        raise SystemExit(
            "Refusing to inspect locked_test without --allow-locked-test. Run it only once "
            "for the final promotion gate after thresholds are frozen."
        )
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
    except ImportError as exc:
        raise SystemExit("Install training/requirements-train.txt before evaluation.") from exc
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this evaluation command.")

    model_dir = args.model_dir.resolve()
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    resolution = int(checkpoint.get("config", {}).get("resolution", 384))
    model = lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=1,
    )
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()
    dataset = ManifestSegmentationDataset(
        args.data_dir.resolve(),
        split=args.split,
        resolution=resolution,
        augment=False,
        max_samples=args.max_samples,
        seed=557,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.workers),
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    rows: list[dict[str, Any]] = []
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["pixel_values"].cuda(non_blocking=True)
            probabilities = torch.sigmoid(model(images)["out"])
            probabilities = torch.nn.functional.interpolate(
                probabilities,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            ).detach().cpu().numpy()
            for index in range(len(probabilities)):
                example = dataset.examples[processed]
                original_probability = probabilities[index, 0]
                original_score = _image_score(original_probability)
                support, region, area_ratio = _localized_support(
                    original_probability,
                    threshold=args.pixel_threshold,
                )
                view_scores = [original_score]
                for variant in ("jpeg70", "resize75"):
                    tensor = _controlled_view_tensor(
                        example.image_path,
                        resolution=resolution,
                        variant=variant,
                        torch=torch,
                    ).unsqueeze(0).cuda(non_blocking=True)
                    view_probability = torch.sigmoid(model(tensor)["out"])
                    view_probability = torch.nn.functional.interpolate(
                        view_probability,
                        size=(resolution, resolution),
                        mode="bilinear",
                        align_corners=False,
                    )
                    view_scores.append(_image_score(view_probability[0, 0].detach().cpu().numpy()))
                view_range = max(view_scores) - min(view_scores)
                rows.append(
                    {
                        "path": example.relative_path,
                        "label": example.class_label,
                        "generator_or_editor": example.generator_or_editor,
                        "manipulation_score": round(original_score, 8),
                        "localized_or_persistent_support": bool(support),
                        "suspicious_region": json.dumps(region) if region else "",
                        "predicted_region_area_ratio": round(area_ratio, 8),
                        "view_score_range": round(view_range, 8),
                        "stable_across_views": bool(view_range <= args.max_view_range),
                        "original_score": round(view_scores[0], 8),
                        "jpeg70_score": round(view_scores[1], 8),
                        "resize75_score": round(view_scores[2], 8),
                    }
                )
                processed += 1
                if processed % 50 == 0 or processed == len(dataset):
                    print(f"Scored {processed}/{len(dataset)}", flush=True)

    report = _report(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"Saved predictions to: {args.output}", flush=True)
    print(f"Saved report to: {report_path}", flush=True)


def _controlled_view_tensor(
    path: Path,
    *,
    resolution: int,
    variant: str,
    torch: Any,
) -> Any:
    from torchvision.transforms import functional as vision

    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if variant == "jpeg70":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        buffer.seek(0)
        with Image.open(buffer) as handle:
            image = handle.convert("RGB")
    elif variant == "resize75":
        reduced = (
            max(64, int(image.width * 0.75)),
            max(64, int(image.height * 0.75)),
        )
        image = image.resize(reduced, Image.Resampling.LANCZOS)
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    tensor = vision.pil_to_tensor(image).float().div_(255.0)
    return vision.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def _localized_support(
    probability: np.ndarray,
    *,
    threshold: float,
    minimum_area_ratio: float = 0.001,
) -> tuple[bool, list[int] | None, float]:
    binary = (np.asarray(probability) >= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return False, None, 0.0
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[largest_index, cv2.CC_STAT_LEFT])
    y = int(stats[largest_index, cv2.CC_STAT_TOP])
    width = int(stats[largest_index, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_index, cv2.CC_STAT_HEIGHT])
    area = int(stats[largest_index, cv2.CC_STAT_AREA])
    ratio = area / max(1, binary.size)
    return ratio >= minimum_area_ratio, [x, y, x + width, y + height], float(ratio)


def _report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.asarray(
        [str(row["label"]).lower() in MANIPULATED_LABELS for row in rows],
        dtype=np.int64,
    )
    scores = np.asarray([float(row["manipulation_score"]) for row in rows], dtype=np.float64)
    support = np.asarray(
        [bool(row["localized_or_persistent_support"]) for row in rows],
        dtype=bool,
    )
    stable = np.asarray([bool(row["stable_across_views"]) for row in rows], dtype=bool)
    best = _best_constrained_threshold(rows)
    return {
        "record_count": len(rows),
        "roc_auc": round(float(roc_auc_score(truth, scores)), 6),
        "average_precision": round(float(average_precision_score(truth, scores)), 6),
        "localized_support_rate": round(float(support.mean()), 6),
        "manipulated_localized_support_rate": round(float(support[truth == 1].mean()), 6),
        "authentic_localized_support_rate": round(
            float(
                np.mean(
                    [
                        bool(row["localized_or_persistent_support"])
                        for row in rows
                        if str(row["label"]).lower() in AUTHENTIC_LABELS
                    ]
                )
            ),
            6,
        ),
        "unstable_across_views": int((~stable).sum()),
        "score_quantiles_0_10_50_90_99_100_percent": {
            label: [round(float(value), 6) for value in np.quantile(values, [0, 0.1, 0.5, 0.9, 0.99, 1])]
            for label in sorted({str(row["label"]) for row in rows})
            if len(
                values := np.asarray(
                    [float(row["manipulation_score"]) for row in rows if str(row["label"]) == label]
                )
            )
        },
        "metrics_at_0_5": _threshold_metrics(rows, 0.5),
        "best_threshold_meeting_tuning_constraints": best,
    }


def _best_constrained_threshold(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = sorted({float(row["manipulation_score"]) for row in rows}, reverse=True)
    best = None
    for threshold in candidates:
        metrics = _threshold_metrics(rows, threshold)
        if (
            metrics["precision"] >= 0.95
            and metrics["authentic_false_warning_rate"] <= 0.01
            and metrics["generated_false_manipulation_rate"] <= 0.01
        ):
            if best is None or metrics["recall"] > best["recall"]:
                best = metrics
    return best


def _threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    predicted = [
        row
        for row in rows
        if float(row["manipulation_score"]) >= threshold
        and bool(row["localized_or_persistent_support"])
        and bool(row["stable_across_views"])
    ]
    true_manipulated = sum(str(row["label"]).lower() in MANIPULATED_LABELS for row in predicted)
    false_warnings = len(predicted) - true_manipulated
    total_manipulated = sum(str(row["label"]).lower() in MANIPULATED_LABELS for row in rows)
    authentic = [row for row in rows if str(row["label"]).lower() in AUTHENTIC_LABELS]
    generated = [row for row in rows if str(row["label"]).lower() in GENERATED_LABELS]
    authentic_warnings = sum(str(row["label"]).lower() in AUTHENTIC_LABELS for row in predicted)
    generated_warnings = sum(str(row["label"]).lower() in GENERATED_LABELS for row in predicted)
    return {
        "threshold": round(float(threshold), 8),
        "predicted_manipulated": len(predicted),
        "true_manipulated": true_manipulated,
        "false_manipulation_warnings": false_warnings,
        "precision": true_manipulated / max(1, len(predicted)),
        "recall": true_manipulated / max(1, total_manipulated),
        "authentic_false_warning_rate": authentic_warnings / max(1, len(authentic)),
        "generated_false_manipulation_rate": generated_warnings / max(1, len(generated)),
    }


if __name__ == "__main__":
    main()

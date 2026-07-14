from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TruthShield image detector against common upload transformations."
    )
    parser.add_argument("images", nargs="+", help="One or more image paths to check.")
    parser.add_argument("--model-dir", default="training/models/truthshield-image-detector-v2")
    parser.add_argument(
        "--minimum-ai-probability",
        type=float,
        default=0.70,
        help="Minimum acceptable AI probability for every variant (default: 0.70).",
    )
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "backend"
    sys.path.insert(0, str(backend_dir))

    from analyzers.ai_detectors import (  # pylint: disable=import-outside-toplevel
        _load_pipeline,
        _normalize_outputs,
        _prepare_classifier_image,
        _synthetic_probability,
    )

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"Model folder not found: {model_dir}")
    classifier = _load_pipeline(str(model_dir))
    reports: List[Dict[str, Any]] = []
    failed_variants: List[str] = []

    for raw_path in args.images:
        image_path = Path(raw_path).resolve()
        if not image_path.is_file():
            raise SystemExit(f"Image not found: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        variants = list(_robustness_variants(image))
        prepared = [_prepare_classifier_image(classifier, variant) for _, variant in variants]
        outputs = classifier(prepared, top_k=None, batch_size=min(8, len(prepared)))
        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]

        variant_reports = []
        probabilities = []
        for (variant_name, variant), raw_output in zip(variants, outputs):
            normalized = _normalize_outputs(raw_output)
            probability = _synthetic_probability(normalized)
            probabilities.append(probability)
            passed = probability >= args.minimum_ai_probability
            if not passed:
                failed_variants.append(f"{image_path.name}:{variant_name}")
            variant_reports.append(
                {
                    "variant": variant_name,
                    "width": variant.width,
                    "height": variant.height,
                    "ai_probability": round(probability, 6),
                    "passed": passed,
                    "top_labels": normalized[:3],
                }
            )

        reports.append(
            {
                "image": str(image_path),
                "variant_count": len(variant_reports),
                "minimum_ai_probability": round(min(probabilities), 6),
                "mean_ai_probability": round(statistics.fmean(probabilities), 6),
                "median_ai_probability": round(statistics.median(probabilities), 6),
                "all_variants_passed": all(item["passed"] for item in variant_reports),
                "variants": variant_reports,
            }
        )

    report = {
        "model_dir": str(model_dir),
        "acceptance_threshold": args.minimum_ai_probability,
        "all_variants_passed": not failed_variants,
        "failed_variants": failed_variants,
        "images": reports,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    if failed_variants:
        raise SystemExit(2)


def _robustness_variants(image: Image.Image) -> Iterable[Tuple[str, Image.Image]]:
    rgb = image.convert("RGB")
    yield "original", rgb
    for quality in (95, 75, 50):
        yield f"jpeg_quality_{quality}", _jpeg_round_trip(rgb, quality)

    for scale in (0.75, 0.50):
        resized = rgb.resize(
            (max(64, round(rgb.width * scale)), max(64, round(rgb.height * scale))),
            Image.Resampling.LANCZOS,
        )
        yield f"resize_{round(scale * 100)}pct", resized

    crop_scale = 0.85
    crop_width = max(64, round(rgb.width * crop_scale))
    crop_height = max(64, round(rgb.height * crop_scale))
    left = max(0, (rgb.width - crop_width) // 2)
    top = max(0, (rgb.height - crop_height) // 2)
    yield "center_crop_85pct", rgb.crop((left, top, left + crop_width, top + crop_height))

    social = rgb.resize(
        (max(64, round(rgb.width * 0.55)), max(64, round(rgb.height * 0.55))),
        Image.Resampling.LANCZOS,
    )
    yield "social_resize_jpeg_70", _jpeg_round_trip(social, 70)


def _jpeg_round_trip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


if __name__ == "__main__":
    main()

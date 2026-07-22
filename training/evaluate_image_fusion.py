from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for root in (REPO_ROOT, BACKEND_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from analyzers.ai_detectors import _prepare_classifier_image, _synthetic_probability  # noqa: E402
from analyzers.community_forensics import _official_test_preprocess  # noqa: E402
from analyzers.image_fusion import _load_fusion_bundle, _logit  # noqa: E402
from training.media_manifest import read_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a generator-isolated image split with Community Forensics, the "
            "TruthShield comparison model, and the frozen v4 fusion head."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="locked_test")
    parser.add_argument("--community-repo", type=Path, required=True)
    parser.add_argument("--community-model-id", default="OwensLab/commfor-model-224")
    parser.add_argument("--comparison-model-dir", type=Path, required=True)
    parser.add_argument("--fusion-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-view-score-range", type=float, default=0.18)
    return parser.parse_args()


def main() -> None:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    args = parse_args()
    manifest_path = args.data_dir / "manifest.v4.jsonl"
    records = [
        row
        for row in read_manifest(manifest_path)
        if str(row.get("media_type") or "") == "image"
        and str(row.get("split") or "") == args.split
    ]
    if args.max_records > 0:
        records = records[: args.max_records]
    if not records:
        raise SystemExit(f"No image records found for split {args.split!r} in {manifest_path}.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")

    _verify_comparison_files(args.comparison_model_dir)
    community_model = _load_community_model(
        args.community_repo,
        args.community_model_id,
        device=args.device,
    )
    comparison_processor = AutoImageProcessor.from_pretrained(str(args.comparison_model_dir))
    comparison_model = AutoModelForImageClassification.from_pretrained(
        str(args.comparison_model_dir)
    ).to(args.device).eval()
    fusion_bundle = _load_fusion_bundle(str(args.fusion_model.resolve()))
    fusion_model = fusion_bundle["model"]

    views: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        source = args.data_dir / str(record["path"])
        with Image.open(source) as image:
            rgb = image.convert("RGB")
        for name, view in controlled_views(rgb):
            views.append({"record_index": record_index, "name": name, "image": view})

    print(
        f"Scoring {len(records)} {args.split} records through {len(views)} controlled views on {args.device}.",
        flush=True,
    )
    for start in range(0, len(views), max(1, args.batch_size)):
        batch = views[start : start + max(1, args.batch_size)]
        images = [item["image"] for item in batch]
        community_values = torch.cat(
            [_official_test_preprocess(image, torch=torch) for image in images], dim=0
        ).to(args.device)
        comparison_images = [
            _prepare_classifier_image(_ClassifierConfigAdapter(comparison_model), image)
            for image in images
        ]
        comparison_values = comparison_processor(images=comparison_images, return_tensors="pt")
        comparison_values = {
            key: value.to(args.device) if hasattr(value, "to") else value
            for key, value in comparison_values.items()
        }
        with torch.inference_mode():
            community_probabilities = torch.sigmoid(community_model(community_values)).reshape(-1)
            comparison_logits = comparison_model(**comparison_values).logits
            comparison_probabilities = torch.softmax(comparison_logits, dim=-1)
        for item, community_probability, comparison_scores in zip(
            batch,
            community_probabilities.detach().cpu().tolist(),
            comparison_probabilities.detach().cpu().tolist(),
        ):
            comparison_probability = comparison_synthetic_probability(
                comparison_scores,
                comparison_model.config.id2label,
            )
            if comparison_probability is None:
                raise RuntimeError(
                    "The comparison model labels cannot be mapped to authentic/generated probabilities."
                )
            features = [[_logit(float(community_probability)), _logit(comparison_probability)]]
            fused = fusion_model.predict_proba(features)[0]
            classes = list(getattr(fusion_model, "classes_", range(len(fused))))
            positive_index = classes.index(1) if 1 in classes else len(fused) - 1
            item["community_score"] = float(community_probability)
            item["comparison_score"] = float(comparison_probability)
            item["fusion_score"] = float(fused[positive_index])
            item.pop("image", None)
        completed = min(start + len(batch), len(views))
        if completed % 160 == 0 or completed == len(views):
            print(f"Scored {completed}/{len(views)} views", flush=True)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for view in views:
        grouped.setdefault(int(view["record_index"]), []).append(view)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        record_views = grouped[index]
        original = next(view for view in record_views if view["name"] == "original")
        fused_scores = [float(view["fusion_score"]) for view in record_views]
        score_range = max(fused_scores) - min(fused_scores)
        unstable = score_range > args.max_view_score_range
        rows.append(
            {
                "path": str(record["path"]),
                "sha256": str(record.get("sha256") or ""),
                "label": str(record.get("class_label") or ""),
                "generator_or_editor": str(record.get("generator_or_editor") or ""),
                "semantic_category": str(record.get("semantic_category") or ""),
                "transformation": str(record.get("transformation") or ""),
                "generation_score": round(float(original["fusion_score"]), 8),
                "manipulation_score": "",
                "community_forensics_score": round(float(original["community_score"]), 8),
                "comparison_model_score": round(float(original["comparison_score"]), 8),
                "view_score_range": round(score_range, 8),
                "force_inconclusive": str(unstable).lower(),
                "view_scores_json": json.dumps(
                    {view["name"]: round(float(view["fusion_score"]), 8) for view in record_views},
                    sort_keys=True,
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = build_scoring_report(rows, args, fusion_bundle)
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved locked predictions to: {args.output}")
    print(f"Saved scoring report to: {report_path}")


def controlled_views(image: Image.Image) -> list[tuple[str, Image.Image]]:
    rgb = image.convert("RGB")
    views: list[tuple[str, Image.Image]] = [("original", rgb)]
    longest = max(rgb.size)
    if longest > 768:
        scale = 768.0 / longest
        views.append(
            (
                "resized_768",
                rgb.resize(
                    (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                    Image.Resampling.LANCZOS,
                ),
            )
        )
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=78, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        views.append(("jpeg_q78", recompressed.convert("RGB")))
    return views


def comparison_synthetic_probability(
    scores: Iterable[float],
    id2label: dict[int, str] | dict[str, str],
) -> float | None:
    outputs = [
        {
            "label": str(id2label.get(index, id2label.get(str(index), str(index)))),
            "score": float(score),
        }
        for index, score in enumerate(scores)
    ]
    return _synthetic_probability(outputs)


def build_scoring_report(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    fusion_bundle: dict[str, Any],
) -> dict[str, Any]:
    scores = [float(row["generation_score"]) for row in rows]
    labels = [str(row["label"]).lower() in {"generated", "ai_generated"} for row in rows]
    return {
        "split": args.split,
        "record_count": len(rows),
        "authentic_count": sum(not label for label in labels),
        "generated_count": sum(labels),
        "generation_roc_auc": _auc(scores, labels),
        "unstable_across_views": sum(row["force_inconclusive"] == "true" for row in rows),
        "max_view_score_range": args.max_view_score_range,
        "model_versions": {
            "community_forensics": args.community_model_id,
            "comparison_model": args.comparison_model_dir.name,
            "fusion_inputs": fusion_bundle.get("input_model_versions", {}),
        },
    }


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    positive = [score for score, label in zip(scores, labels) if label]
    negative = [score for score, label in zip(scores, labels) if not label]
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return round(wins / (len(positive) * len(negative)), 8)


def _load_community_model(repo: Path, model_id: str, *, device: str) -> Any:
    models_path = repo / "models.py"
    license_path = repo / "LICENSE"
    if not models_path.is_file() or not license_path.is_file():
        raise SystemExit(f"Community Forensics checkout is incomplete: {repo}")
    if "mit license" not in license_path.read_text(encoding="utf-8", errors="ignore").lower():
        raise SystemExit(f"Community Forensics checkout has no expected MIT license: {repo}")
    spec = importlib.util.spec_from_file_location("_truthshield_locked_commfor", models_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {models_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ViTClassifier.from_pretrained(model_id, device=device).to(device).eval()


def _verify_comparison_files(model_dir: Path) -> None:
    required = ("config.json", "model.safetensors", "preprocessor_config.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"Comparison model folder {model_dir} is missing: {', '.join(missing)}"
        )


class _ClassifierConfigAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model


if __name__ == "__main__":
    main()

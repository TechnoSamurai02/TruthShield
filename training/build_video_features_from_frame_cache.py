from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


SPLITS = ("train", "validation", "test")
LABELS = ("ai_generated", "real_camera")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build video-level temporal features from cached frame embeddings. This avoids rerunning "
            "the neural backbone while preserving ordered luma, noise, edge, optical-flow, duplicate, "
            "and scene-cut analysis."
        )
    )
    parser.add_argument("--frame-dir", default="training/data/video_frames")
    parser.add_argument("--embedding-dir", default="training/data/video_frame_embeddings")
    parser.add_argument("--frame-model", default="training/models/truthshield-video-frame-detector")
    parser.add_argument("--output", default="training/data/video_features_from_frame_cache.jsonl")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForImageClassification

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from analyzers.video_forensics import VideoForensicsAccumulator

    frame_dir = Path(args.frame_dir)
    embedding_dir = Path(args.embedding_dir)
    model_path = str(Path(args.frame_model).resolve())
    model = AutoModelForImageClassification.from_pretrained(model_path)
    head = _classification_linear(model, 2)
    if head is None:
        raise SystemExit("The frame model does not expose a two-class linear classifier head.")
    weight = head.weight.detach().cpu().numpy().astype(np.float64)
    bias = (
        head.bias.detach().cpu().numpy().astype(np.float64)
        if head.bias is not None
        else np.zeros(2, dtype=np.float64)
    )
    ai_index = _ai_label_index(model.config.id2label)
    del model

    records: List[Dict[str, Any]] = []
    for split in SPLITS:
        samples = _collect_samples(frame_dir / split)
        cache_path = embedding_dir / f"{split}.npz"
        if not cache_path.is_file():
            raise SystemExit(f"Missing cached embeddings: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cached:
            embeddings = np.asarray(cached["embeddings"], dtype=np.float64)
        if len(samples) != len(embeddings):
            raise SystemExit(
                f"{split} sample/cache mismatch: {len(samples)} frame paths versus "
                f"{len(embeddings)} embeddings."
            )
        probabilities = _softmax(embeddings @ weight.T + bias)[:, ai_index]
        grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for index, (path, label) in enumerate(samples):
            grouped[(label, _source_id(path))].append(index)

        for video_number, ((label, source_id), indices) in enumerate(sorted(grouped.items()), start=1):
            indices.sort(key=lambda index: _frame_number(samples[index][0]))
            accumulator = VideoForensicsAccumulator()
            for index in indices:
                frame_path = samples[index][0]
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Could not decode extracted frame: {frame_path}")
                probability = float(probabilities[index])
                frame_result = {
                    # Production truth_score also includes still-image forensic evidence, which is
                    # intentionally not recomputed in this fast cache path. Keep it neutral so a
                    # regenerated cache cannot learn from an artificial probability surrogate.
                    "truth_score": 50.0,
                    "detectors": [
                        {
                            "name": "truthshield_video_frame_detector_cached",
                            "status": "completed",
                            "synthetic_probability": probability,
                        }
                    ],
                    # These remain constant by design. The temporal model therefore cannot learn
                    # a misleading dependence on still-image forensic columns absent from this cache.
                    "technical_details": {
                        "forensic_analysis": {"synthetic_artifact_probability": 0.5}
                    },
                }
                accumulator.update(frame, frame_result)
            summary = accumulator.summary()
            records.append(
                {
                    "source_id": f"{split}/{label}/{source_id}",
                    "source_video": source_id,
                    "split": split,
                    "label": label,
                    "features": summary["features"],
                    "frames_analyzed": summary["frames_seen"],
                    "analysis_coverage": {
                        "native_pixels_examined": summary["native_pixels_examined"],
                        "training_frame_sampling": "16 uniformly spaced frames per source video",
                    },
                    "analysis_config": {
                        "frame_model": model_path,
                        "source": "cached_uniform_frame_embeddings",
                        "tile_analysis": False,
                    },
                }
            )
            if video_number % 50 == 0 or video_number == len(grouped):
                print(f"{split}: {video_number}/{len(grouped)} videos", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temp_path.replace(output_path)
    print(f"Wrote {len(records)} video feature records: {output_path.resolve()}", flush=True)


def _collect_samples(split_dir: Path) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    for label in LABELS:
        folder = split_dir / label
        samples.extend(
            (path, label)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return samples


def _source_id(path: Path) -> str:
    return path.stem.split("_frame_", 1)[0]


def _frame_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_frame_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _classification_linear(model: Any, output_labels: int) -> Any | None:
    import torch

    candidates = [
        module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and module.out_features == output_labels
        and any(marker in name.lower() for marker in ("classifier", "score", "head"))
    ]
    return candidates[-1] if candidates else None


def _ai_label_index(id2label: Dict[int, str]) -> int:
    for index, label in id2label.items():
        normalized = str(label).lower().replace("-", "_").replace(" ", "_")
        if "ai" in normalized or "synthetic" in normalized or "generated" in normalized:
            return int(index)
    raise SystemExit(f"Could not identify the AI class in id2label={id2label}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


if __name__ == "__main__":
    main()

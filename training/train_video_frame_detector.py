from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "validation", "test")
LABELS = ("ai_generated", "real_camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CPU-friendly video-frame head from a validated ResNet forensic backbone. "
            "Embeddings are extracted once and cached, then regularized heads are selected on validation."
        )
    )
    parser.add_argument("--data-dir", default="training/data/video_frames")
    parser.add_argument("--base-model", default="training/models/truthshield-image-detector-v2")
    parser.add_argument("--output-dir", default="training/models/truthshield-video-frame-detector")
    parser.add_argument("--embedding-dir", default="training/data/video_frame_embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--preprocess-max-dimension", type=int, default=384)
    parser.add_argument("--c-values", default="0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    args = parse_args()
    data_dir = Path(args.data_dir)
    base_model = str(Path(args.base_model).resolve()) if Path(args.base_model).exists() else args.base_model
    output_dir = Path(args.output_dir)
    embedding_dir = Path(args.embedding_dir)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    samples = {split: _collect_samples(data_dir / split) for split in SPLITS}
    for split, split_samples in samples.items():
        counts = {label: sum(sample[1] == label for sample in split_samples) for label in LABELS}
        if not split_samples or any(count == 0 for count in counts.values()):
            raise SystemExit(f"{split} needs both labels. Found: {counts}")
        print(f"{split}: {counts}", flush=True)

    processor = AutoImageProcessor.from_pretrained(base_model)
    source_model = AutoModelForImageClassification.from_pretrained(base_model)
    source_model.eval()
    if not hasattr(source_model, "resnet"):
        raise SystemExit("The CPU embedding trainer currently requires a Hugging Face ResNet backbone.")

    arrays: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
    with torch.inference_mode():
        for split in SPLITS:
            arrays[split] = _load_or_extract_embeddings(
                split=split,
                samples=samples[split],
                cache_path=embedding_dir / f"{split}.npz",
                base_model=base_model,
                processor=processor,
                source_model=source_model,
                batch_size=max(1, args.batch_size),
                max_dimension=max(224, args.preprocess_max_dimension),
                rebuild=args.rebuild_embeddings,
            )

    train_x, train_y, _ = arrays["train"]
    validation_x, validation_y, validation_paths = arrays["validation"]
    test_x, test_y, test_paths = arrays["test"]
    candidates = []
    for scaling in ("none", "standard"):
        scaler = None
        fit_train_x = train_x
        if scaling == "standard":
            scaler = StandardScaler()
            fit_train_x = scaler.fit_transform(train_x)
        for c_value in _parse_c_values(args.c_values):
            print(f"Fitting {scaling}-scaled regularized head C={c_value:g}...", flush=True)
            classifier = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=max(100, args.max_iterations),
                solver="lbfgs",
                random_state=args.seed,
            )
            classifier.fit(fit_train_x, train_y)
            validation_probabilities = _positive_probabilities(
                classifier, validation_x, scaler=scaler
            )
            validation_video_y, validation_video_probabilities = _aggregate_by_video(
                validation_y, validation_probabilities, validation_paths
            )
            threshold = _best_threshold(validation_video_y, validation_video_probabilities)
            frame_metrics = _evaluate(validation_y, validation_probabilities, 0.5)
            video_metrics = _evaluate(validation_video_y, validation_video_probabilities, threshold)
            selection_score = (
                float(video_metrics["balanced_accuracy"])
                + float(frame_metrics["roc_auc"]) * 0.08
                - float(frame_metrics["brier_score"]) * 0.04
            )
            candidates.append(
                {
                    "c": c_value,
                    "scaling": scaling,
                    "scaler": scaler,
                    "classifier": classifier,
                    "threshold": threshold,
                    "selection_score": selection_score,
                    "validation_frame": frame_metrics,
                    "validation_video": video_metrics,
                }
            )
    candidates.sort(
        key=lambda item: (float(item["selection_score"]), -abs(float(item["threshold"]) - 0.5)),
        reverse=True,
    )
    best = candidates[0]
    classifier = best["classifier"]
    scaler = best["scaler"]
    test_probabilities = _positive_probabilities(classifier, test_x, scaler=scaler)
    test_video_y, test_video_probabilities = _aggregate_by_video(test_y, test_probabilities, test_paths)
    test_frame_metrics = _evaluate(test_y, test_probabilities, 0.5)
    test_video_metrics = _evaluate(test_video_y, test_video_probabilities, float(best["threshold"]))

    del source_model
    label2id = {"ai_generated": 0, "real_camera": 1}
    id2label = {index: label for label, index in label2id.items()}
    target_model = AutoModelForImageClassification.from_pretrained(
        base_model,
        num_labels=2,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    _install_binary_head(target_model, classifier, scaler=scaler)
    target_model.config.truthshield_video_threshold = float(best["threshold"])
    target_model.config.truthshield_training_method = "frozen_resnet_embeddings_regularized_logistic_head"
    target_model.config.truthshield_base_model = base_model
    target_model.config.truthshield_training_frame_encoding = "opencv_jpeg_95"
    target_model.config.truthshield_preprocess_max_dimension = int(
        max(224, args.preprocess_max_dimension)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    target_model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)

    report = {
        "training_method": "frozen_resnet_embeddings_regularized_logistic_head",
        "base_model": base_model,
        "selected_c": float(best["c"]),
        "selected_scaling": str(best["scaling"]),
        "video_decision_threshold": float(best["threshold"]),
        "dataset_counts": {
            split: {
                label: sum(sample[1] == label for sample in samples[split])
                for label in LABELS
            }
            for split in SPLITS
        },
        "embedding_dimension": int(train_x.shape[1]),
        "validation": {
            "frame": best["validation_frame"],
            "video": best["validation_video"],
        },
        "test": {
            "frame": test_frame_metrics,
            "video": test_video_metrics,
        },
        "candidate_heads": [
            {
                "c": float(candidate["c"]),
                "scaling": str(candidate["scaling"]),
                "selection_score": round(float(candidate["selection_score"]), 8),
                "threshold": float(candidate["threshold"]),
                "validation_frame": candidate["validation_frame"],
                "validation_video": candidate["validation_video"],
            }
            for candidate in candidates
        ],
    }
    metrics_path = output_dir / "truthshield_video_frame_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_c": report["selected_c"],
                "selected_scaling": report["selected_scaling"],
                "threshold": report["video_decision_threshold"],
                "validation": report["validation"],
                "test": report["test"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Model: {output_dir.resolve()}", flush=True)
    print(f"Metrics: {metrics_path.resolve()}", flush=True)


def _collect_samples(split_dir: Path) -> List[Tuple[Path, str]]:
    samples = []
    for label in LABELS:
        folder = split_dir / label
        if not folder.is_dir():
            continue
        samples.extend(
            (path, label)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return samples


def _load_or_extract_embeddings(
    split: str,
    samples: Sequence[Tuple[Path, str]],
    cache_path: Path,
    base_model: str,
    processor: Any,
    source_model: Any,
    batch_size: int,
    max_dimension: int,
    rebuild: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    key = _cache_key(samples, base_model, max_dimension)
    paths = [str(path) for path, _ in samples]
    labels = np.asarray([1 if label == "ai_generated" else 0 for _, label in samples], dtype=np.int64)
    if cache_path.is_file() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_key = str(cached["key"][0])
            if cached_key == key and len(cached["labels"]) == len(samples):
                print(f"Using cached {split} embeddings: {cache_path}", flush=True)
                return np.asarray(cached["embeddings"], dtype=np.float32), labels, paths

    import torch

    batches = []
    total_batches = (len(samples) + batch_size - 1) // batch_size
    print(f"Extracting {split} embeddings from {len(samples)} frames...", flush=True)
    for batch_index, start in enumerate(range(0, len(samples), batch_size), start=1):
        batch_samples = samples[start : start + batch_size]
        images = []
        for path, _ in batch_samples:
            with Image.open(path) as image:
                images.append(_resize_for_model_prep(image.convert("RGB"), max_dimension))
        encoded = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            output = source_model.resnet(pixel_values=encoded["pixel_values"])
            pooled = output.pooler_output.flatten(1).cpu().numpy().astype(np.float32, copy=False)
        batches.append(pooled)
        if batch_index % 20 == 0 or batch_index == total_batches:
            print(f"  {split}: {min(start + len(batch_samples), len(samples))}/{len(samples)}", flush=True)
    embeddings = np.concatenate(batches, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        np.savez(handle, embeddings=embeddings, labels=labels, key=np.asarray([key]))
    temp_path.replace(cache_path)
    return embeddings, labels, paths


def _cache_key(samples: Sequence[Tuple[Path, str]], base_model: str, max_dimension: int) -> str:
    digest = hashlib.sha256()
    digest.update(base_model.encode("utf-8"))
    digest.update(str(max_dimension).encode("ascii"))
    for path, label in samples:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(label.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _resize_for_model_prep(image: Image.Image, max_dimension: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_dimension:
        return image
    scale = max_dimension / float(longest)
    return image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.LANCZOS,
    )


def _parse_c_values(value: str) -> List[float]:
    parsed = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not parsed or any(item <= 0 for item in parsed):
        raise SystemExit("--c-values must contain positive numbers.")
    return parsed


def _positive_probabilities(
    classifier: Any, features: np.ndarray, scaler: Any | None = None
) -> np.ndarray:
    classes = [int(value) for value in classifier.classes_]
    if scaler is not None:
        features = scaler.transform(features)
    return np.asarray(classifier.predict_proba(features)[:, classes.index(1)], dtype=np.float64)


def _source_id(path: str) -> str:
    return Path(path).stem.split("_frame_", 1)[0]


def _aggregate_by_video(
    truth: np.ndarray, probabilities: np.ndarray, paths: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    grouped: Dict[str, List[int]] = {}
    for index, path in enumerate(paths):
        grouped.setdefault(_source_id(path), []).append(index)
    video_truth = []
    video_probabilities = []
    for source_id in sorted(grouped):
        indices = grouped[source_id]
        labels = {int(truth[index]) for index in indices}
        if len(labels) != 1:
            raise RuntimeError(f"Mixed labels inside source video {source_id}")
        values = np.asarray([probabilities[index] for index in indices], dtype=np.float64)
        aggregate = float(np.mean(values) * 0.70 + np.quantile(values, 0.90) * 0.30)
        video_truth.append(labels.pop())
        video_probabilities.append(max(0.0, min(1.0, aggregate)))
    return np.asarray(video_truth, dtype=np.int64), np.asarray(video_probabilities, dtype=np.float64)


def _best_threshold(truth: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = _evaluate(truth, probabilities, float(threshold))
        key = (
            float(metrics["balanced_accuracy"]),
            float(metrics["f1"]),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return round(best_threshold, 4)


def _evaluate(truth: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, float | int]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    predicted = probabilities >= threshold
    positive = truth == 1
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    true_negative = int(np.sum(~predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))
    recall = true_positive / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    precision = true_positive / max(1, true_positive + false_positive)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round((true_positive + true_negative) / max(1, len(truth)), 6),
        "balanced_accuracy": round((recall + specificity) / 2.0, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
        "roc_auc": round(float(roc_auc_score(truth, probabilities)), 6),
        "average_precision": round(float(average_precision_score(truth, probabilities)), 6),
        "brier_score": round(float(brier_score_loss(truth, probabilities)), 6),
        "true_ai": true_positive,
        "false_ai_alarm": false_positive,
        "true_real": true_negative,
        "missed_ai": false_negative,
    }


def _install_binary_head(target_model: Any, classifier: Any, scaler: Any | None = None) -> None:
    import torch

    head = _classification_linear(target_model, 2)
    if head is None:
        raise RuntimeError("Could not find the two-class linear head in the exported model.")
    coefficients = np.asarray(classifier.coef_[0], dtype=np.float32)
    intercept = float(classifier.intercept_[0])
    if scaler is not None:
        scale = np.asarray(scaler.scale_, dtype=np.float32)
        mean = np.asarray(scaler.mean_, dtype=np.float32)
        coefficients = coefficients / scale
        intercept -= float(np.dot(coefficients, mean))
    if head.in_features != len(coefficients):
        raise RuntimeError(
            f"Embedding/head mismatch: {len(coefficients)} features versus {head.in_features} head inputs."
        )
    with torch.no_grad():
        vector = torch.from_numpy(coefficients).to(dtype=head.weight.dtype)
        # Class 0 is AI. Symmetric logits make softmax(class 0) exactly equal
        # to sklearn's sigmoid probability for the positive AI class.
        head.weight[0].copy_(vector * 0.5)
        head.weight[1].copy_(vector * -0.5)
        if head.bias is not None:
            head.bias[0] = intercept * 0.5
            head.bias[1] = intercept * -0.5


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


if __name__ == "__main__":
    main()

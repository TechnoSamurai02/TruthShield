from __future__ import annotations

import argparse
import io
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an image classifier for TruthShield.")
    parser.add_argument("--data-dir", default="training/data/defactify_sample")
    parser.add_argument("--output-dir", default="training/models/truthshield-image-detector")
    parser.add_argument("--base-model", default="microsoft/resnet-50")
    parser.add_argument(
        "--detector-task",
        choices=("generation", "manipulation"),
        default="generation",
        help="Record the specialist task and require its positive training class.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--preprocess-max-dimension", type=int, default=384)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=250,
        help="Checkpoint/evaluation interval. Frequent checkpoints make free-GPU sessions resumable.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional short Hugging Face cache path, such as C:/hf. The user cache is used by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    if cache_dir is not None:
        (cache_dir / "datasets").mkdir(parents=True, exist_ok=True)
        (cache_dir / "hub").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))

    from datasets import load_dataset
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForImageClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Dataset folder not found: {data_dir}")

    dataset_kwargs = {"cache_dir": str(cache_dir / "datasets")} if cache_dir is not None else {}
    data_files = _training_data_files(data_dir)
    dataset = load_dataset("imagefolder", data_files=data_files, **dataset_kwargs)
    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.15, seed=42)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    labels = dataset["train"].features["label"].names
    required_label = "ai_manipulated" if args.detector_task == "manipulation" else "ai_generated"
    if required_label not in labels:
        raise SystemExit(
            f"The {args.detector_task} task requires a '{required_label}' class; found {labels}."
        )
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    print(f"Labels: {labels}", flush=True)

    if args.max_train_samples > 0:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_eval_samples > 0:
        dataset["validation"] = dataset["validation"].shuffle(seed=42).select(
            range(min(args.max_eval_samples, len(dataset["validation"])))
        )

    processor = AutoImageProcessor.from_pretrained(args.base_model)
    print(f"Loading base model: {args.base_model}", flush=True)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.config.truthshield_detector_task = args.detector_task
    model.config.truthshield_preprocess_max_dimension = max(224, args.preprocess_max_dimension)
    _reuse_matching_classifier_rows(
        AutoConfig=AutoConfig,
        AutoModelForImageClassification=AutoModelForImageClassification,
        base_model=args.base_model,
        target_model=model,
        target_label2id=label2id,
    )

    def train_transform(batch: Dict[str, Any]) -> Dict[str, Any]:
        images = [
            _resize_for_model_prep(image.convert("RGB"), max(224, args.preprocess_max_dimension))
            for image in batch["image"]
        ]
        if not args.no_augmentation:
            images = [_augment_image(image) for image in images]
        encoded = processor(images=images, return_tensors="pt")
        encoded["labels"] = batch["label"]
        return encoded

    def evaluation_transform(batch: Dict[str, Any]) -> Dict[str, Any]:
        images = [
            _resize_for_model_prep(image.convert("RGB"), max(224, args.preprocess_max_dimension))
            for image in batch["image"]
        ]
        encoded = processor(images=images, return_tensors="pt")
        encoded["labels"] = batch["label"]
        return encoded

    train_dataset = dataset["train"].with_transform(train_transform)
    evaluation_dataset = dataset["validation"].with_transform(evaluation_transform)
    test_dataset = dataset["test"].with_transform(evaluation_transform) if "test" in dataset else None

    training_args = _build_training_args(TrainingArguments, args)

    trainer = _build_trainer(
        Trainer=Trainer,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=evaluation_dataset,
        processor=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=max(1, args.early_stopping_patience))],
    )
    resume_checkpoint = _latest_checkpoint(Path(args.output_dir)) if args.resume else None
    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None)
    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    all_metrics: Dict[str, Any] = {"validation": validation_metrics}
    print(validation_metrics, flush=True)
    if test_dataset is not None:
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        all_metrics["test"] = test_metrics
        print(test_metrics, flush=True)

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "truthshield_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("Saved model.", flush=True)
    variable = "AI_MANIPULATION_DETECTOR_MODELS" if args.detector_task == "manipulation" else "AI_IMAGE_DETECTOR_MODELS"
    print(f"Set {variable}={Path(args.output_dir).resolve()}", flush=True)


def _build_training_args(TrainingArguments: Any, args: argparse.Namespace) -> Any:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.epochs,
        "save_strategy": "steps",
        "save_steps": max(1, args.save_steps),
        "eval_steps": max(1, args.save_steps),
        "logging_steps": 20,
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        # Optimize the weakest class as well as the majority classes. Plain
        # accuracy can select a checkpoint that improves camera photos while
        # regressing AI or edited-image recall.
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": [],
        "save_total_limit": 3,
    }
    if "seed" in parameters:
        kwargs["seed"] = args.seed
    if "data_seed" in parameters:
        kwargs["data_seed"] = args.seed
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "steps"
    else:
        print("Warning: this Transformers version does not expose an evaluation strategy argument.")
    return TrainingArguments(**kwargs)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint-*"):
            try:
                step = int(path.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if (path / "trainer_state.json").is_file():
                checkpoints.append((step, path))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def _training_data_files(data_dir: Path) -> Dict[str, List[str]]:
    """Load only train/tuning for v4; never expose calibration or locked_test to Trainer."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def files(folder: Path) -> List[str]:
        return [str(path) for path in sorted(folder.rglob("*")) if path.is_file() and path.suffix.lower() in extensions]

    train_files = files(data_dir / "train")
    validation_dir = data_dir / "tuning" if (data_dir / "tuning").is_dir() else data_dir / "validation"
    validation_files = files(validation_dir)
    result: Dict[str, List[str]] = {}
    if train_files:
        result["train"] = train_files
    if validation_files:
        result["validation"] = validation_files
    # Legacy datasets may have a normal test split. V4 uses locked_test, which
    # is deliberately not loaded here and is evaluated only by the gate script.
    test_files = files(data_dir / "test")
    if test_files and not (data_dir / "locked_test").exists():
        result["test"] = test_files
    if not result.get("train"):
        raise SystemExit(f"No training images were found below {data_dir / 'train'}")
    return result


def _reuse_matching_classifier_rows(
    AutoConfig: Any,
    AutoModelForImageClassification: Any,
    base_model: str,
    target_model: Any,
    target_label2id: Dict[str, int],
) -> None:
    """Reuse learned class rows when changing from three image classes to two video classes."""
    import torch

    try:
        source_config = AutoConfig.from_pretrained(base_model)
        configured_label2id = {
            str(label): int(index)
            for label, index in getattr(source_config, "label2id", {}).items()
        }
    except Exception:
        configured_label2id = {}
    if not set(configured_label2id) & set(target_label2id):
        return
    # Load the original weights only when its config contains semantic rows
    # that can actually be reused by the new classifier.
    try:
        source_model = AutoModelForImageClassification.from_pretrained(base_model)
    except Exception as exc:
        print(f"Could not reuse prior classifier rows: {str(exc)[:160]}", flush=True)
        return
    try:
        source_label2id = {
            str(label): int(index)
            for label, index in getattr(source_model.config, "label2id", {}).items()
        }
        shared = sorted(set(source_label2id) & set(target_label2id))
        if not shared:
            return
        source_head = _classification_linear(source_model, len(source_label2id))
        target_head = _classification_linear(target_model, len(target_label2id))
        if source_head is None or target_head is None:
            return
        with torch.no_grad():
            for label in shared:
                source_index = source_label2id[label]
                target_index = target_label2id[label]
                target_head.weight[target_index].copy_(source_head.weight[source_index])
                if target_head.bias is not None and source_head.bias is not None:
                    target_head.bias[target_index].copy_(source_head.bias[source_index])
        print(f"Reused learned classifier rows for: {shared}", flush=True)
    finally:
        del source_model


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


def _build_trainer(
    Trainer: Any,
    model: Any,
    training_args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    processor: Any,
    callbacks: list[Any] | None = None,
) -> Any:
    parameters = inspect.signature(Trainer.__init__).parameters
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks or [],
    }
    if "processing_class" in parameters:
        kwargs["processing_class"] = processor
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = processor
    else:
        print("Warning: this Transformers version does not expose processing_class/tokenizer on Trainer.")
    return Trainer(**kwargs)


def compute_metrics(eval_prediction: Any) -> Dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=1)
    accuracy = float((predictions == labels).mean())
    class_ids = sorted(set(int(value) for value in labels.tolist()))
    recalls = []
    f1_scores = []
    for class_id in class_ids:
        true_positive = int(np.sum((predictions == class_id) & (labels == class_id)))
        false_positive = int(np.sum((predictions == class_id) & (labels != class_id)))
        false_negative = int(np.sum((predictions != class_id) & (labels == class_id)))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall)
        f1_scores.append(f1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
    }


def _augment_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.45:
        width, height = image.size
        scale = random.uniform(0.82, 1.0)
        crop_width = max(1, int(width * scale))
        crop_height = max(1, int(height * scale))
        left = random.randint(0, max(0, width - crop_width))
        top = random.randint(0, max(0, height - crop_height))
        image = image.crop((left, top, left + crop_width, top + crop_height))
    if random.random() < 0.55:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.82, 1.18))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.82, 1.18))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.20:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 1.1)))
    if random.random() < 0.35:
        # Apply the same resolution degradation to both classes so the model
        # cannot win by memorizing that one generator family is lower-res.
        width, height = image.size
        scale = random.uniform(0.35, 0.85)
        down_width = max(32, int(width * scale))
        down_height = max(32, int(height * scale))
        image = image.resize((down_width, down_height), Image.Resampling.BILINEAR).resize(
            (width, height), Image.Resampling.BICUBIC
        )
    if random.random() < 0.18:
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=random.randint(80, 160), threshold=3))
    if random.random() < 0.45:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(52, 94))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return image


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


if __name__ == "__main__":
    main()

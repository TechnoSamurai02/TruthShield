from __future__ import annotations

import argparse
import io
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.media_manifest import read_manifest


MANIPULATED_LABELS = {"manipulated", "ai_manipulated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight pixel-mask manipulation localizer."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=441)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-tuning-samples", type=int, default=0)
    return parser.parse_args()


@dataclass(frozen=True)
class SegmentationExample:
    image_path: Path
    mask_path: Path | None
    label: int
    class_label: str
    relative_path: str
    generator_or_editor: str


class ManifestSegmentationDataset:
    def __init__(
        self,
        data_dir: Path,
        *,
        split: str,
        resolution: int,
        augment: bool,
        max_samples: int = 0,
        seed: int = 0,
    ) -> None:
        try:
            from torch.utils.data import Dataset
        except ImportError as exc:  # pragma: no cover - exercised by CLI setup
            raise SystemExit("PyTorch is required for localizer training.") from exc
        del Dataset
        self.data_dir = data_dir.resolve()
        self.split = split
        self.resolution = resolution
        self.augment = augment
        localization = {
            str(row.get("path") or ""): str(row.get("mask_path") or "")
            for row in read_manifest(self.data_dir / "localization.v4.jsonl")
            if str(row.get("split") or "") == split
        }
        examples: list[SegmentationExample] = []
        for row in read_manifest(self.data_dir / "manifest.v4.jsonl"):
            if str(row.get("split") or "") != split:
                continue
            relative_path = str(row.get("path") or "")
            image_path = self.data_dir / relative_path
            if not image_path.is_file():
                continue
            label = int(str(row.get("class_label") or "").lower() in MANIPULATED_LABELS)
            mask_value = localization.get(relative_path)
            mask_path = self.data_dir / mask_value if mask_value else None
            if label and (mask_path is None or not mask_path.is_file()):
                raise SystemExit(f"Positive example is missing its mask: {relative_path}")
            examples.append(
                SegmentationExample(
                    image_path=image_path,
                    mask_path=mask_path,
                    label=label,
                    class_label=str(row.get("class_label") or "unknown"),
                    relative_path=relative_path,
                    generator_or_editor=str(row.get("generator_or_editor") or "unknown"),
                )
            )
        examples.sort(key=lambda item: item.relative_path)
        if max_samples > 0 and len(examples) > max_samples:
            randomizer = random.Random(seed)
            selected = list(range(len(examples)))
            randomizer.shuffle(selected)
            examples = [examples[index] for index in sorted(selected[:max_samples])]
        if not examples:
            raise SystemExit(f"No examples found for split '{split}' below {self.data_dir}.")
        if not any(item.label for item in examples):
            raise SystemExit(f"Split '{split}' contains no manipulated examples.")
        if all(item.label for item in examples):
            raise SystemExit(f"Split '{split}' contains no negative examples.")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        from torchvision.transforms import functional as vision

        example = self.examples[index]
        with Image.open(example.image_path) as handle:
            image = handle.convert("RGB")
        if example.mask_path is not None:
            with Image.open(example.mask_path) as handle:
                mask = handle.convert("L")
        else:
            mask = Image.new("L", image.size, 0)
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        mask = mask.resize((self.resolution, self.resolution), Image.Resampling.NEAREST)
        if self.augment:
            image, mask = _augment_pair(image, mask)
        image_tensor = vision.pil_to_tensor(image).float().div_(255.0)
        image_tensor = vision.normalize(
            image_tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        mask_tensor = vision.pil_to_tensor(mask).float().div_(255.0).gt_(0.05).float()
        return {
            "pixel_values": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(example.label, dtype=torch.long),
            "class_label": example.class_label,
            "path": example.relative_path,
            "generator_or_editor": example.generator_or_editor,
        }


def main() -> None:
    args = parse_args()
    if args.resolution < 224 or args.resolution % 32:
        raise SystemExit("--resolution must be at least 224 and divisible by 32.")
    try:
        import torch
        import torch.nn.functional as functional
        from torch.cuda.amp import GradScaler, autocast
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from torchvision.models import MobileNet_V3_Large_Weights
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
    except ImportError as exc:
        raise SystemExit("Install training/requirements-train.txt before training.") from exc
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for localizer training.")

    _seed_everything(args.seed, torch)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = ManifestSegmentationDataset(
        data_dir,
        split="train",
        resolution=args.resolution,
        augment=True,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    tuning_dataset = ManifestSegmentationDataset(
        data_dir,
        split="tuning",
        resolution=args.resolution,
        augment=False,
        max_samples=args.max_tuning_samples,
        seed=args.seed + 1,
    )
    sample_weights = _balanced_sample_weights([item.label for item in train_dataset.examples])
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": max(0, args.workers),
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    tuning_loader = DataLoader(tuning_dataset, shuffle=False, **loader_kwargs)

    model = lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        num_classes=1,
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.learning_rate * 0.05,
    )
    scaler = GradScaler(enabled=True)
    start_epoch = 0
    best_score = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    last_path = output_dir / "last.pt"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = list(checkpoint.get("history", []))
        print(f"Resuming after epoch {start_epoch}", flush=True)

    print(f"Training on: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"Train records: {len(train_dataset)} | tuning records: {len(tuning_dataset)}",
        flush=True,
    )
    patience = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            images = batch["pixel_values"].cuda(non_blocking=True)
            masks = batch["mask"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                logits = model(images)["out"]
                logits = functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = _segmentation_loss(logits, masks, functional=functional, torch=torch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            if step % 25 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={running_loss / step:.4f}",
                    flush=True,
                )
        scheduler.step()
        validation = evaluate_loader(model, tuning_loader, torch=torch)
        validation["training_loss"] = running_loss / max(1, len(train_loader))
        validation["epoch"] = epoch + 1
        selection_score = (
            0.65 * float(validation["image_average_precision"])
            + 0.35 * float(validation["pixel_f1"])
        )
        validation["selection_score"] = selection_score
        history.append(validation)
        improved = selection_score > best_score
        if improved:
            best_score = selection_score
            best_epoch = epoch + 1
            patience = 0
            _save_model_state(output_dir / "model.pt", model, args=args, metrics=validation, torch=torch)
        else:
            patience += 1
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "history": history,
            "config": _config_dict(args),
        }
        _atomic_torch_save(checkpoint, last_path, torch=torch)
        print(json.dumps(validation, indent=2, sort_keys=True), flush=True)
        if patience >= max(1, args.early_stopping_patience):
            print(f"Early stopping after {patience} non-improving epochs.", flush=True)
            break

    best_path = output_dir / "model.pt"
    if not best_path.is_file():
        raise SystemExit("Training produced no best model checkpoint.")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    model.cuda().eval()
    _export_torchscript(model, output_dir / "model.ts", args.resolution, torch=torch)
    report = {
        "model_type": "torchvision-lraspp-mobilenet-v3-large",
        "detector_task": "manipulation-localization",
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_tuning_metrics": best.get("metrics", {}),
        "history": history,
        "config": _config_dict(args),
        "license": {
            "implementation": "BSD-3-Clause (torchvision)",
            "training_data": str(data_dir / "licenses.v4.json"),
        },
    }
    (output_dir / "truthshield_localizer_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "preprocess.json").write_text(
        json.dumps(
            {
                "resolution": args.resolution,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "image_score": "mean of highest 1 percent of manipulation probabilities",
                "pixel_threshold_requires_calibration": True,
                "model_output": "single-channel manipulation logits",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Best localizer saved at: {output_dir}", flush=True)
    print("This remains a candidate until calibration and locked promotion gates pass.", flush=True)


def evaluate_loader(model: Any, loader: Any, *, torch: Any) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    image_scores: list[float] = []
    image_labels: list[int] = []
    true_positive = false_positive = false_negative = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["pixel_values"].cuda(non_blocking=True)
            masks = batch["mask"].cuda(non_blocking=True)
            probabilities = torch.sigmoid(model(images)["out"])
            probabilities = torch.nn.functional.interpolate(
                probabilities,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for probability, mask, label in zip(probabilities, masks, batch["label"]):
                image_scores.append(_image_score(probability.detach().cpu().numpy()))
                image_labels.append(int(label))
                predicted = probability >= 0.5
                actual = mask >= 0.5
                true_positive += int(torch.logical_and(predicted, actual).sum().item())
                false_positive += int(torch.logical_and(predicted, ~actual).sum().item())
                false_negative += int(torch.logical_and(~predicted, actual).sum().item())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    pixel_f1 = 2 * precision * recall / max(1e-12, precision + recall)
    labels = np.asarray(image_labels, dtype=np.int64)
    scores = np.asarray(image_scores, dtype=np.float64)
    return {
        "records": int(len(labels)),
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(pixel_f1),
        "pixel_iou": float(true_positive / max(1, true_positive + false_positive + false_negative)),
        "image_roc_auc": float(roc_auc_score(labels, scores)),
        "image_average_precision": float(average_precision_score(labels, scores)),
    }


def _segmentation_loss(logits: Any, masks: Any, *, functional: Any, torch: Any) -> Any:
    positives = masks.sum()
    negatives = masks.numel() - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), 1.0, 20.0)
    bce = functional.binary_cross_entropy_with_logits(logits, masks, pos_weight=pos_weight)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * masks).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return 0.60 * bce + 0.40 * dice_loss


def _image_score(probability: np.ndarray, top_fraction: float = 0.01) -> float:
    flattened = np.asarray(probability, dtype=np.float32).reshape(-1)
    if not len(flattened):
        return 0.0
    count = max(1, int(round(len(flattened) * top_fraction)))
    start = max(0, len(flattened) - count)
    return float(np.partition(flattened, start)[start:].mean())


def _balanced_sample_weights(labels: list[int]) -> list[float]:
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        raise ValueError("Balanced sampling requires both positive and negative records.")
    return [1.0 / positive if label else 1.0 / negative for label in labels]


def _augment_pair(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    if random.random() < 0.35:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.35:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.35:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(55, 95))
        buffer.seek(0)
        with Image.open(buffer) as handle:
            image = handle.convert("RGB")
    return image, mask


def _save_model_state(path: Path, model: Any, *, args: argparse.Namespace, metrics: dict[str, Any], torch: Any) -> None:
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "config": _config_dict(args),
            "metrics": metrics,
        },
        path,
        torch=torch,
    )


def _atomic_torch_save(value: Any, path: Path, *, torch: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def _export_torchscript(model: Any, path: Path, resolution: int, *, torch: Any) -> None:
    class OutputWrapper(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixel_values: Any) -> Any:
            return self.wrapped(pixel_values)["out"]

    wrapper = OutputWrapper(model).eval()
    example = torch.zeros((1, 3, resolution, resolution), device="cuda")
    traced = torch.jit.trace(wrapper, example, strict=True)
    traced = torch.jit.freeze(traced)
    torch.jit.save(traced.cpu(), str(path))


def _config_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "resolution": args.resolution,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "model_type": "torchvision-lraspp-mobilenet-v3-large",
        "task": "manipulation-localization",
    }


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()

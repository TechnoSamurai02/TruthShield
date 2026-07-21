from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.media_manifest import MediaRecord, read_manifest, sha256_file


AUTHENTIC_LABELS = {"authentic", "real", "real_camera"}
GENERATED_LABELS = {"generated", "ai_generated"}
SUPPORTED_SPLITS = ("train", "tuning", "calibration", "locked_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paired authentic/manipulated images, localized masks, and a "
            "split-safe TruthShield v4 manifest from an existing image manifest."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=0,
        help="Maximum authentic parent images per split. Zero keeps every eligible parent.",
    )
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove only generated split/mask folders below output-dir before rebuilding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_manifest = source_dir / "manifest.v4.jsonl"
    if not source_manifest.is_file():
        raise SystemExit(f"Source manifest not found: {source_manifest}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for name in (*SUPPORTED_SPLITS, "localization_masks"):
            target = output_dir / name
            if target.is_dir():
                shutil.rmtree(target)

    rows = [dict(row) for row in read_manifest(source_manifest)]
    authentic_by_split = {
        split: sorted(
            (
                row
                for row in rows
                if str(row.get("split") or "") == split
                and str(row.get("class_label") or "").lower() in AUTHENTIC_LABELS
                and (source_dir / str(row.get("path") or "")).is_file()
            ),
            key=lambda row: str(row.get("path") or ""),
        )
        for split in SUPPORTED_SPLITS
    }
    generated_by_split = {
        split: sorted(
            (
                row
                for row in rows
                if str(row.get("split") or "") == split
                and str(row.get("class_label") or "").lower() in GENERATED_LABELS
                and (source_dir / str(row.get("path") or "")).is_file()
            ),
            key=lambda row: str(row.get("path") or ""),
        )
        for split in SUPPORTED_SPLITS
    }

    manifest_rows: list[dict[str, Any]] = []
    localization_rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {}
    for split, split_rows in authentic_by_split.items():
        if args.max_per_split > 0:
            split_rows = split_rows[: args.max_per_split]
        generated_rows = generated_by_split[split]
        if args.max_per_split > 0:
            generated_rows = generated_rows[: args.max_per_split]
        summary[split] = {
            "authentic_parents": 0,
            "generated_parents": 0,
            "authentic_examples": 0,
            "generated_examples": 0,
            "manipulated_examples": 0,
        }

        for index, row in enumerate(split_rows):
            source_path = source_dir / str(row["path"])
            donor_row = split_rows[(index + 1) % len(split_rows)] if len(split_rows) > 1 else row
            donor_path = source_dir / str(donor_row["path"])
            try:
                with Image.open(source_path) as image:
                    original = image.convert("RGB")
                with Image.open(donor_path) as image:
                    donor = image.convert("RGB")
            except Exception as exc:
                print(f"Skipped {source_path}: {exc}")
                continue
            if min(original.size) < 96:
                print(f"Skipped undersized image: {source_path}")
                continue

            randomizer = random.Random(_record_seed(args.seed, split, str(row["path"])))
            manipulated, mask, editor = _make_manipulation(
                original,
                donor,
                split=split,
                index=index,
                randomizer=randomizer,
            )
            bounds = mask.getbbox()
            if bounds is None:
                print(f"Skipped empty manipulation mask: {source_path}")
                continue
            patch_box = _expanded_box(bounds, original.size, padding_ratio=0.30)
            original_patch = original.crop(patch_box)
            manipulated_patch = manipulated.crop(patch_box)
            patch_mask = mask.crop(patch_box)

            base_name = f"{split}_{index:06d}"
            authentic_dir = output_dir / split / "real_camera"
            manipulated_dir = output_dir / split / "ai_manipulated"
            mask_dir = output_dir / "localization_masks" / split
            for directory in (authentic_dir, manipulated_dir, mask_dir):
                directory.mkdir(parents=True, exist_ok=True)

            outputs = (
                (original, authentic_dir / f"{base_name}_original.jpg", "real_camera", "paired_original_copy", None),
                (original_patch, authentic_dir / f"{base_name}_patch_real.jpg", "real_camera", "paired_original_patch", None),
                (manipulated, manipulated_dir / f"{base_name}_manipulated.jpg", "ai_manipulated", editor, mask),
                (
                    manipulated_patch,
                    manipulated_dir / f"{base_name}_patch_manipulated.jpg",
                    "ai_manipulated",
                    f"{editor}:localized_patch",
                    patch_mask,
                ),
            )
            group = str(row.get("source_group") or _stable_group(str(row["path"])))
            for media, path, class_label, transformation, media_mask in outputs:
                media.save(path, format="JPEG", quality=92)
                mask_path: Path | None = None
                if media_mask is not None:
                    mask_path = mask_dir / f"{path.stem}.png"
                    media_mask.save(mask_path, format="PNG")
                record = MediaRecord(
                    path=path.relative_to(output_dir).as_posix(),
                    sha256=sha256_file(path),
                    media_type="image",
                    class_label=class_label,
                    source=f"{row.get('source')}; locally-derived-paired-edit",
                    license=str(row.get("license") or "license-review-required"),
                    generator_or_editor="authentic" if class_label == "real_camera" else editor,
                    parent_media=str(row.get("path") or "") if class_label == "ai_manipulated" else None,
                    transformation=transformation,
                    semantic_category=str(row.get("semantic_category") or "unspecified"),
                    source_group=group,
                    split=split,
                )
                rendered = asdict(record)
                manifest_rows.append(rendered)
                if mask_path is not None:
                    localized_bounds = media_mask.getbbox() or (0, 0, media.width, media.height)
                    localization_rows.append(
                        {
                            **rendered,
                            "mask_path": mask_path.relative_to(output_dir).as_posix(),
                            "suspicious_region": list(localized_bounds),
                            "mask_coverage": round(_mask_coverage(media_mask), 6),
                        }
                    )

            summary[split]["authentic_parents"] += 1
            summary[split]["authentic_examples"] += 2
            summary[split]["manipulated_examples"] += 2
            if summary[split]["authentic_parents"] % 50 == 0:
                print(f"{split}: prepared {summary[split]['authentic_parents']} paired authentic parents")

        # Fully generated media is a separate negative class for the
        # manipulation specialist. Without it, the model can confuse a wholly
        # synthetic image with a localized edit and steal precedence from the
        # generation specialist.
        for index, row in enumerate(generated_rows):
            source_path = source_dir / str(row["path"])
            try:
                with Image.open(source_path) as image:
                    generated = image.convert("RGB")
            except Exception as exc:
                print(f"Skipped {source_path}: {exc}")
                continue
            if min(generated.size) < 96:
                print(f"Skipped undersized image: {source_path}")
                continue

            randomizer = random.Random(_record_seed(args.seed, split, str(row["path"])))
            generated_patch = generated.crop(_random_patch_box(generated.size, randomizer))
            generated_dir = output_dir / split / "ai_generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            base_name = f"{split}_{index:06d}"
            outputs = (
                (generated, generated_dir / f"{base_name}_generated.jpg", "generated_reference_copy"),
                (
                    generated_patch,
                    generated_dir / f"{base_name}_patch_generated.jpg",
                    "generated_reference_patch",
                ),
            )
            for media, path, transformation in outputs:
                media.save(path, format="JPEG", quality=92)
                manifest_rows.append(
                    asdict(
                        MediaRecord(
                            path=path.relative_to(output_dir).as_posix(),
                            sha256=sha256_file(path),
                            media_type="image",
                            class_label="ai_generated",
                            source=f"{row.get('source')}; locally-resaved-generation-reference",
                            license=str(row.get("license") or "license-review-required"),
                            generator_or_editor=str(row.get("generator_or_editor") or "unknown-generator"),
                            parent_media=str(row.get("path") or ""),
                            transformation=transformation,
                            semantic_category=str(row.get("semantic_category") or "unspecified"),
                            source_group=str(row.get("source_group") or _stable_group(str(row["path"]))),
                            split=split,
                        )
                    )
                )
            summary[split]["generated_parents"] += 1
            summary[split]["generated_examples"] += 2
            if summary[split]["generated_parents"] % 50 == 0:
                print(f"{split}: prepared {summary[split]['generated_parents']} generated references")

    _write_dict_jsonl(output_dir / "manifest.v4.jsonl", manifest_rows)
    _write_dict_jsonl(output_dir / "localization.v4.jsonl", localization_rows)
    (output_dir / "preparation-report.json").write_text(
        json.dumps(
            {
                "source_manifest": str(source_manifest),
                "seed": args.seed,
                "split_editors": _split_editor_summary(),
                "summary": summary,
                "manifest_records": len(manifest_rows),
                "localized_records": len(localization_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {len(manifest_rows)} media records and {len(localization_rows)} localized records.")
    print(f"Done. Paired manipulation dataset is at: {output_dir}")


def _make_manipulation(
    original: Image.Image,
    donor: Image.Image,
    *,
    split: str,
    index: int,
    randomizer: random.Random,
) -> tuple[Image.Image, Image.Image, str]:
    if split == "train":
        if index % 2 == 0:
            edited, mask = _copy_move(original, randomizer)
            return edited, mask, "local-copy-move-train-v1"
        edited, mask = _splice(original, donor, randomizer, feather=True)
        return edited, mask, "local-alpha-splice-train-v1"
    if split == "tuning":
        edited, mask = _inpaint(original, randomizer)
        return edited, mask, "local-opencv-inpaint-tuning-v1"
    if split == "calibration":
        edited, mask = _splice(original, donor, randomizer, feather=True, color_match=True)
        return edited, mask, "local-feathered-replacement-calibration-v1"
    edited, mask = _splice(original, donor, randomizer, feather=False)
    return edited, mask, "local-hard-splice-locked-v1"


def _copy_move(original: Image.Image, randomizer: random.Random) -> tuple[Image.Image, Image.Image]:
    source_box, target_box = _source_and_target_boxes(original.size, randomizer)
    patch = original.crop(source_box)
    edited = original.copy()
    alpha = _feather_mask(patch.size, radius=max(2, min(patch.size) // 18))
    edited.paste(patch, target_box[:2], alpha)
    mask = Image.new("L", original.size, 0)
    mask.paste(alpha, target_box[:2])
    return edited, mask


def _splice(
    original: Image.Image,
    donor: Image.Image,
    randomizer: random.Random,
    *,
    feather: bool,
    color_match: bool = False,
) -> tuple[Image.Image, Image.Image]:
    _, target_box = _source_and_target_boxes(original.size, randomizer)
    target_width = target_box[2] - target_box[0]
    target_height = target_box[3] - target_box[1]
    donor = donor.convert("RGB")
    donor_box, _ = _source_and_target_boxes(donor.size, randomizer, target_size=(target_width, target_height))
    patch = donor.crop(donor_box).resize((target_width, target_height), Image.Resampling.LANCZOS)
    if color_match:
        destination = original.crop(target_box)
        patch = _match_mean_color(patch, destination)
    alpha = (
        _feather_mask(patch.size, radius=max(2, min(patch.size) // 16))
        if feather
        else Image.new("L", patch.size, 255)
    )
    edited = original.copy()
    edited.paste(patch, target_box[:2], alpha)
    mask = Image.new("L", original.size, 0)
    mask.paste(alpha, target_box[:2])
    return edited, mask


def _inpaint(original: Image.Image, randomizer: random.Random) -> tuple[Image.Image, Image.Image]:
    _, target_box = _source_and_target_boxes(original.size, randomizer)
    mask_array = np.zeros((original.height, original.width), dtype=np.uint8)
    cv2.ellipse(
        mask_array,
        (
            (target_box[0] + target_box[2]) // 2,
            (target_box[1] + target_box[3]) // 2,
        ),
        (
            max(2, (target_box[2] - target_box[0]) // 2),
            max(2, (target_box[3] - target_box[1]) // 2),
        ),
        0,
        0,
        360,
        255,
        -1,
    )
    rgb = np.asarray(original.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask_array, 5, cv2.INPAINT_TELEA)
    edited = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    return edited, Image.fromarray(mask_array, mode="L")


def _source_and_target_boxes(
    size: tuple[int, int],
    randomizer: random.Random,
    target_size: tuple[int, int] | None = None,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    width, height = size
    if target_size is None:
        patch_width = max(48, round(width * randomizer.uniform(0.18, 0.34)))
        patch_height = max(48, round(height * randomizer.uniform(0.18, 0.34)))
    else:
        patch_width, patch_height = target_size
    patch_width = min(patch_width, max(32, width - 2))
    patch_height = min(patch_height, max(32, height - 2))

    def box() -> tuple[int, int, int, int]:
        x = randomizer.randint(0, max(0, width - patch_width))
        y = randomizer.randint(0, max(0, height - patch_height))
        return (x, y, x + patch_width, y + patch_height)

    source = box()
    target = box()
    for _ in range(12):
        if _intersection_over_union(source, target) < 0.20:
            break
        target = box()
    return source, target


def _random_patch_box(
    size: tuple[int, int],
    randomizer: random.Random,
) -> tuple[int, int, int, int]:
    width, height = size
    crop_width = max(64, round(width * randomizer.uniform(0.42, 0.70)))
    crop_height = max(64, round(height * randomizer.uniform(0.42, 0.70)))
    crop_width = min(width, crop_width)
    crop_height = min(height, crop_height)
    left = randomizer.randint(0, max(0, width - crop_width))
    top = randomizer.randint(0, max(0, height - crop_height))
    return left, top, left + crop_width, top + crop_height


def _feather_mask(size: tuple[int, int], radius: int) -> Image.Image:
    width, height = size
    inset = max(1, min(width, height) // 18)
    mask = Image.new("L", size, 0)
    core = Image.new("L", (max(1, width - inset * 2), max(1, height - inset * 2)), 255)
    mask.paste(core, (inset, inset))
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def _match_mean_color(patch: Image.Image, destination: Image.Image) -> Image.Image:
    patch_array = np.asarray(patch, dtype=np.float32)
    destination_array = np.asarray(destination.resize(patch.size), dtype=np.float32)
    patch_mean = patch_array.reshape(-1, 3).mean(axis=0)
    destination_mean = destination_array.reshape(-1, 3).mean(axis=0)
    adjusted = np.clip(patch_array + destination_mean - patch_mean, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted, mode="RGB")


def _expanded_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    width, height = image_size
    padding_x = round((right - left) * padding_ratio)
    padding_y = round((bottom - top) * padding_ratio)
    return (
        max(0, left - padding_x),
        max(0, top - padding_y),
        min(width, right + padding_x),
        min(height, bottom + padding_y),
    )


def _intersection_over_union(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def _mask_coverage(mask: Image.Image) -> float:
    values = np.asarray(mask, dtype=np.float32) / 255.0
    return float(values.mean())


def _record_seed(seed: int, split: str, path: str) -> int:
    value = hashlib.sha256(f"{seed}:{split}:{path}".encode("utf-8")).hexdigest()[:16]
    return int(value, 16)


def _stable_group(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _split_editor_summary() -> dict[str, list[str]]:
    return {
        "train": ["local-copy-move-train-v1", "local-alpha-splice-train-v1"],
        "tuning": ["local-opencv-inpaint-tuning-v1"],
        "calibration": ["local-feathered-replacement-calibration-v1"],
        "locked_test": ["local-hard-splice-locked-v1"],
    }


def _write_dict_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

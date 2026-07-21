from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.media_manifest import MediaRecord, sha256_file, stable_group, write_jsonl


LABEL_TO_FOLDER = {
    0: "real_camera",
    1: "ai_generated",
}

GENERATOR_NAMES = {
    0: "authentic",
    1: "stable-diffusion-2.1",
    2: "stable-diffusion-xl",
    3: "stable-diffusion-3",
    4: "dall-e-3",
    5: "midjourney-6",
}

V4_GENERATOR_SPLITS = {
    1: "train",
    2: "train",
    3: "tuning",
    4: "calibration",
    5: "locked_test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a balanced local sample from the Defactify real-vs-AI image dataset."
    )
    parser.add_argument(
        "--dataset-name",
        default="Rajarshi-Roy-research/Defactify_Image_Dataset",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--output-dir",
        default="training/data/defactify_sample",
        help="Where to write ImageFolder-style files.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=800,
        help="Maximum images per label per split. Start small on a laptop.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to export.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the selected output split folders before exporting. Use this when changing sample sizes.",
    )
    parser.add_argument(
        "--split-policy",
        choices=("source", "generator-heldout-v4"),
        default="generator-heldout-v4",
        help="Use generator-separated v4 splits or retain the dataset's source split.",
    )
    parser.add_argument(
        "--dataset-license",
        default="license-review-required",
        help="License identifier recorded in the manifest. Confirm it before public redistribution.",
    )
    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_counts: Dict[tuple[int, str], int] = {}

    if args.clean_output:
        target_names = set(args.splits)
        if args.split_policy == "generator-heldout-v4":
            target_names.update({"train", "tuning", "calibration", "locked_test"})
        for target_name in target_names:
            target_dir = output_dir / target_name
            if target_dir.exists():
                shutil.rmtree(target_dir)

    manifest_records: list[MediaRecord] = []
    for split in args.splits:
        print(f"Preparing split: {split}")
        split_dir = output_dir / split
        dataset = load_dataset(args.dataset_name, split=split, streaming=True)
        counts: Dict[int, int] = {label: 0 for label in LABEL_TO_FOLDER}

        for example in dataset:
            raw_label = example.get("Label_A")
            if raw_label not in LABEL_TO_FOLDER:
                continue
            raw_generator = _integer_label(example.get("Label_B"), default=0 if raw_label == 0 else -1)
            image = example.get("Image")
            if image is None:
                continue
            caption = _first_text(example, "Caption", "caption", "Prompt", "prompt", "Text", "text")
            source_id = _first_text(example, "id", "ID", "source_id", "filename", "File_Name")
            pixel_group = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            source_group_key = caption or source_id or pixel_group
            target_split = _target_split(
                source_split=split,
                raw_label=raw_label,
                raw_generator=raw_generator,
                source_group_key=source_group_key,
                split_policy=args.split_policy,
            )
            if target_split is None:
                continue
            count_key = (raw_label, target_split)
            target_count = target_counts.get(count_key, 0)
            if target_count >= args.max_per_label:
                continue

            folder = output_dir / target_split / LABEL_TO_FOLDER[raw_label]
            folder.mkdir(parents=True, exist_ok=True)
            index = target_count
            path = folder / f"{target_split}_{LABEL_TO_FOLDER[raw_label]}_{index:06d}.jpg"
            try:
                image.convert("RGB").save(path, format="JPEG", quality=92)
            except Exception as exc:
                print(f"Skipped one image: {exc}")
                continue

            counts[raw_label] += 1
            target_counts[count_key] = target_count + 1
            group = stable_group(source_group_key)
            manifest_records.append(
                MediaRecord(
                    path=path.relative_to(output_dir).as_posix(),
                    sha256=sha256_file(path),
                    media_type="image",
                    class_label=LABEL_TO_FOLDER[raw_label],
                    source=args.dataset_name,
                    license=args.dataset_license,
                    generator_or_editor=GENERATOR_NAMES.get(raw_generator, f"label-b-{raw_generator}"),
                    parent_media=None,
                    transformation="jpeg_quality_92_export",
                    semantic_category="unspecified",
                    source_group=group,
                    split=target_split,
                )
            )
            if sum(counts.values()) % 100 == 0:
                print(f"  saved {counts}")

        print(f"Finished {split}: {counts}")

    write_jsonl(output_dir / "manifest.v4.jsonl", manifest_records)
    print(f"Wrote {len(manifest_records)} manifest records with Label_B generator identities.")
    print(f"Done. Dataset is at: {output_dir}")


def _integer_label(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_text(example: dict, *keys: str) -> str:
    for key in keys:
        value = example.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _target_split(
    *,
    source_split: str,
    raw_label: int,
    raw_generator: int,
    source_group_key: str,
    split_policy: str,
) -> str | None:
    if split_policy == "source":
        return source_split
    group_split = _group_split(source_group_key)
    if raw_label == 1:
        generator_split = V4_GENERATOR_SPLITS.get(raw_generator)
        # Defactify repeats captions across every generator. Retain a generated
        # variant only when its held-out family belongs to the caption's one
        # assigned split; otherwise the same semantic source would leak across
        # train, tuning, calibration, and locked test.
        return generator_split if generator_split == group_split else None
    return group_split


def _group_split(source_group_key: str) -> str:
    bucket = int(hashlib.sha256(source_group_key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "tuning"
    if bucket < 90:
        return "calibration"
    return "locked_test"


if __name__ == "__main__":
    main()

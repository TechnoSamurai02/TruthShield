from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict


LABEL_TO_FOLDER = {
    0: "real_camera",
    1: "ai_generated",
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
    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"Preparing split: {split}")
        split_dir = output_dir / split
        if args.clean_output and split_dir.exists():
            shutil.rmtree(split_dir)
        dataset = load_dataset(args.dataset_name, split=split, streaming=True)
        counts: Dict[int, int] = {label: 0 for label in LABEL_TO_FOLDER}

        for example in dataset:
            raw_label = example.get("Label_A")
            if raw_label not in LABEL_TO_FOLDER:
                continue
            if counts[raw_label] >= args.max_per_label:
                if all(count >= args.max_per_label for count in counts.values()):
                    break
                continue

            image = example.get("Image")
            if image is None:
                continue

            folder = output_dir / split / LABEL_TO_FOLDER[raw_label]
            folder.mkdir(parents=True, exist_ok=True)
            index = counts[raw_label]
            path = folder / f"{split}_{LABEL_TO_FOLDER[raw_label]}_{index:06d}.jpg"
            try:
                image.convert("RGB").save(path, format="JPEG", quality=92)
            except Exception as exc:
                print(f"Skipped one image: {exc}")
                continue

            counts[raw_label] += 1
            if sum(counts.values()) % 100 == 0:
                print(f"  saved {counts}")

        print(f"Finished {split}: {counts}")

    print(f"Done. Dataset is at: {output_dir}")


if __name__ == "__main__":
    main()

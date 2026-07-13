from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CAPTIONS = [
    "YES",
    "NO WAY",
    "BREAKING",
    "WOW",
    "REAL?",
    "LOOK AT THIS",
    "UNBELIEVABLE",
    "SHARE THIS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create real-but-captioned training examples from real_camera images."
    )
    parser.add_argument(
        "--dataset-dir",
        default="training/data/defactify_sample",
        help="ImageFolder dataset directory containing train/validation/test folders.",
    )
    parser.add_argument("--max-per-split", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean-generated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete caption variants previously created by this script before rebuilding them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    dataset_dir = Path(args.dataset_dir)

    for split in ["train", "validation", "test"]:
        source_dir = dataset_dir / split / "real_camera"
        target_dir = dataset_dir / split / "real_edited_or_captioned"
        if not source_dir.exists():
            print(f"Skipping missing folder: {source_dir}")
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        if args.clean_generated:
            for generated in target_dir.glob("*_captioned_*.jpg"):
                generated.unlink()

        files = [
            path
            for path in source_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        random.shuffle(files)
        for index, path in enumerate(files[: args.max_per_split]):
            try:
                image = Image.open(path).convert("RGB")
            except Exception as exc:
                print(f"Skipped {path.name}: {exc}")
                continue
            captioned = add_caption(image, random.choice(CAPTIONS))
            captioned.save(target_dir / f"{path.stem}_captioned_{index:05d}.jpg", quality=90)

        print(f"Created captioned variants for {split}: {min(len(files), args.max_per_split)}")


def add_caption(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font_size = max(18, min(width, height) // 8)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=3)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = max(8, (width - text_width) // 2)
    y = random.choice([max(8, height // 18), max(8, height - text_height - height // 12)])

    if random.random() < 0.35:
        padding = max(8, font_size // 4)
        draw.rectangle(
            [0, max(0, y - padding), width, min(height, y + text_height + padding)],
            fill=(255, 255, 255),
        )
        fill = (0, 0, 0)
        stroke_fill = (255, 255, 255)
    else:
        fill = random.choice([(255, 255, 255), (0, 0, 0), (255, 230, 0)])
        stroke_fill = (0, 0, 0) if fill != (0, 0, 0) else (255, 255, 255)

    draw.text((x, y), text, font=font, fill=fill, stroke_width=3, stroke_fill=stroke_fill)
    return image


if __name__ == "__main__":
    main()

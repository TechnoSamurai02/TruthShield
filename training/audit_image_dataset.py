from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find duplicate images and split leakage in an ImageFolder dataset.")
    parser.add_argument("--data-dir", default="training/data/defactify_sample")
    parser.add_argument("--output", default="training/data/image_dataset_audit.json")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Hash workers. Keep at 1 on synchronized/OneDrive folders to avoid I/O contention.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    files = [path for path in sorted(data_dir.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise SystemExit(f"No images found below {data_dir}")

    exact: DefaultDict[str, List[str]] = defaultdict(list)
    pixel: DefaultDict[str, List[str]] = defaultdict(list)
    perceptual: DefaultDict[str, List[str]] = defaultdict(list)
    perceptual_entries: List[Tuple[int, str]] = []
    counts: DefaultDict[str, int] = defaultdict(int)
    failures: List[Dict[str, str]] = []
    worker = partial(_fingerprint_file, data_dir=data_dir)
    with ThreadPoolExecutor(max_workers=max(1, args.workers), thread_name_prefix="truthshield-audit") as executor:
        results = executor.map(worker, files)
        for index, result in enumerate(results, start=1):
            relative = result["relative"]
            counts[f"{result['split']}/{result['label']}"] += 1
            if result.get("error"):
                failures.append({"path": relative, "error": str(result["error"])[:200]})
            else:
                exact[str(result["exact"])].append(relative)
                pixel[str(result["pixel"])].append(relative)
                phash = str(result["phash"])
                perceptual[phash].append(relative)
                perceptual_entries.append((int(phash, 16), relative))
            if index % 1000 == 0 or index == len(files):
                print(f"  audited {index}/{len(files)}", flush=True)

    report = {
        "data_dir": str(data_dir.resolve()),
        "file_count": len(files),
        "counts": dict(sorted(counts.items())),
        "cross_split_exact_duplicates": _cross_split_groups(exact),
        "cross_split_normalized_pixel_duplicates": _cross_split_groups(pixel),
        "cross_split_perceptual_hash_collisions": _cross_split_groups(perceptual),
        "cross_split_perceptual_near_matches": _perceptual_near_matches(perceptual_entries),
        "within_split_exact_duplicate_groups": _within_split_group_count(exact),
        "decode_failures": failures,
        "notes": [
            "Exact and normalized-pixel duplicates across splits are strong leakage evidence.",
            "Perceptual-hash collisions are clues only and need visual inspection because unrelated images can collide.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "file_count": report["file_count"],
                "cross_split_exact_duplicate_groups": len(report["cross_split_exact_duplicates"]),
                "cross_split_normalized_pixel_duplicate_groups": len(report["cross_split_normalized_pixel_duplicates"]),
                "cross_split_perceptual_hash_collision_groups": len(report["cross_split_perceptual_hash_collisions"]),
                "cross_split_perceptual_near_matches": len(report["cross_split_perceptual_near_matches"]),
                "decode_failures": len(failures),
            },
            indent=2,
        )
    )
    print(f"Full report: {output.resolve()}", flush=True)


def _phash(image: Image.Image) -> str:
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coefficients = cv2.dct(gray)[:8, :8]
    values = coefficients.flatten()
    threshold = float(np.median(values[1:]))
    bits = values >= threshold
    return f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"


def _fingerprint_file(path: Path, data_dir: Path) -> Dict[str, str]:
    relative_path = path.relative_to(data_dir)
    relative = str(relative_path)
    parts = relative_path.parts
    result = {
        "relative": relative,
        "split": parts[0] if len(parts) > 0 else "unknown",
        "label": parts[1] if len(parts) > 1 else "unknown",
    }
    try:
        raw = path.read_bytes()
        result["exact"] = hashlib.sha256(raw).hexdigest()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            normalized = rgb.resize((256, 256), Image.Resampling.LANCZOS)
            result["pixel"] = hashlib.sha256(np.asarray(normalized).tobytes()).hexdigest()
            result["phash"] = _phash(rgb)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _split_for(path: str) -> str:
    return Path(path).parts[0] if Path(path).parts else "unknown"


def _cross_split_groups(groups: Dict[str, List[str]], limit: int = 200) -> List[Dict[str, object]]:
    results = []
    for fingerprint, paths in groups.items():
        splits = sorted({_split_for(path) for path in paths})
        if len(splits) <= 1:
            continue
        results.append({"fingerprint": fingerprint, "splits": splits, "paths": paths[:20], "count": len(paths)})
        if len(results) >= limit:
            break
    return results


def _within_split_group_count(groups: Dict[str, List[str]]) -> int:
    count = 0
    for paths in groups.values():
        split_counts: DefaultDict[str, int] = defaultdict(int)
        for path in paths:
            split_counts[_split_for(path)] += 1
        count += sum(value > 1 for value in split_counts.values())
    return count


def _perceptual_near_matches(
    entries: List[Tuple[int, str]],
    maximum_hamming_distance: int = 4,
    limit: int = 500,
) -> List[Dict[str, object]]:
    # Eight independent 8-bit indexes avoid an O(n^2) scan. With at most four
    # changed bits, at least one chunk must still match exactly.
    buckets: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    matches: List[Dict[str, object]] = []
    seen_pairs = set()
    for index, (fingerprint, path) in enumerate(entries):
        candidates = set()
        for chunk_index in range(8):
            chunk = (fingerprint >> (chunk_index * 8)) & 0xFF
            candidates.update(buckets[(chunk_index, chunk)])
        for candidate_index in candidates:
            candidate_fingerprint, candidate_path = entries[candidate_index]
            if _split_for(path) == _split_for(candidate_path):
                continue
            pair = tuple(sorted((path, candidate_path)))
            if pair in seen_pairs:
                continue
            distance = (fingerprint ^ candidate_fingerprint).bit_count()
            if distance <= maximum_hamming_distance:
                seen_pairs.add(pair)
                matches.append(
                    {
                        "hamming_distance": distance,
                        "paths": list(pair),
                        "splits": sorted({_split_for(value) for value in pair}),
                    }
                )
                if len(matches) >= limit:
                    return sorted(matches, key=lambda item: int(item["hamming_distance"]))
        for chunk_index in range(8):
            chunk = (fingerprint >> (chunk_index * 8)) & 0xFF
            buckets[(chunk_index, chunk)].append(index)
    return sorted(matches, key=lambda item: int(item["hamming_distance"]))


if __name__ == "__main__":
    main()

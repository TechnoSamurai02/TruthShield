from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import DefaultDict, Dict, List

import cv2
import numpy as np


SPLITS = ("train", "validation", "test")
LABELS = ("ai_generated", "real_camera")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract labeled video frames while keeping whole videos isolated to one data split."
    )
    parser.add_argument("--source-dir", default="training/data/video_source")
    parser.add_argument("--output-dir", default="training/data/video_frames")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    videos = _collect_videos(source_dir)
    if not videos:
        raise SystemExit(
            f"No videos found. Put them under {source_dir}/train|validation|test/ai_generated|real_camera/."
        )
    leakage = _find_exact_video_leakage(videos)
    if leakage:
        examples = "\n".join(f"  {digest}: {paths}" for digest, paths in list(leakage.items())[:5])
        raise SystemExit(f"The same video file appears in more than one split. Fix these first:\n{examples}")
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    rows: List[Dict[str, object]] = []
    totals: DefaultDict[str, int] = defaultdict(int)
    worker = partial(
        _extract_video_job,
        source_dir=source_dir,
        output_dir=output_dir,
        frame_stride=max(1, args.frame_stride),
        max_frames=max(0, args.max_frames_per_video),
        jpeg_quality=max(40, min(100, args.jpeg_quality)),
    )
    workers = max(1, args.workers)
    if workers == 1:
        results = map(worker, videos)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="truthshield-frame")
        results = executor.map(worker, videos)
    try:
        for relative, split, label, extracted, video_rows in results:
            rows.extend(video_rows)
            totals[f"{split}/{label}"] += extracted
            print(f"{relative}: {extracted} frames", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "label",
                "source_video",
                "source_id",
                "source_frame_number",
                "timestamp_seconds",
                "output_image",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print("Frame totals:", flush=True)
    for key, value in sorted(totals.items()):
        print(f"  {key}: {value}", flush=True)
    print(f"Manifest: {manifest_path.resolve()}", flush=True)


def _collect_videos(source_dir: Path) -> List[Path]:
    videos: List[Path] = []
    for split in SPLITS:
        for label in LABELS:
            folder = source_dir / split / label
            if not folder.exists():
                continue
            videos.extend(
                path
                for path in sorted(folder.rglob("*"))
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            )
    return videos


def _find_exact_video_leakage(videos: List[Path]) -> Dict[str, List[str]]:
    hashes: DefaultDict[str, List[str]] = defaultdict(list)
    for video in videos:
        digest = hashlib.sha256()
        with video.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[digest.hexdigest()].append(str(video))
    leakage: Dict[str, List[str]] = {}
    for digest, paths in hashes.items():
        splits = {
            next((part for part in Path(path).parts if part in SPLITS), "unknown")
            for path in paths
        }
        if len(splits) > 1:
            leakage[digest] = paths
    return leakage


def _extract_video_job(
    video: Path,
    source_dir: Path,
    output_dir: Path,
    frame_stride: int,
    max_frames: int,
    jpeg_quality: int,
) -> tuple[Path, str, str, int, List[Dict[str, object]]]:
    relative = video.relative_to(source_dir)
    split, label = relative.parts[0], relative.parts[1]
    source_id = hashlib.sha256(str(relative).lower().encode("utf-8")).hexdigest()[:16]
    # Keep frames directly below the class directory. Hugging Face's
    # imagefolder builder uses directories as labels; nesting a unique
    # source_id directory here can accidentally create one class per video.
    target_dir = output_dir / split / label
    target_dir.mkdir(parents=True, exist_ok=True)
    video_rows: List[Dict[str, object]] = []
    extracted = _extract_video(
        video=video,
        target_dir=target_dir,
        source_id=source_id,
        frame_stride=frame_stride,
        max_frames=max_frames,
        jpeg_quality=jpeg_quality,
        split=split,
        label=label,
        source_relative=str(relative),
        rows=video_rows,
    )
    return relative, split, label, extracted, video_rows


def _extract_video(
    video: Path,
    target_dir: Path,
    source_id: str,
    frame_stride: int,
    max_frames: int,
    jpeg_quality: int,
    split: str,
    label: str,
    source_relative: str,
    rows: List[Dict[str, object]],
) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"Warning: could not open {video}", flush=True)
        return 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    selected_positions = _uniform_positions(reported_frame_count, frame_stride, max_frames)
    source_frame = 0
    extracted = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            should_extract = (
                source_frame in selected_positions
                if selected_positions is not None
                else source_frame % frame_stride == 0
            )
            if should_extract:
                output_name = f"{source_id}_frame_{source_frame + 1:08d}.jpg"
                output_path = target_dir / output_name
                written = cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if written:
                    extracted += 1
                    rows.append(
                        {
                            "split": split,
                            "label": label,
                            "source_video": source_relative,
                            "source_id": source_id,
                            "source_frame_number": source_frame + 1,
                            "timestamp_seconds": round(source_frame / fps, 6) if fps > 0 else "",
                            "output_image": str(output_path),
                        }
                    )
                if selected_positions is not None and extracted >= len(selected_positions):
                    break
            source_frame += 1
    finally:
        capture.release()
    return extracted


def _uniform_positions(frame_count: int, frame_stride: int, max_frames: int) -> set[int] | None:
    if max_frames <= 0 or frame_count <= 0:
        return None
    candidates = list(range(0, frame_count, max(1, frame_stride)))
    if len(candidates) <= max_frames:
        return set(candidates)
    indices = sorted({int(round(value)) for value in np.linspace(0, len(candidates) - 1, max_frames)})
    return {candidates[index] for index in indices}


if __name__ == "__main__":
    main()

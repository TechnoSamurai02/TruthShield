from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


SPLITS = ("train", "validation", "test")
LABELS = ("ai_generated", "real_camera")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze every frame and save one temporal-feature record per labeled video."
    )
    parser.add_argument("--source-dir", default="training/data/video_source")
    parser.add_argument("--output", default="training/data/video_features.jsonl")
    parser.add_argument("--frame-model", default="")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--tile-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use overlapping full-coverage model tiles. Accurate but much slower on CPU.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    os.environ["ENABLE_ENHANCED_ANALYSIS"] = "true"
    os.environ["ENABLE_LOCAL_AI_MODELS"] = "true"
    os.environ["BRAVE_SEARCH_API_KEY"] = ""
    os.environ["GOOGLE_VISION_API_KEY"] = ""
    os.environ["VIDEO_ANALYSIS_MODE"] = "exhaustive"
    os.environ["VIDEO_FRAME_STRIDE"] = str(max(1, args.frame_stride))
    os.environ["VIDEO_MAX_FRAMES"] = str(max(0, args.max_frames))
    os.environ["VIDEO_TILE_ANALYSIS"] = "true" if args.tile_analysis else "false"
    os.environ["AI_VIDEO_TEMPORAL_MODEL_PATH"] = ""
    if args.frame_model:
        os.environ["AI_VIDEO_FRAME_DETECTOR_MODELS"] = str(Path(args.frame_model).resolve())

    from analyzers.video_analyzer import analyze_video_path

    source_dir = Path(args.source_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_config = {
        "frame_model": str(Path(args.frame_model).resolve()) if args.frame_model else "configured_default",
        "frame_stride": max(1, args.frame_stride),
        "max_frames": max(0, args.max_frames),
        "tile_analysis": bool(args.tile_analysis),
    }
    completed = _completed_source_ids(output_path, analysis_config) if args.resume else set()
    videos = _collect_videos(source_dir)
    if not videos:
        raise SystemExit(
            f"No videos found below {source_dir}/train|validation|test/ai_generated|real_camera/."
        )

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for index, video in enumerate(videos, start=1):
            relative = video.relative_to(source_dir)
            source_id = str(relative).replace("\\", "/").lower()
            if source_id in completed:
                print(f"[{index}/{len(videos)}] already done: {relative}", flush=True)
                continue
            split, label = relative.parts[0], relative.parts[1]
            print(f"[{index}/{len(videos)}] deep analysis: {relative}", flush=True)
            try:
                result = analyze_video_path(str(video), filename=video.name)
                technical = result.get("technical_details") or {}
                record: Dict[str, Any] = {
                    "source_id": source_id,
                    "source_video": str(relative),
                    "split": split,
                    "label": label,
                    "features": technical.get("video_model_features") or {},
                    "frames_analyzed": result.get("frames_analyzed", 0),
                    "analysis_coverage": technical.get("analysis_coverage") or {},
                    "truth_score": result.get("truth_score"),
                    "analysis_config": analysis_config,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
            except Exception as exc:
                print(f"  ERROR: {str(exc)[:300]}", flush=True)
    print(f"Feature records: {output_path.resolve()}", flush=True)


def _collect_videos(source_dir: Path) -> List[Path]:
    videos: List[Path] = []
    for split in SPLITS:
        for label in LABELS:
            folder = source_dir / split / label
            if folder.exists():
                videos.extend(
                    path
                    for path in sorted(folder.rglob("*"))
                    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
                )
    return videos


def _completed_source_ids(output_path: Path, expected_config: Dict[str, Any]) -> set[str]:
    if not output_path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(output_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        stored_config = record.get("analysis_config")
        if stored_config != expected_config:
            raise SystemExit(
                f"Feature settings at line {line_number} do not match this run. "
                "Use a new --output path or rerun with --no-resume after removing the old feature file."
            )
        source_id = record.get("source_id")
        if source_id:
            completed.add(str(source_id))
    return completed


if __name__ == "__main__":
    main()

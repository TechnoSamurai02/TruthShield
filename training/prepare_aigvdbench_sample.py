from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import random
import re
import shutil
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable
from urllib.parse import quote

import cv2
import requests


DATASET_ID = "AIGVDBench/AIGVDBench"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}


@dataclass(frozen=True)
class ArchivePlan:
    split: str
    source: str
    path: str
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced, leakage-resistant AIGVDBench subset without downloading "
            "the benchmark's complete multi-hundred-gigabyte archives."
        )
    )
    parser.add_argument("--output-dir", default="training/data/video_source")
    parser.add_argument("--train-per-source", type=int, default=80)
    parser.add_argument("--validation-per-source", type=int, default=40)
    parser.add_argument("--test-per-source", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--chunk-size-mb", type=int, default=4)
    parser.add_argument("--max-cache-chunks", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if args.clean_output and output_dir.exists():
        _assert_safe_output(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plans = _plans(args)
    manifest_path = output_dir / "aigvdbench_manifest.csv"
    completed = _read_completed(manifest_path)
    rows = _deduplicate_content_groups(list(completed.values()))
    allowed_ai_sources = {(plan.split, plan.source) for plan in plans}
    filtered_rows = []
    for row in rows:
        if row["label"] == "ai_generated" and (row["split"], row["source"]) not in allowed_ai_sources:
            Path(row["local_path"]).unlink(missing_ok=True)
            continue
        filtered_rows.append(row)
    rows = filtered_rows
    _write_manifest(manifest_path, rows)

    session = requests.Session()
    session.headers.update({"User-Agent": "TruthShield-AIGVDBench-Sampler/1.0"})
    for plan in plans:
        existing = [
            row
            for row in rows
            if row["split"] == plan.split
            and row["label"] == "ai_generated"
            and row["source"] == plan.source
            and Path(row["local_path"]).is_file()
        ]
        needed = max(0, plan.count - len(existing))
        if needed:
            new_rows = _extract_remote_sample(
                session=session,
                plan=plan,
                label="ai_generated",
                count=needed,
                output_dir=output_dir,
                seed=args.seed,
                timeout=max(30, args.request_timeout),
                chunk_size=max(1, args.chunk_size_mb) * 1024 * 1024,
                max_cache_chunks=max(2, args.max_cache_chunks),
                excluded={row["archive_member"] for row in existing},
                excluded_groups={
                    _source_group(row["archive_member"])
                    for row in rows
                    if row["label"] == "ai_generated"
                },
            )
            rows.extend(new_rows)
            _write_manifest(manifest_path, rows)

    real_counts = {
        split: sum(plan.count for plan in plans if plan.split == split)
        for split in ("train", "validation", "test")
    }
    real_archive = "AIGVDBench/Real/Real.zip"
    for split, target_count in real_counts.items():
        existing = [
            row
            for row in rows
            if row["split"] == split
            and row["label"] == "real_camera"
            and row["source"] == "Real"
            and Path(row["local_path"]).is_file()
        ]
        needed = max(0, target_count - len(existing))
        if not needed:
            continue
        new_rows = _extract_remote_sample(
            session=session,
            plan=ArchivePlan(split, "Real", real_archive, target_count),
            label="real_camera",
            count=needed,
            output_dir=output_dir,
            seed=args.seed + {"train": 1000, "validation": 2000, "test": 3000}[split],
            timeout=max(30, args.request_timeout),
            chunk_size=max(1, args.chunk_size_mb) * 1024 * 1024,
            max_cache_chunks=max(2, args.max_cache_chunks),
            excluded={
                row["archive_member"]
                for row in rows
                if row["label"] == "real_camera"
            },
            excluded_groups={
                _source_group(row["archive_member"])
                for row in rows
                if row["label"] == "real_camera"
            },
        )
        rows.extend(new_rows)
        _write_manifest(manifest_path, rows)

    _write_manifest(manifest_path, rows)
    print("Dataset totals:", flush=True)
    for split in ("train", "validation", "test"):
        for label in ("ai_generated", "real_camera"):
            count = sum(row["split"] == split and row["label"] == label for row in rows)
            print(f"  {split}/{label}: {count}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


def _plans(args: argparse.Namespace) -> list[ArchivePlan]:
    return [
        ArchivePlan("train", "Open-Sora-T2V", "AIGVDBench/OpenSource/T2V/Open-Sora.zip", args.train_per_source),
        ArchivePlan("train", "AnimateDiff-T2V", "AIGVDBench/OpenSource/T2V/AnimateDiff.zip", args.train_per_source),
        ArchivePlan("train", "SVD-I2V", "AIGVDBench/OpenSource/I2V/SVD.zip", args.train_per_source),
        ArchivePlan(
            "train",
            "CogVideoX1.5-V2V",
            "AIGVDBench/OpenSource/V2V/Cogvideox1.5.zip",
            args.train_per_source,
        ),
        ArchivePlan("train", "LTX-V2V", "AIGVDBench/OpenSource/V2V/LTX.zip", args.train_per_source),
        ArchivePlan(
            "validation",
            "HunyuanVideo-T2V",
            "AIGVDBench/OpenSource/T2V/HunyuanVideo.zip",
            args.validation_per_source,
        ),
        ArchivePlan(
            "validation",
            "EasyAnimate-I2V",
            "AIGVDBench/OpenSource/I2V/EasyAnimate.zip",
            args.validation_per_source,
        ),
        # Pika is a closed-source family absent from training and validation.
        # Its score is therefore a more honest cross-generator test.
        ArchivePlan("test", "Pika-closed", "AIGVDBench/ClosedSource/pika.zip", args.test_per_source),
    ]


def _extract_remote_sample(
    session: requests.Session,
    plan: ArchivePlan,
    label: str,
    count: int,
    output_dir: Path,
    seed: int,
    timeout: int,
    chunk_size: int,
    max_cache_chunks: int,
    excluded: set[str],
    excluded_groups: set[str],
) -> list[dict[str, str]]:
    url = _resolve_url(plan.path)
    print(f"Opening remote archive {plan.source}...", flush=True)
    remote = HttpRangeReader(
        session=session,
        url=url,
        timeout=timeout,
        chunk_size=chunk_size,
        max_cache_chunks=max_cache_chunks,
    )
    rows: list[dict[str, str]] = []
    try:
        with zipfile.ZipFile(remote) as archive:
            candidates = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and Path(info.filename).suffix.lower() in VIDEO_EXTENSIONS
                and info.filename not in excluded
                and _source_group(info.filename) not in excluded_groups
                and info.file_size > 0
            ]
            random.Random(seed + _stable_int(plan.source)).shuffle(candidates)
            target_dir = output_dir / plan.split / label
            target_dir.mkdir(parents=True, exist_ok=True)
            selected_groups: set[str] = set()
            for info in candidates:
                if len(rows) >= count:
                    break
                source_group = _source_group(info.filename)
                if source_group in selected_groups:
                    continue
                digest = hashlib.sha256(f"{plan.source}:{info.filename}".encode("utf-8")).hexdigest()[:16]
                suffix = Path(info.filename).suffix.lower() or ".mp4"
                local_path = target_dir / f"{_slug(plan.source)}_{digest}{suffix}"
                if not local_path.exists():
                    print(
                        f"  {plan.split}/{label} {plan.source}: {len(rows) + 1}/{count} "
                        f"({info.file_size / 1024 / 1024:.1f} MB)",
                        flush=True,
                    )
                    temp_path = local_path.with_suffix(local_path.suffix + ".partial")
                    try:
                        with archive.open(info) as source, temp_path.open("wb") as target:
                            shutil.copyfileobj(source, target, length=1024 * 1024)
                        if not _readable_video(temp_path):
                            temp_path.unlink(missing_ok=True)
                            continue
                        temp_path.replace(local_path)
                    except Exception:
                        temp_path.unlink(missing_ok=True)
                        raise
                if not _readable_video(local_path):
                    local_path.unlink(missing_ok=True)
                    continue
                rows.append(
                    {
                        "split": plan.split,
                        "label": label,
                        "source": plan.source,
                        "archive_path": plan.path,
                        "archive_member": info.filename,
                        "source_group": source_group,
                        "local_path": str(local_path),
                        "size_bytes": str(local_path.stat().st_size),
                        "sha256": _sha256(local_path),
                        "license": "CC-BY-4.0",
                        "dataset_url": f"https://huggingface.co/datasets/{DATASET_ID}",
                    }
                )
                selected_groups.add(source_group)
    finally:
        remote.close()
    if len(rows) < count:
        raise RuntimeError(f"Only extracted {len(rows)} readable videos from {plan.source}; requested {count}.")
    return rows


class HttpRangeReader(io.RawIOBase):
    """Small seekable HTTP reader suitable for zipfile central-directory access."""

    def __init__(
        self,
        session: requests.Session,
        url: str,
        timeout: int,
        chunk_size: int,
        max_cache_chunks: int,
    ) -> None:
        super().__init__()
        self.session = session
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.max_cache_chunks = max_cache_chunks
        response = session.head(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
        self.url = response.url
        self.length = int(response.headers.get("content-length") or 0)
        if self.length <= 0:
            raise RuntimeError("Remote archive did not report a usable content length.")
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        self.position = max(0, min(self.length, position))
        return self.position

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        size = len(data)
        buffer[:size] = data
        return size

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
        remaining = min(size, self.length - self.position)
        parts: list[bytes] = []
        while remaining > 0:
            chunk_index = self.position // self.chunk_size
            chunk = self._chunk(chunk_index)
            offset = self.position % self.chunk_size
            take = min(remaining, len(chunk) - offset)
            if take <= 0:
                break
            parts.append(chunk[offset : offset + take])
            self.position += take
            remaining -= take
        return b"".join(parts)

    def _chunk(self, index: int) -> bytes:
        cached = self.cache.pop(index, None)
        if cached is not None:
            self.cache[index] = cached
            return cached
        start = index * self.chunk_size
        end = min(self.length - 1, start + self.chunk_size - 1)
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=self.timeout,
        )
        if response.status_code != 206:
            raise RuntimeError(f"Remote archive ignored byte range {start}-{end}: HTTP {response.status_code}")
        data = response.content
        expected = end - start + 1
        if len(data) != expected:
            raise RuntimeError(f"Short range response: expected {expected} bytes, received {len(data)}")
        self.cache[index] = data
        while len(self.cache) > self.max_cache_chunks:
            self.cache.popitem(last=False)
        return data


def _resolve_url(path: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in path.split("/"))
    return f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/{encoded}"


def _readable_video(path: Path) -> bool:
    capture = cv2.VideoCapture(str(path))
    try:
        success, frame = capture.read()
        return bool(capture.isOpened() and success and frame is not None and frame.size > 0)
    finally:
        capture.release()


def _read_completed(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["split"], row["label"], row["source"], row["archive_member"]): row
        for row in rows
    }


def _write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    fieldnames = (
        "split",
        "label",
        "source",
        "archive_path",
        "archive_member",
        "source_group",
        "local_path",
        "size_bytes",
        "sha256",
        "license",
        "dataset_url",
    )
    ordered = sorted(rows, key=lambda row: (row["split"], row["label"], row["source"], row["archive_member"]))
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
    temp_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")


def _source_group(archive_member: str) -> str:
    stem = Path(archive_member).stem.lower()
    # AIGVDBench real names commonly end in _clipIndex_starttoend. Strip
    # those clip coordinates so separate cuts of one original stay together.
    grouped = re.sub(r"_\d+_\d+to\d+$", "", stem)
    return grouped or stem


def _deduplicate_content_groups(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # Protect untouched evaluation data first. If the same underlying content
    # appears in several generator archives, remove it from training rather
    # than weakening the held-out validation or test split.
    priority = {"test": 0, "validation": 1, "train": 2}
    ordered = sorted(rows, key=lambda row: (priority.get(row.get("split", ""), 9), row.get("local_path", "")))
    kept: list[dict[str, str]] = []
    seen_groups: dict[str, set[str]] = {"ai_generated": set(), "real_camera": set()}
    for row in ordered:
        group = row.get("source_group") or _source_group(row.get("archive_member", ""))
        row["source_group"] = group
        label = row.get("label", "")
        label_groups = seen_groups.setdefault(label, set())
        if group in label_groups:
            Path(row.get("local_path", "")).unlink(missing_ok=True)
            continue
        label_groups.add(group)
        kept.append(row)
    return kept


def _assert_safe_output(path: Path) -> None:
    expected_suffix = Path("training/data/video_source")
    if len(path.parts) < len(expected_suffix.parts) or Path(*path.parts[-3:]) != expected_suffix:
        raise RuntimeError(f"Refusing to clean unexpected output directory: {path}")


if __name__ == "__main__":
    main()

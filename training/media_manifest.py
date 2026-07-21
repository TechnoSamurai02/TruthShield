from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


REQUIRED_FIELDS = (
    "path",
    "sha256",
    "media_type",
    "class_label",
    "source",
    "license",
    "generator_or_editor",
    "source_group",
    "split",
)


@dataclass(frozen=True)
class MediaRecord:
    path: str
    sha256: str
    media_type: str
    class_label: str
    source: str
    license: str
    generator_or_editor: str
    parent_media: str | None
    transformation: str
    semantic_category: str
    source_group: str
    split: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_group(*values: object) -> str:
    joined = "\u241f".join(str(value or "").strip() for value in values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def write_jsonl(path: Path, records: Iterable[MediaRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_manifest(path: Path) -> Iterator[dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def validate_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(records)
    errors: list[str] = []
    hashes: dict[str, str] = {}
    groups: dict[str, str] = {}
    generators: dict[str, set[str]] = {}
    for index, row in enumerate(rows, start=1):
        missing = [field for field in REQUIRED_FIELDS if not str(row.get(field) or "").strip()]
        if missing:
            errors.append(f"row {index}: missing {', '.join(missing)}")
        split = str(row.get("split") or "")
        sha256 = str(row.get("sha256") or "")
        group = str(row.get("source_group") or "")
        generator = str(row.get("generator_or_editor") or "")
        if sha256 in hashes and hashes[sha256] != split:
            errors.append(f"hash leakage: {sha256[:12]} appears in {hashes[sha256]} and {split}")
        hashes[sha256] = split
        if group in groups and groups[group] != split:
            errors.append(f"source-group leakage: {group} appears in {groups[group]} and {split}")
        groups[group] = split
        if generator and generator.lower() not in {"authentic", "real", "camera", "unknown"}:
            generators.setdefault(generator, set()).add(split)
    generator_leakage = {
        generator: sorted(splits)
        for generator, splits in generators.items()
        if len(splits) > 1
    }
    for generator, splits in generator_leakage.items():
        errors.append(f"generator-family leakage: {generator} appears in {', '.join(splits)}")
    return {
        "valid": not errors,
        "record_count": len(rows),
        "errors": errors,
        "generator_splits": {key: sorted(value) for key, value in sorted(generators.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a TruthShield v4 media manifest for leakage.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_records(read_manifest(args.manifest))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

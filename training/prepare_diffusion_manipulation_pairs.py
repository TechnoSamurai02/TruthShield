from __future__ import annotations

import argparse
import gc
import inspect
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.media_manifest import MediaRecord, read_manifest, sha256_file


AUTHENTIC_LABELS = {"authentic", "real", "real_camera"}
GENERATED_LABELS = {"generated", "ai_generated"}
SUPPORTED_SPLITS = ("train", "tuning", "calibration", "locked_test")
DATASET_LICENSE = "source-license-retained; local derivative for detector research"

# An editor family never crosses a split. Do not casually change this mapping:
# generator separation is the point of this dataset.
SPLIT_MODEL_SPECS: dict[str, dict[str, str]] = {
    "train": {
        "model_id": "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "family": "stable-diffusion-v1.5-inpainting",
        "variant": "fp16",
        "license": "CreativeML OpenRAIL-M",
        "license_url": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting",
    },
    "tuning": {
        "model_id": "sd2-community/stable-diffusion-2-inpainting",
        "family": "stable-diffusion-v2-inpainting",
        "license": "CreativeML Open RAIL++-M",
        "license_url": "https://huggingface.co/sd2-community/stable-diffusion-2-inpainting",
    },
    "calibration": {
        "model_id": "kandinsky-community/kandinsky-2-2-decoder-inpaint",
        "family": "kandinsky-v2.2-inpainting",
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint",
    },
    "locked_test": {
        "model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "family": "stable-diffusion-xl-inpainting",
        "license": "CreativeML Open RAIL++-M",
        "license_url": "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    },
}

PROMPTS = (
    "a realistic bicycle naturally integrated into the photograph, natural lighting, detailed",
    "a realistic potted plant naturally integrated into the photograph, natural lighting, detailed",
    "a realistic wooden chair naturally integrated into the photograph, natural lighting, detailed",
    "a realistic backpack naturally integrated into the photograph, natural lighting, detailed",
    "a realistic street sign naturally integrated into the photograph, natural lighting, detailed",
    "a realistic table lamp naturally integrated into the photograph, natural lighting, detailed",
    "a realistic bowl of fruit naturally integrated into the photograph, natural lighting, detailed",
    "a realistic small tree naturally integrated into the photograph, natural lighting, detailed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one resumable split of mask-supervised diffusion inpainting edits. "
            "Run train, tuning, calibration, and locked_test as separate processes."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--variants",
        type=int,
        default=0,
        help="Manipulated variants per authentic parent. Zero uses 2 for train and 1 elsewhere.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2404)
    parser.add_argument(
        "--clean-split",
        action="store_true",
        help="Delete only this generated split below output-dir before rebuilding it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_manifest = source_dir / "manifest.v4.jsonl"
    if not source_manifest.is_file():
        raise SystemExit(f"Source manifest not found: {source_manifest}")
    if args.resolution < 256 or args.resolution % 64:
        raise SystemExit("--resolution must be at least 256 and divisible by 64.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_split:
        _clean_generated_split(output_dir, args.split)

    source_rows = [dict(row) for row in read_manifest(source_manifest)]
    authentic_rows = _eligible_rows(source_rows, source_dir, args.split, AUTHENTIC_LABELS)
    generated_rows = _eligible_rows(source_rows, source_dir, args.split, GENERATED_LABELS)
    if args.max_images > 0:
        authentic_rows = authentic_rows[: args.max_images]
        generated_rows = generated_rows[: args.max_images]
    if not authentic_rows:
        raise SystemExit(f"No authentic parents were found for split '{args.split}'.")
    if not generated_rows:
        raise SystemExit(f"No fully generated negative references were found for split '{args.split}'.")

    variants = args.variants or (2 if args.split == "train" else 1)
    spec = SPLIT_MODEL_SPECS[args.split]
    record_dir = output_dir / "records" / args.split
    record_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_parent_indices(
        record_dir,
        output_dir,
        required_variants=variants,
    )
    pending = [index for index in range(len(authentic_rows)) if index not in completed]

    print(f"Split: {args.split}", flush=True)
    print(f"Editor family: {spec['family']}", flush=True)
    print(f"Model: {spec['model_id']}", flush=True)
    print(f"Parents: {len(authentic_rows)} ({len(pending)} pending)", flush=True)
    print(f"Variants per parent: {variants}", flush=True)
    if pending:
        pipeline, torch = _load_pipeline(spec, args.cache_dir)
        try:
            for ordinal, index in enumerate(pending, start=1):
                row = authentic_rows[index]
                generated_row = generated_rows[index % len(generated_rows)]
                try:
                    bundle = _generate_parent_bundle(
                        pipeline=pipeline,
                        torch=torch,
                        source_dir=source_dir,
                        output_dir=output_dir,
                        split=args.split,
                        index=index,
                        authentic_row=row,
                        generated_row=generated_row,
                        variants=variants,
                        resolution=args.resolution,
                        steps=max(4, args.steps),
                        seed=args.seed,
                        spec=spec,
                    )
                except Exception as exc:
                    print(f"Skipped parent {index}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                record_path = record_dir / f"{index:06d}.json"
                record_path.write_text(
                    json.dumps(bundle, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if ordinal % 10 == 0 or ordinal == len(pending):
                    print(
                        f"Generated {ordinal}/{len(pending)} pending parents "
                        f"({len(completed) + ordinal}/{len(authentic_rows)} total)",
                        flush=True,
                    )
        finally:
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = _finalize_dataset(output_dir, source_manifest=source_manifest)
    split_records = [row for row in report["records"] if row["split"] == args.split]
    rendered_report = {
        "source_manifest": str(source_manifest),
        "split": args.split,
        "model": spec,
        "requested_parents": len(authentic_rows),
        "completed_parents": len(
            _completed_parent_indices(
                record_dir,
                output_dir,
                required_variants=variants,
            )
        ),
        "manifest_records_for_split": len(split_records),
        "localized_records_for_split": sum(
            1 for row in report["localization"] if row["split"] == args.split
        ),
        "resolution": args.resolution,
        "steps": max(4, args.steps),
        "variants": variants,
        "resume_supported": True,
    }
    (output_dir / f"preparation-report.{args.split}.json").write_text(
        json.dumps(rendered_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rendered_report, indent=2, sort_keys=True), flush=True)
    print(f"Combined manifest: {output_dir / 'manifest.v4.jsonl'}", flush=True)


def _load_pipeline(spec: dict[str, str], cache_dir: Path | None) -> tuple[Any, Any]:
    try:
        import torch
        from diffusers import AutoPipelineForInpainting
    except ImportError as exc:
        raise SystemExit(
            "Install the training requirements, including diffusers, before generating edits."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for diffusion edit generation.")
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
    }
    if spec.get("variant"):
        kwargs["variant"] = spec["variant"]
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir.resolve())
    print(f"Loading {spec['model_id']} on {torch.cuda.get_device_name(0)}", flush=True)
    pipeline = AutoPipelineForInpainting.from_pretrained(spec["model_id"], **kwargs)
    if hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    return pipeline, torch


def _generate_parent_bundle(
    *,
    pipeline: Any,
    torch: Any,
    source_dir: Path,
    output_dir: Path,
    split: str,
    index: int,
    authentic_row: dict[str, Any],
    generated_row: dict[str, Any],
    variants: int,
    resolution: int,
    steps: int,
    seed: int,
    spec: dict[str, str],
) -> dict[str, Any]:
    with Image.open(source_dir / str(authentic_row["path"])) as handle:
        original = _square_image(handle.convert("RGB"), resolution)
    with Image.open(source_dir / str(generated_row["path"])) as handle:
        generated_negative = _square_image(handle.convert("RGB"), resolution)

    base_name = f"{split}_{index:06d}"
    authentic_path = output_dir / split / "real_camera" / f"{base_name}_real.png"
    generated_path = output_dir / split / "ai_generated" / f"{base_name}_generated.png"
    authentic_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    original.save(authentic_path, format="PNG", optimize=True)
    generated_negative.save(generated_path, format="PNG", optimize=True)

    source_group = str(authentic_row.get("source_group") or f"{split}-{index:06d}")
    records = [
        _record_for_output(
            output_dir,
            authentic_path,
            class_label="real_camera",
            source_row=authentic_row,
            generator_or_editor="authentic",
            parent_media=None,
            transformation="square-resize-for-diffusion-paired-original",
            source_group=source_group,
        ),
        _record_for_output(
            output_dir,
            generated_path,
            class_label="ai_generated",
            source_row=generated_row,
            generator_or_editor=str(generated_row.get("generator_or_editor") or "unknown-generator"),
            parent_media=str(generated_row.get("path") or ""),
            transformation="square-resize-generated-negative-reference",
            source_group=str(generated_row.get("source_group") or f"{split}-generated-{index:06d}"),
        ),
    ]
    localization = []
    for variant in range(variants):
        item_seed = _stable_seed(seed, split, index, variant)
        randomizer = random.Random(item_seed)
        mask = _random_inpainting_mask((resolution, resolution), randomizer)
        prompt = PROMPTS[randomizer.randrange(len(PROMPTS))]
        torch_generator = torch.Generator(device="cuda").manual_seed(item_seed)
        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": original,
            "mask_image": mask,
            "generator": torch_generator,
            "num_inference_steps": steps,
        }
        parameters = inspect.signature(pipeline.__call__).parameters
        if "height" in parameters:
            call_kwargs["height"] = resolution
        if "width" in parameters:
            call_kwargs["width"] = resolution
        if "strength" in parameters:
            call_kwargs["strength"] = 0.98
        if "guidance_scale" in parameters:
            call_kwargs["guidance_scale"] = 7.5
        result = pipeline(**call_kwargs)
        if not result.images or result.images[0] is None:
            raise RuntimeError("The inpainting pipeline returned no image (possibly safety-filtered).")
        inpainted = result.images[0].convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
        alpha = mask.filter(ImageFilter.GaussianBlur(radius=max(2, resolution // 128)))
        manipulated = Image.composite(inpainted, original, alpha)
        ground_truth = alpha.point(lambda value: 255 if value >= 8 else 0)

        manipulated_path = output_dir / split / "ai_manipulated" / f"{base_name}_v{variant}.png"
        mask_path = output_dir / "localization_masks" / split / f"{base_name}_v{variant}.png"
        manipulated_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        manipulated.save(manipulated_path, format="PNG", optimize=True)
        ground_truth.save(mask_path, format="PNG", optimize=True)
        rendered = _record_for_output(
            output_dir,
            manipulated_path,
            class_label="ai_manipulated",
            source_row=authentic_row,
            generator_or_editor=spec["family"],
            parent_media=str(authentic_row.get("path") or ""),
            transformation=f"diffusion-inpaint;variant={variant};prompt={prompt}",
            source_group=source_group,
            derivative_license=f"{spec['license']} model output; source license retained",
        )
        records.append(rendered)
        bounds = ground_truth.getbbox() or (0, 0, resolution, resolution)
        localization.append(
            {
                **rendered,
                "mask_path": mask_path.relative_to(output_dir).as_posix(),
                "suspicious_region": list(bounds),
                "mask_coverage": round(_mask_coverage(ground_truth), 6),
                "prompt": prompt,
                "editor_model_id": spec["model_id"],
                "editor_model_license": spec["license"],
                "editor_model_license_url": spec["license_url"],
            }
        )
    return {
        "parent_index": index,
        "split": split,
        "records": records,
        "localization": localization,
    }


def _record_for_output(
    output_dir: Path,
    path: Path,
    *,
    class_label: str,
    source_row: dict[str, Any],
    generator_or_editor: str,
    parent_media: str | None,
    transformation: str,
    source_group: str,
    derivative_license: str | None = None,
) -> dict[str, Any]:
    source_license = str(source_row.get("license") or DATASET_LICENSE)
    return asdict(
        MediaRecord(
            path=path.relative_to(output_dir).as_posix(),
            sha256=sha256_file(path),
            media_type="image",
            class_label=class_label,
            source=f"{source_row.get('source')}; TruthShield paired diffusion-localization corpus",
            license=(
                f"source={source_license}; derivative={derivative_license}"
                if derivative_license
                else source_license
            ),
            generator_or_editor=generator_or_editor,
            parent_media=parent_media,
            transformation=transformation,
            semantic_category=str(source_row.get("semantic_category") or "unspecified"),
            source_group=source_group,
            split=str(source_row.get("split") or ""),
        )
    )


def _eligible_rows(
    rows: list[dict[str, Any]],
    source_dir: Path,
    split: str,
    labels: set[str],
) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if str(row.get("split") or "") == split
            and str(row.get("class_label") or "").lower() in labels
            and (source_dir / str(row.get("path") or "")).is_file()
        ),
        key=lambda row: str(row.get("path") or ""),
    )


def _square_image(image: Image.Image, resolution: int) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    return image.crop((left, top, left + side, top + side)).resize(
        (resolution, resolution), Image.Resampling.LANCZOS
    )


def _random_inpainting_mask(size: tuple[int, int], randomizer: random.Random) -> Image.Image:
    width, height = size
    for _ in range(32):
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        target_fraction = randomizer.uniform(0.06, 0.28)
        target_area = width * height * target_fraction
        aspect = randomizer.uniform(0.55, 1.8)
        box_width = int(max(width * 0.14, min(width * 0.62, (target_area * aspect) ** 0.5)))
        box_height = int(max(height * 0.14, min(height * 0.62, target_area / max(1, box_width))))
        left = randomizer.randint(0, max(0, width - box_width))
        top = randomizer.randint(0, max(0, height - box_height))
        box = (left, top, left + box_width, top + box_height)
        if randomizer.random() < 0.55:
            draw.ellipse(box, fill=255)
        else:
            radius = max(4, min(box_width, box_height) // randomizer.randint(5, 10))
            draw.rounded_rectangle(box, radius=radius, fill=255)
        if randomizer.random() < 0.45:
            points = []
            for _ in range(randomizer.randint(3, 6)):
                points.append(
                    (
                        randomizer.randint(left, left + box_width),
                        randomizer.randint(top, top + box_height),
                    )
                )
            draw.line(
                points,
                fill=255,
                width=max(8, min(width, height) // randomizer.randint(18, 35)),
                joint="curve",
            )
        coverage = _mask_coverage(mask)
        if 0.045 <= coverage <= 0.36:
            return mask
    raise RuntimeError("Could not create a mask inside the configured coverage range.")


def _mask_coverage(mask: Image.Image) -> float:
    histogram = mask.convert("L").histogram()
    foreground = sum(histogram[1:])
    return foreground / max(1, mask.width * mask.height)


def _stable_seed(seed: int, split: str, index: int, variant: int) -> int:
    split_offset = SUPPORTED_SPLITS.index(split) * 1_000_003
    return int(seed + split_offset + index * 1009 + variant * 97)


def _completed_parent_indices(
    record_dir: Path,
    output_dir: Path,
    *,
    required_variants: int,
) -> set[int]:
    completed = set()
    for path in sorted(record_dir.glob("*.json")) if record_dir.is_dir() else []:
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
            referenced = [
                output_dir / str(row["path"])
                for row in bundle.get("records", [])
            ] + [
                output_dir / str(row["mask_path"])
                for row in bundle.get("localization", [])
            ]
            localized = list(bundle.get("localization", []))
            if (
                len(localized) >= required_variants
                and referenced
                and all(item.is_file() for item in referenced)
            ):
                completed.add(int(bundle["parent_index"]))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    return completed


def _finalize_dataset(output_dir: Path, *, source_manifest: Path) -> dict[str, Any]:
    records = []
    localization = []
    for split in SUPPORTED_SPLITS:
        record_dir = output_dir / "records" / split
        for path in sorted(record_dir.glob("*.json")) if record_dir.is_dir() else []:
            try:
                bundle = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            records.extend(bundle.get("records", []))
            localization.extend(bundle.get("localization", []))
    records.sort(key=lambda row: (str(row.get("split")), str(row.get("path"))))
    localization.sort(key=lambda row: (str(row.get("split")), str(row.get("path"))))
    _write_jsonl(output_dir / "manifest.v4.jsonl", records)
    _write_jsonl(output_dir / "localization.v4.jsonl", localization)
    license_artifact = {
        "source_manifest": str(source_manifest),
        "dataset_policy": (
            "Source licenses are retained per record. Diffusion editor families are isolated "
            "by split. Model weights are downloaded for generation only and are not redistributed."
        ),
        "editor_models": SPLIT_MODEL_SPECS,
    }
    (output_dir / "licenses.v4.json").write_text(
        json.dumps(license_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"records": records, "localization": localization}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _clean_generated_split(output_dir: Path, split: str) -> None:
    resolved_output = output_dir.resolve()
    targets = [
        output_dir / split,
        output_dir / "localization_masks" / split,
        output_dir / "records" / split,
    ]
    for target in targets:
        resolved_target = target.resolve()
        if resolved_output not in resolved_target.parents:
            raise SystemExit(f"Refusing to clean a path outside output-dir: {resolved_target}")
        if resolved_target.is_dir():
            shutil.rmtree(resolved_target)


if __name__ == "__main__":
    main()

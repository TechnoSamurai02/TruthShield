# TruthShield v4 accuracy workflow

V4 separates model training from product claims. A model may be added to the registry, but it cannot issue decisive verdicts until it has a license record, generator-isolated calibration, checksums, and a passing locked evaluation report.

## 1. Build manifests, not anonymous folders

Every record must include a content hash, source, license, generator/editor, parent media, transformation, semantic category, source group, and split. Export Defactify while retaining `Label_B`:

```bash
python training/prepare_defactify_sample.py \
  --output-dir training/data/defactify_v4 \
  --split-policy generator-heldout-v4 \
  --dataset-license "REVIEW_AND_REPLACE"
python training/media_manifest.py training/data/defactify_v4/manifest.v4.jsonl \
  --report training/evaluation/defactify-v4-leakage.json
```

The validator fails if a hash, source group, or generated family crosses splits. Do not waive the failure. Rebuild the split. The locked test split must never be used for training, checkpoint choice, feature selection, threshold selection, or manual failure review.

Hard-real categories must be represented separately: phone photography, CGI, digital art, animation, game footage, screenshots, memes, low light, sports, reflections, and social-media recompression. Apply upload transformations symmetrically to authentic, generated, and manipulated classes.

## 2. Specialist model integration

The registry is `training/models/model-registry.v4.json`.

- Community Forensics is supported through the official-repository adapter. Review and clone its MIT repository, install its pinned dependencies, then set `COMMUNITY_FORENSICS_REPO_PATH` and `COMMUNITY_FORENSICS_MODEL_ID=OwensLab/commfor-model-224`. The adapter uses official preprocessing and weights; it does not execute arbitrary Hub code.
- SPAI is intentionally optional until its pinned code/weights are packaged and CPU behavior is benchmarked. Do not silently replace it with a handcrafted FFT score.
- Manipulation weights are disabled until redistribution terms are verified. A legacy edited/captioned class can screen for a low editing score, but cannot issue a positive AI-manipulation verdict.
- The first locally derived whole-image/tile fallback is retained only as a rejected comparison experiment. Its full-image and tiled tuning results did not meet the low-false-warning gate.
- The current fallback uses `prepare_diffusion_manipulation_pairs.py` plus `train_manipulation_localizer.py`. It generates real diffusion-inpainted edits with exact masks, keeps fully generated images as manipulation-negative examples, and isolates editor families across train, tuning, calibration, and locked test. It remains disabled for decisive outcomes until calibration and the locked gate pass.
- The complete-video scan includes a non-decisive second-order motion screen. The product does not label it “D3” unless the official MIT implementation and an appropriately licensed checkpoint are actually installed and validated.

These restrictions are accuracy features: a missing specialist produces `inconclusive`, not a fabricated substitute score.

Build and train the mask-supervised fallback without exposing calibration or locked data to the trainer. Run the two generation commands separately because each downloads and uses one GPU inpainting family. Both commands resume from completed parent records:

```bash
python training/prepare_diffusion_manipulation_pairs.py \
  --source-dir training/data/defactify_v4 \
  --output-dir training/data/manipulation_diffusion_v4 \
  --split train
python training/prepare_diffusion_manipulation_pairs.py \
  --source-dir training/data/defactify_v4 \
  --output-dir training/data/manipulation_diffusion_v4 \
  --split tuning
python training/media_manifest.py training/data/manipulation_diffusion_v4/manifest.v4.jsonl \
  --report training/evaluation/manipulation-diffusion-v4-leakage.json
python training/train_manipulation_localizer.py \
  --data-dir training/data/manipulation_diffusion_v4 \
  --output-dir training/models/truthshield-manipulation-localizer-v4-candidate \
  --epochs 12 \
  --batch-size 8 \
  --resolution 384 \
  --resume
python training/evaluate_manipulation_localizer.py \
  --data-dir training/data/manipulation_diffusion_v4 \
  --model-dir training/models/truthshield-manipulation-localizer-v4-candidate \
  --split tuning \
  --output training/evaluation/manipulation-localizer-tuning-v4.csv
```

The generator uses Stable Diffusion 1.5 only for training, Stable Diffusion 2 only for tuning, Kandinsky 2.2 only for calibration, and SDXL only for the locked test. Model license URLs are written to `licenses.v4.json`; downloaded editor weights are not redistributed. The trainer exports a lightweight LR-ASPP MobileNetV3 TorchScript candidate. Do not configure it in a public deployment until its calibration artifact and locked report are packaged.

## 3. Calibrate only on the calibration split

For a new video fusion model, first regenerate features with the exact adaptive production policy. Do not train a v4 model from the old 16-uniform-frame cache:

```bash
python training/extract_video_features.py \
  --source-dir training/data/video_source \
  --frame-model training/models/truthshield-video-frame-detector-v4 \
  --output training/data/video_features_adaptive_v4.jsonl \
  --keyframe-max 64 \
  --window-max 8 \
  --resume
python training/train_video_detector.py \
  --features training/data/video_features_adaptive_v4.jsonl \
  --output training/models/truthshield-video-temporal-v4-candidate.joblib
```

The model bundle records `sampling_policy=adaptive_v4`. Runtime selects adaptive features only for such bundles and keeps the old 16-frame feature distribution for rollback models.

Export one CSV row per calibration item with at least:

```text
path,label,generator_or_editor,generation_score,manipulation_score,localized_or_persistent_support
```

Then fit constrained thresholds:

```bash
python training/calibrate_media_policy.py training/evaluation/image-calibration.csv \
  --media-type image \
  --output training/models/media-policy-v4.image.calibrated.json
```

The calibrator disables a decisive outcome when no threshold satisfies the false-warning and precision limits. It never relaxes those limits to improve coverage.

## 4. Run the locked promotion gate

```bash
python training/evaluate_media_policy.py training/evaluation/image-locked.csv \
  --media-type image \
  --policy training/models/media-policy-v4.image.calibrated.json \
  --output training/evaluation/image-v4-locked-report.json
```

The report includes four-way confusion counts, per-class and per-generator recall/coverage, decisive AI precision, authentic false-warning rate, synthetic false-authentic rate, ROC-AUC, calibration error, bootstrap intervals, and the promotion decision.

Do not promote unless every gate passes and there is no hard-real or supported-transformation regression. Once a locked failure is inspected, retire that suite into development data and create a new locked suite.

## 5. Production variables

```text
MEDIA_DECISION_POLICY_PATH=/app/training/models/media-policy-v4.image.calibrated.json
VIDEO_ANALYSIS_MODE=adaptive
VIDEO_KEYFRAME_MAX=64
VIDEO_WINDOW_MAX=8
IMAGE_TRANSFORMATION_CHECKS=true
COMMUNITY_FORENSICS_REPO_PATH=/app/vendor/Community-Forensics
COMMUNITY_FORENSICS_MODEL_ID=OwensLab/commfor-model-224
AI_MANIPULATION_DETECTOR_MODELS=/app/training/models/licensed-manipulation-specialist
```

`/api/health` reports artifact checksums, calibration status, and which decisive outcomes the configured service can support. Retain legacy model directories as an explicit rollback; do not mix their scores with web matches, missing metadata, or a generic truth score.

## Cloud training

Use `training/cloud/truthshield_v4_lightning_colab.ipynb`. Put caches and checkpoints on persistent storage, keep `--resume` enabled, and export only final model artifacts, registry metadata, calibration policy, and evaluation reports. Free GPU availability is not guaranteed, so no stage assumes one uninterrupted session.

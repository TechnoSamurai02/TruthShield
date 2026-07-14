# Image detector evaluation

This folder holds reproducible decision-level reports for the image detector. The source images are intentionally not committed: `training/data/` is local and ignored because the Defactify images are redistributed under their own dataset terms.

The measured results and interpretation are summarized in [SUMMARY.md](./SUMMARY.md). Keep that summary tied to the checked-in JSON reports; do not replace abstentions with forced binary labels when comparing versions.

## Evaluation populations

The primary untouched test population is built by `prepare_defactify_sample.py` from the official Defactify test split:

- `real_camera`: real MS COCO photographs, including people, animals, food, indoor scenes, outdoor scenes, streets, buildings, and landscapes.
- `real_edited_or_captioned`: real photographs with deterministic crops, captions, meme-like overlays, and recompression.
- `ai_generated`: images from the generators represented by Defactify.

The report generator samples each class independently, excludes cross-split leakage identified by `audit_image_dataset.py`, and reports both the former frontend rule (`AI-class score >= 0.70`) and the conservative three-way decision (`<= 0.15` authentic, `>= 0.95` AI, otherwise inconclusive).

Run a balanced test report:

```powershell
.\backend\venv\Scripts\python.exe .\training\evaluate_image_detector.py `
  --model-dir .\training\models\truthshield-image-detector-v2 `
  --data-dir .\training\data\defactify_sample `
  --split test `
  --batch-size 32 `
  --max-per-class 600 `
  --seed 20260714 `
  --output .\training\evaluation\decision_report_test_1800.json
```

Run transformation robustness checks against known real and known AI sources separately:

```powershell
.\backend\venv\Scripts\python.exe .\training\check_image_robustness.py <real image paths> `
  --expected authentic `
  --output .\training\evaluation\real_transform_robustness.json

.\backend\venv\Scripts\python.exe .\training\check_image_robustness.py <AI image paths> `
  --expected ai `
  --output .\training\evaluation\ai_transform_robustness.json
```

Every report records actual model outputs. An inconclusive decision is counted explicitly; it is never silently converted into either class.

For a slower end-to-end sample that also executes decode, metadata, pixel forensics, provenance/web fallback, aggregation, and response-schema validation:

```powershell
.\backend\venv\Scripts\python.exe .\training\evaluate_image_pipeline.py `
  --max-per-class 100 `
  --seed 20260714 `
  --output .\training\evaluation\full_pipeline_report_test_300.json
```

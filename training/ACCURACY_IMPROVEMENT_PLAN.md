# TruthShield accuracy improvement plan

This is the recommended next training cycle for the image and video models. It is written in the order you should do the work.

No AI detector can be perfectly accurate on every new generator, edit, screenshot, or compressed upload. The safest goal is:

1. Reduce false AI warnings.
2. Catch more AI content without undoing step 1.
3. Return **inconclusive** when the evidence is weak.
4. Measure every change on files that were never used for training or model selection.

## Where the models are now

### Image model

The packaged `truthshield-image-detector-v2` report contains 8,396 leakage-filtered test images.

- Binary AI detection accuracy: **90.14%**.
- ROC-AUC: **0.9612**.
- False-positive rate at the ordinary 0.50 threshold: **9.87%**.
- Conservative app policy: **98.15% accuracy among decisive results**, with **63.11% decisive coverage** on the balanced 1,800-image score evaluation.
- Full app pipeline: **97.10% accuracy among decisive results**, with **69.00% decisive coverage** on the 300-image evaluation.

The conservative app result is safer than forcing every image into real or AI, but it leaves many images inconclusive. The next image model should improve coverage without increasing false accusations.

### Video models

The packaged temporal model was tested on 80 AI and 80 real held-out videos.

- Balanced accuracy: **83.13%**.
- ROC-AUC: **0.9069**.
- False AI warnings: **17 of 80 real videos** at the selected 0.25 threshold.
- Missed AI videos: **10 of 80 AI videos**.

The packaged video-frame model's held-out video-level balanced accuracy is **79.38%**, with 16 false AI warnings and 17 missed AI videos. The temporal combination restores AI recall while preserving most of the frame model's improvement on real videos. Video still needs a broader locked challenge set and more hard-real data before its score should be treated as representative of the open world.

## Before training anything

1. Keep the current test folders untouched.
2. Create a new, locked challenge set for the next version. Do not train on it.
3. Keep related files together. Frames from one video, crops of one image, or clips from one source must all stay in the same split.
4. Keep at least one entire AI generator family out of training and validation.
5. Record the source, license, generator or camera source, and split in a manifest.
6. Apply compression, resizing, captions, and screenshots to both real and AI classes. Otherwise the model may learn file quality instead of AI generation.

## Train the next image model

### 1. Build a broader `v3` dataset

Start with a larger Defactify export in a new folder so the current dataset remains reproducible:

```powershell
cd "C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai"
.\backend\venv\Scripts\python.exe .\training\prepare_defactify_sample.py `
  --output-dir .\training\data\image_v3 `
  --max-per-label 8000 `
  --clean-output
```

Add real-but-edited examples:

```powershell
.\backend\venv\Scripts\python.exe .\training\make_captioned_real_variants.py `
  --dataset-dir .\training\data\image_v3 `
  --max-per-split 3000
```

Add new licensed data that is not just more of the same source:

- Real phone photos from several devices, lighting conditions, and subjects.
- Screenshots, memes, social-media downloads, and repeatedly recompressed real photos.
- Real CGI, digital art, game screenshots, charts, and illustrations as hard negatives.
- AI images from generator families absent from the current training data.
- Cropped, captioned, resized, and recompressed AI images.
- Local edits and face swaps as a separate benchmark. Do not silently label every edited real photo as fully AI-generated.

Balance scene types in every split: people, hands, text, animals, food, buildings, landscapes, low light, reflections, and fine textures.

### 2. Audit before training

```powershell
.\backend\venv\Scripts\python.exe .\training\audit_image_dataset.py `
  --data-dir .\training\data\image_v3 `
  --output .\training\data\image_v3_audit.json `
  --workers 1
```

Resolve exact duplicates and visually inspect near-duplicate warnings across splits. A high score caused by leakage is not a real improvement.

### 3. Fine-tune a new version

Use a CUDA GPU if possible. A CPU run is valid but will be much slower.

Smoke test:

```powershell
.\backend\venv\Scripts\python.exe .\training\train_image_detector.py `
  --data-dir .\training\data\image_v3 `
  --base-model .\training\models\truthshield-image-detector-v2 `
  --output-dir .\training\models\truthshield-image-detector-v3-smoke `
  --epochs 1 --batch-size 4 `
  --max-train-samples 600 --max-eval-samples 200
```

Full run:

```powershell
.\backend\venv\Scripts\python.exe .\training\train_image_detector.py `
  --data-dir .\training\data\image_v3 `
  --base-model .\training\models\truthshield-image-detector-v2 `
  --output-dir .\training\models\truthshield-image-detector-v3 `
  --epochs 3 --batch-size 8 `
  --early-stopping-patience 2
```

### 4. Evaluate honestly

```powershell
.\backend\venv\Scripts\python.exe .\training\evaluate_image_detector.py `
  --model-dir .\training\models\truthshield-image-detector-v3 `
  --data-dir .\training\data\image_v3 `
  --split test `
  --batch-size 16 `
  --audit-report .\training\data\image_v3_audit.json `
  --output .\training\evaluation\image_v3_test.json
```

Do not promote `v3` only because its average accuracy is higher. Require all of these:

- Lower false AI warning rate on real-camera and real-edited images.
- No major drop in AI recall.
- Better macro F1 and balanced accuracy.
- Similar or better calibration error.
- Better conservative three-way coverage at the same false-warning target.
- No failures on the locked transformation and out-of-domain challenge sets.

## Train the next video models

### 1. Improve the video data first

The existing AIGVDBench sample already separates generator families. Keep that structure, then add more independent sources.

Add hard real examples that currently resemble AI artifacts:

- Animation, CGI, game footage, and screen recordings.
- Fast sports, water, smoke, crowds, reflections, low light, and camera shake.
- Slow motion, stabilization, filters, transitions, and heavy social-media compression.
- Different phones, cameras, resolutions, aspect ratios, and frame rates.

Add hard AI examples:

- New text-to-video, image-to-video, and video-to-video generator families.
- Videos with logos, subtitles, crops, cuts, speed changes, and recompression.
- Longer clips where only part of the timeline contains generation or manipulation.

Match duration, frame rate, resolution, codec, and compression distributions across real and AI classes. Otherwise the model will learn shortcuts.

### 2. Rebuild frames

```powershell
.\backend\venv\Scripts\python.exe .\training\prepare_video_frames.py `
  --source-dir .\training\data\video_source `
  --output-dir .\training\data\video_frames_v4 `
  --frame-stride 1 `
  --max-frames-per-video 16 `
  --clean-output
```

Sixteen uniformly spaced frames keep each source video equally weighted. Do not train on every nearly identical adjacent frame.

### 3. Train the frame detector

CPU-friendly run:

```powershell
.\backend\venv\Scripts\python.exe .\training\train_video_frame_detector.py `
  --data-dir .\training\data\video_frames_v4 `
  --base-model .\training\models\truthshield-image-detector-v3 `
  --output-dir .\training\models\truthshield-video-frame-detector-v4 `
  --embedding-dir .\training\data\video_frame_embeddings_v4 `
  --batch-size 32
```

For the highest practical accuracy, compare that frozen-backbone result with a short full-backbone fine-tune on a GPU. Keep whichever wins on generator-separated validation and the locked test—not whichever has the best training score.

### 4. Rebuild and train the temporal model

```powershell
.\backend\venv\Scripts\python.exe .\training\build_video_features_from_frame_cache.py `
  --frame-dir .\training\data\video_frames_v4 `
  --embedding-dir .\training\data\video_frame_embeddings_v4 `
  --frame-model .\training\models\truthshield-video-frame-detector-v4 `
  --output .\training\data\video_features_v4.jsonl

.\backend\venv\Scripts\python.exe .\training\train_video_detector.py `
  --features .\training\data\video_features_v4.jsonl `
  --output .\training\models\truthshield-video-temporal-v4.joblib `
  --exclude-features pixel_forensic_probability_mean,pixel_forensic_probability_p95,frame_truth_score_mean,frame_truth_score_std,frame_truth_score_p10
```

The temporal trainer writes `truthshield-video-temporal-v4.metrics.json` next to the model automatically.

### 5. Set a safer video decision policy

The current temporal model's 0.25 threshold catches 70 of 80 held-out AI videos but still falsely warns on 17 of 80 held-out real videos. For the next model:

1. Calibrate probabilities on validation only.
2. Choose two thresholds: one for a strong AI warning and one for a strong lower-AI result.
3. Return **inconclusive** between those thresholds.
4. Select thresholds for a target false-warning rate, not maximum raw accuracy.
5. Report performance per generator, per real-video category, and after common upload transformations.

Do not tune thresholds on the final test set.

## Promotion checklist

Deploy a new model only when all boxes are checked:

- Dataset license and source manifest saved.
- Exact and near-duplicate split audit reviewed.
- Generator or source groups do not cross splits.
- Locked test and challenge sets were not used for training or threshold selection.
- False AI warnings, missed AI, balanced accuracy, macro F1, ROC-AUC, calibration error, and inconclusive coverage reported.
- Image results reported separately for real camera, real edited, and AI.
- Video results reported separately by generator family and hard-real category.
- JPEG, resize, crop, caption, screenshot, and social-media recompression tests passed.
- The new model beats the deployed model on safety and coverage, not only average accuracy.
- App smoke test confirms the trained models are loaded instead of fallback heuristics.

If a new version catches more AI but produces many more false accusations, do not promote it. Keep the safer model and improve the data first.

# TruthShield Video Detector: Every Step

## Start Here: The Dataset Is Ready

As of July 15, 2026, the local AIGVDBench sample is ready at:

```text
training/data/video_source
```

It contains 1,120 videos with an exactly balanced class split:

| Split | AI-generated | Real | Generator/source policy |
| --- | ---: | ---: | --- |
| Train | 400 | 400 | Open-Sora T2V, AnimateDiff T2V, SVD I2V, CogVideoX 1.5 V2V, and LTX V2V |
| Validation | 80 | 80 | HunyuanVideo T2V and EasyAnimate I2V; not used for fitting |
| Test | 80 | 80 | Pika held out as an unseen AI generator family |

The manifest is `training/data/video_source/aigvdbench_manifest.csv`. It records the source URL, archive member, split, label, generator/source, CC-BY-4.0 license, byte size, and SHA-256 checksum. All 1,120 local videos were checked: no missing files, size mismatches, hash mismatches, or duplicate hashes were found.

The extracted 16-frame-per-video dataset and neural embedding cache also already exist:

```text
training/data/video_frames                       17,920 JPEG frames
training/data/video_frame_embeddings/train.npz  cached training embeddings
training/data/video_frame_embeddings/validation.npz
training/data/video_frame_embeddings/test.npz
```

Do not download the dataset or extract the frames again for the first run below. Do not move files between splits. The test split must remain untouched until the final evaluation.

## Train A Safe Candidate Now

This computer has CPU-only PyTorch, so the following workflow reuses the cached 2,048-dimensional ResNet embeddings. It trains a new regularized frame head and temporal model without overwriting the models currently used by the app.

Open PowerShell and run these commands from the repository root:

```powershell
cd "C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai"
```

### 1. Train a candidate frame detector

```powershell
.\backend\venv\Scripts\python.exe .\training\train_video_frame_detector.py `
  --data-dir .\training\data\video_frames `
  --base-model .\training\models\truthshield-image-detector-v2 `
  --output-dir .\training\models\truthshield-video-frame-detector-v2 `
  --embedding-dir .\training\data\video_frame_embeddings `
  --batch-size 32 `
  --seed 42
```

Expected behavior:

- It should load the three existing `.npz` embedding caches instead of running ResNet over all 17,920 frames again.
- It fits several regularized logistic heads using only `train`.
- It selects regularization and the video decision threshold using `validation`.
- It evaluates `test` once and writes `training/models/truthshield-video-frame-detector-v2/truthshield_video_frame_metrics.json`.

Do not add `--rebuild-embeddings` unless the frame images or base image model changed.

### 2. Rebuild temporal features with the candidate head

```powershell
.\backend\venv\Scripts\python.exe .\training\build_video_features_from_frame_cache.py `
  --frame-dir .\training\data\video_frames `
  --embedding-dir .\training\data\video_frame_embeddings `
  --frame-model .\training\models\truthshield-video-frame-detector-v2 `
  --output .\training\data\video_features_v2.jsonl
```

This creates one ordered feature record for each of the 1,120 source videos. It combines the candidate frame probabilities with luma, noise, edge, optical-flow, duplicate-frame, and scene-cut signals.

### 3. Train a candidate temporal detector

```powershell
.\backend\venv\Scripts\python.exe .\training\train_video_detector.py `
  --features .\training\data\video_features_v2.jsonl `
  --output .\training\models\truthshield-video-temporal-v2.joblib `
  --trees 600 `
  --max-depth 14 `
  --min-samples-leaf 2 `
  --seed 42 `
  --exclude-features pixel_forensic_probability_mean,pixel_forensic_probability_p95,frame_truth_score_mean,frame_truth_score_std,frame_truth_score_p10
```

The trainer compares regularized logistic regression, Random Forest, Extra Trees, and histogram gradient boosting on validation data. It writes the chosen model to `truthshield-video-temporal-v2.joblib` and the complete report to `truthshield-video-temporal-v2.metrics.json`.

### 4. Compare the candidate with the current baseline

The current held-out 160-video test baseline is:

| Metric | Current result |
| --- | ---: |
| Balanced accuracy | 78.75% |
| ROC-AUC | 0.878906 |
| AI recall | 88.75% (71/80) |
| Real specificity | 68.75% (55/80) |
| False AI warnings | 25/80 real videos |
| Missed AI videos | 9/80 AI videos |

Read the candidate report:

```powershell
$candidate = Get-Content .\training\models\truthshield-video-temporal-v2.metrics.json -Raw | ConvertFrom-Json
$candidate.metrics.test | Format-List balanced_accuracy,roc_auc,precision,recall,specificity,f1,false_ai_alarm,missed_ai,brier_score,expected_calibration_error
```

Do not deploy the candidate merely because one number increased. Require lower false-AI warnings, no serious loss of AI recall, and equal or better calibration. A useful safety target for the next data cycle is at most 8 false warnings out of 80 real test videos (90% specificity) while keeping AI recall at or above 85%.

With the same videos, frames, base model, and seed, this run will probably reproduce the existing result. That proves the training pipeline is working, but it is not evidence of a more accurate model.

## Make The Next Run Genuinely Better

Accuracy improvements must come from broader, carefully matched data rather than repeatedly tuning against the Pika test set.

1. Leave the existing 160-video test split locked. Never use it to choose features, hyperparameters, or thresholds.
2. Add at least 400 hard real videos to `train/real_camera`: animation, CGI, game footage, screen recordings, sports, water, smoke, reflections, low light, camera shake, filters, transitions, and heavily compressed phone videos.
3. Add an equal number of AI videos to `train/ai_generated` from generator families not already represented in training. Include text-to-video, image-to-video, and video-to-video output.
4. Add at least 100 real and 100 AI videos from separate source groups to validation. Do not reuse originals, clips, or transformations from training.
5. Create a new locked challenge test with at least 200 real and 200 AI videos from sources and AI generator families absent from train and validation.
6. Match duration, resolution, aspect ratio, frame rate, codec, and compression between classes. Apply crops, captions, resizing, speed changes, and recompression to both classes.
7. Keep every clip and transformation derived from one original video in the same split.
8. Record each file's origin, license, generator/camera source, transformation, split, SHA-256 hash, duration, frame rate, resolution, and codec in a manifest.

After adding data, extract a new frame set and rebuild embeddings so the current reproducible cache stays intact:

```powershell
.\backend\venv\Scripts\python.exe .\training\prepare_video_frames.py `
  --source-dir .\training\data\video_source_v2 `
  --output-dir .\training\data\video_frames_v2 `
  --frame-stride 1 `
  --max-frames-per-video 16 `
  --jpeg-quality 92 `
  --workers 1 `
  --clean-output

.\backend\venv\Scripts\python.exe .\training\train_video_frame_detector.py `
  --data-dir .\training\data\video_frames_v2 `
  --base-model .\training\models\truthshield-image-detector-v2 `
  --output-dir .\training\models\truthshield-video-frame-detector-v3 `
  --embedding-dir .\training\data\video_frame_embeddings_v2 `
  --batch-size 32 `
  --seed 42 `
  --rebuild-embeddings
```

Then repeat the temporal feature and training steps with `v3` paths. Run the new challenge test only after model and threshold choices are frozen. If you inspect its failures and change the model, that challenge set becomes development data and you need a new final test.

This guide assumes you are using Windows PowerShell and the project is here:

```text
C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai
```

TruthShield's video detector has three cooperating parts:

1. The image model checks the complete frame.
2. A tiled scan overlaps smaller regions so every source pixel belongs to at least one inspected tile.
3. Temporal checks compare every neighboring frame for motion-warp errors, noise flicker, edge flicker, duplicate frames, scene cuts, and sustained AI signals.

The native-resolution forensic code processes each complete decoded frame. Neural image models still resize frames or tiles to their learned input size. No normal classifier can literally keep a full 4K frame at native size without enormous memory, so TruthShield reports this distinction honestly.

## Part A: Install The Tools

1. Open PowerShell.
2. Paste this command and press Enter:

```powershell
cd "C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai"
```

3. Paste this command and press Enter:

```powershell
.\backend\venv\Scripts\python.exe -m pip install -r .\training\requirements-train.txt
```

You need `scikit-learn` and `joblib` in addition to the image-training packages. They train and save the video-level temporal model.

## Part B: Get Videos Without Paying

Use only data whose license and rules allow your project. Keep a note containing the source URL and license for every dataset.

Good official starting points:

- AIGVDBench (CC-BY-4.0): https://huggingface.co/datasets/AIGVDBench/AIGVDBench
- GenVideo / DeMamba: https://github.com/chenhaoxing/DeMamba
- DeCoF generated-video dataset: https://github.com/LongMa-2025/DeCoF
- GenVidBench: https://github.com/genvidbench/GenVidBench
- GenVidBench files: https://huggingface.co/datasets/jian-0/GenVidBench
- FaceForensics++ for face manipulation: https://github.com/ondyari/FaceForensics

Do not download all of GenVidBench onto this laptop. Its Hugging Face page currently reports about 216 GB. Start with a small, balanced subset.

TruthShield includes a resumable AIGVDBench sampler. It reads only selected files from the official remote ZIP archives, instead of downloading archives as large as 94 GB. Run:

```powershell
.\backend\venv\Scripts\python.exe .\training\prepare_aigvdbench_sample.py --train-per-source 80 --validation-per-source 40 --test-per-source 80
```

That creates 1,120 videos: 560 AI and 560 real. Open-Sora T2V, AnimateDiff T2V, SVD I2V, CogVideoX 1.5 V2V, and LTX V2V are used for training. HunyuanVideo T2V and EasyAnimate I2V are used for validation. The closed-source Pika family is held entirely out for the final cross-generator test. This covers text-to-video, image-to-video, and the harder video-to-video task without sharing generator families between splits. Real clips sharing the same original source ID are kept in one split to prevent leakage. The source URL, archive member, license, checksum, and split are recorded in `training/data/video_source/aigvdbench_manifest.csv`.

A practical first target is:

- 300 AI-generated videos.
- 300 real videos.
- About 2 to 10 seconds per video.
- Several resolutions, frame rates, subjects, and compression levels in both classes.
- At least three AI generator families. Keep one entire generator family out of training for the final test.

For free real videos, record short clips with your phone. Record indoor scenes, outdoor scenes, people, pets, moving objects, camera pans, low light, bright light, and text on screens. Do not record private information or people who did not agree to be recorded.

Important: make the two classes fair. If every AI video is 8 FPS and every real video is 30 FPS, the model may learn frame rate instead of AI artifacts. If every AI video is square and every real video is widescreen, it may learn shape instead of AI artifacts.

## Part C: Make The Exact Folders

Open File Explorer and go to:

```text
C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai\training\data
```

Create this exact folder tree:

```text
video_source
|-- train
|   |-- ai_generated
|   `-- real_camera
|-- validation
|   |-- ai_generated
|   `-- real_camera
`-- test
    |-- ai_generated
    `-- real_camera
```

Use these rules:

- Put about 70 percent of each class in `train`.
- Put about 15 percent in `validation`.
- Put about 15 percent in `test`.
- One video may appear in only one split.
- Different clips cut from the same original video must stay in the same split.
- Keep at least one AI generator entirely in `test`. This tells you whether the detector handles a generator it never memorized.
- Keep the `test` folder locked away mentally. Never move test mistakes into train and then call the same test score independent.

Example for 300 videos in each class:

```text
train/ai_generated: 210
train/real_camera: 210
validation/ai_generated: 45
validation/real_camera: 45
test/ai_generated: 45
test/real_camera: 45
```

## Part D: Train A Video-Frame Model

This teaches a neural image model the appearance of frames that came from real versus AI-generated videos.

1. Extract balanced frames across each complete video:

```powershell
cd "C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai"
.\backend\venv\Scripts\python.exe .\training\prepare_video_frames.py --frame-stride 1 --max-frames-per-video 16 --clean-output
```

The 16 frames are spread uniformly from the beginning through the end of each video. The benchmark's shortest generator clips are 16 frames, so this gives every video and class equal weight while still covering each full timeline. The 1,120-video sample produces 17,920 frames. A 10-second, 30-FPS video has about 300 frames, so extracting every training frame from hundreds of videos can otherwise create hundreds of thousands of near-duplicates.

For a first pipeline test, use every fifth frame instead:

```powershell
.\backend\venv\Scripts\python.exe .\training\prepare_video_frames.py --frame-stride 5 --clean-output
```

Inference can still inspect every frame even when training uses every fifth frame. Adjacent training frames are often nearly identical, so using stride 3 to 5 can reduce memorization and training time.

2. Run a one-epoch smoke test:

```powershell
.\backend\venv\Scripts\python.exe .\training\train_image_detector.py --data-dir .\training\data\video_frames --output-dir .\training\models\truthshield-video-frame-detector-smoke --epochs 1 --batch-size 4 --max-train-samples 600 --max-eval-samples 200
```

3. Train the real frame model on a CPU laptop:

```powershell
.\backend\venv\Scripts\python.exe .\training\train_video_frame_detector.py --data-dir .\training\data\video_frames --base-model .\training\models\truthshield-image-detector-v2 --output-dir .\training\models\truthshield-video-frame-detector --batch-size 32
```

This CPU-friendly trainer runs the validated ResNet forensic backbone once per frame, caches its 2,048-dimensional embeddings, compares several regularized binary heads on generator-separated validation videos, inserts the best head back into a normal Hugging Face ResNet model, and evaluates the untouched Pika test family. If interrupted after one split is cached, rerun the same command and it resumes from that cache.

On a CUDA GPU, a later optional full-backbone fine-tune can use `train_image_detector.py`. On a CPU-only laptop, the embedding/head method is much faster and preserves the already validated forensic backbone instead of spending many hours repeating full ResNet backpropagation.

## Part E: Build Temporal Features

The CPU-friendly path reuses the cached neural embeddings and runs the production temporal accumulator over the 16 uniformly spaced frames already extracted from each source video:

```powershell
.\backend\venv\Scripts\python.exe .\training\build_video_features_from_frame_cache.py --frame-model .\training\models\truthshield-video-frame-detector
```

This preserves ordered luma, noise, edge, optical-flow, duplicate-frame, and scene-cut features without repeating the ResNet pass. Static forensic/truth columns that are not reproduced by this cache path are kept neutral and excluded during training.

For a later GPU/server experiment, the slower exact extraction path opens every source video and can analyze every original frame and tile:

```powershell
.\backend\venv\Scripts\python.exe .\training\extract_video_features.py --frame-model .\training\models\truthshield-video-frame-detector --frame-stride 1 --tile-analysis
```

The command saves completed videos after each one. If PowerShell closes, run the same command again. Resume mode skips videos already finished.

On a CPU laptop, tile analysis over every frame may take hours or days for hundreds of videos. That is normal. To prove the pipeline works first, run:

```powershell
.\backend\venv\Scripts\python.exe .\training\extract_video_features.py --frame-model .\training\models\truthshield-video-frame-detector --frame-stride 1 --no-tile-analysis --max-frames 60
```

Delete `training\data\video_features.jsonl` before the final full run, because features made with different settings should not be mixed.

## Part F: Train The Temporal Video Model

After the CPU-friendly feature build finishes, run:

```powershell
.\backend\venv\Scripts\python.exe .\training\train_video_detector.py --features .\training\data\video_features_from_frame_cache.jsonl --exclude-features pixel_forensic_probability_mean,pixel_forensic_probability_p95,frame_truth_score_mean,frame_truth_score_std,frame_truth_score_p10
```

It trains only on records marked `train`. It compares Random Forest, Extra Trees, gradient boosting, and regularized logistic models on `validation`, chooses a decision threshold there, and reports the final score once on the held-out `test` family. The report also includes ROC-AUC, average precision, calibration error, Brier score, and permutation feature importance.

The current 1,120-video run selected logistic regression. On the untouched 160-video Pika/real test split it measured 78.75% balanced accuracy, 0.878906 ROC-AUC, 88.75% AI recall, and 68.75% real-video specificity. Those figures are useful evidence, not a guarantee; the test still contained 25 false AI alarms and 9 missed AI videos.

The output files are:

```text
training/models/truthshield-video-temporal.joblib
training/models/truthshield-video-temporal.metrics.json
```

Do not download random `.joblib` files. Loading one can execute code. Only load the file you trained or a file from someone you fully trust.

When both files use the standard names shown above, TruthShield discovers them automatically. Environment variables are still useful when comparing alternate model versions.

## Part G: Start TruthShield With Both Video Models

Open PowerShell and run:

```powershell
cd "C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai\backend"
$env:ENABLE_LOCAL_AI_MODELS="true"
$env:AI_VIDEO_FRAME_DETECTOR_MODELS="C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai\training\models\truthshield-video-frame-detector"
$env:AI_VIDEO_TEMPORAL_MODEL_PATH="C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai\training\models\truthshield-video-temporal.joblib"
$env:VIDEO_ANALYSIS_MODE="exhaustive"
$env:VIDEO_FRAME_STRIDE="1"
$env:VIDEO_MAX_FRAMES="0"
$env:VIDEO_TILE_ANALYSIS="true"
.\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

These values mean:

- `exhaustive`: decode the video in order instead of selecting eight samples.
- frame stride `1`: analyze every decoded frame.
- max frames `0`: do not stop early.
- tile analysis `true`: cover the complete frame with overlapping model tiles.

Runtime coverage and learned-model sampling are deliberately separate. Native forensics and tiles still analyze every decoded frame. The trained temporal classifier consumes up to 16 uniformly spaced full-frame neural signals because that is the distribution on which it was validated; feeding it an arbitrary number of tile-combined signals would invalidate the reported metrics.

## Part H: Decide Whether It Is Actually Better

Do not judge the model with one video. Read the `.metrics.json` file and check:

- `false_ai_alarm`: real videos wrongly accused of being AI.
- `missed_ai`: AI videos wrongly called real.
- `balanced_accuracy`: average success across both classes.
- `precision`: how often an AI warning was correct.
- `recall`: how much AI video it caught.

Then make a challenge folder that was never used for training:

- Real phone videos uploaded to and downloaded from social media.
- Real screen recordings and video-game footage.
- Real animation and CGI, because "computer-made" does not automatically mean generative AI.
- AI videos with captions, logos, crops, and recompression.
- AI videos from a generator absent from train and validation.
- Fast sports, water, smoke, crowds, reflections, hands, faces, and camera cuts.

Every time the model is confidently wrong, add similar examples to the next training version. Never move the exact challenge file into training and then reuse it as the final test.

## Reality Check

No detector can guarantee that a video is real or AI-generated. Generator artifacts change, editing can erase them, and real compression can imitate them. TruthShield combines neural frame evidence, complete-frame pixel forensics, temporal consistency, metadata, provenance when available, and source research because disagreement between independent clues is more useful than pretending one score is proof.

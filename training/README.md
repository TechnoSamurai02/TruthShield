# TruthShield Model Training

For the full beginner-friendly video workflow, use [VIDEO_TRAINING_GUIDE.md](./VIDEO_TRAINING_GUIDE.md).

For the recommended next accuracy cycle for both models, including current baselines and exact commands, use [ACCURACY_IMPROVEMENT_PLAN.md](./ACCURACY_IMPROVEMENT_PLAN.md).

The video workflow now includes `prepare_aigvdbench_sample.py`, which built the balanced 2,080-video dataset used by the packaged video models directly from the official CC-BY-4.0 AIGVDBench archives using resumable HTTP ranges. It contains 1,600 training videos, 320 generator-separated validation videos, and an unchanged 160-video closed-source Pika/real test split. It covers T2V, I2V, and V2V families without downloading the complete benchmark.

`train_video_frame_detector.py` provides the CPU training path: cached ResNet forensic embeddings plus validation-selected regularized heads exported back into a standard Hugging Face image-classification model.

`build_video_features_from_frame_cache.py` then reuses those embeddings while applying the production optical-flow/temporal accumulator to the ordered 16-frame sequences. The deployed analyzer still performs native and tiled analysis on every decoded frame; only the learned temporal classifier uses the validated 16-frame representation.

This folder is for training a custom image detector that TruthShield can use as one more evidence signal.

The goal is not to make a magic truth machine. The goal is to teach a model the difference between:

- `real_camera`: real photos.
- `real_edited_or_captioned`: real photos that were cropped, captioned, memed, screenshotted, or recompressed.
- `ai_generated`: images created by AI.

That middle class matters because a real meme image with text on it should not be treated the same as a fully AI-generated image.

## Good Starter Datasets

Use datasets only when their license/rules allow your use.

1. Defactify Image Dataset
   Link: https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset
   Why it is useful: 96,000 images, `train`/`validation`/`test` splits, real-vs-AI labels, and AI-source labels. The dataset card says real images come from MS COCO and AI images come from Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL-E 3, and Midjourney v6.

2. CIFAKE
   Link: https://arxiv.org/abs/2303.14126
   Why it is useful: easy real-vs-fake practice dataset. Weakness: images are tiny, so it is not enough by itself for real-world uploads.

3. RealHD
   Link: https://real-hd.github.io/
   Why it is useful: newer, large, higher-quality AI-image detection dataset. Weakness: very large, so use it later after your pipeline works.

4. Your own images
   Use your own phone photos for `real_camera`. Make captioned/meme versions for `real_edited_or_captioned`. Use AI images you are allowed to use for `ai_generated`.

Avoid random Google Images downloads unless you know the license lets you train on them.

## Step 1: Install Training Tools

Open PowerShell:

```powershell
cd "C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai"
cd training
..\backend\venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

This may take a while because `torch` is large.

## Step 2: Download A Small Defactify Sample

Start small. Do not train on all 96,000 images first.

```powershell
cd "C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai"
.\backend\venv\Scripts\python.exe .\training\prepare_defactify_sample.py --max-per-label 800 --clean-output
```

This creates:

```text
training/data/defactify_sample/train/real_camera
training/data/defactify_sample/train/ai_generated
training/data/defactify_sample/validation/real_camera
training/data/defactify_sample/validation/ai_generated
training/data/defactify_sample/test/real_camera
training/data/defactify_sample/test/ai_generated
```

## Step 3: Add Captioned Real Images

This makes fake-looking captions on top of real photos. These are still real photos, so they go into `real_edited_or_captioned`.

```powershell
cd "C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai"
.\backend\venv\Scripts\python.exe .\training\make_captioned_real_variants.py --max-per-split 400
```

## Step 4: Train The Model

Start with a small test run:

```powershell
cd "C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai"
.\backend\venv\Scripts\python.exe .\training\train_image_detector.py --epochs 1 --batch-size 4 --max-train-samples 600 --max-eval-samples 200
```

If that works, run a real training pass:

```powershell
.\backend\venv\Scripts\python.exe .\training\train_image_detector.py --epochs 3 --batch-size 8
```

By default, the script fine-tunes `microsoft/resnet-50`, a smaller image-classification model than the original ViT default. This is friendlier for normal laptops and free CPU hosting.

The trained model is saved here:

```text
training/models/truthshield-image-detector
```

The improved trainer now uses realistic crops, color changes, blur, and JPEG recompression, keeps only a few checkpoints, selects the best checkpoint by macro F1 instead of majority-class accuracy, stops when validation stops improving, and evaluates the untouched `test` split automatically.

## Step 5: Audit And Evaluate It

Check that identical images did not leak between train, validation, and test:

```powershell
.\backend\venv\Scripts\python.exe .\training\audit_image_dataset.py
```

Run a full independent report for an already-trained model:

```powershell
.\backend\venv\Scripts\python.exe .\training\evaluate_image_detector.py --model-dir .\training\models\truthshield-image-detector-v2 --split test --batch-size 16
```

The report includes accuracy, balanced accuracy, macro F1, per-class precision/recall, false AI alarms, missed AI images, ROC AUC, calibration error, the former 0.70 frontend rule, conservative three-way decisions, inconclusive coverage, and example mistakes. See `training/evaluation/README.md` for the balanced decision-level workflow.

Check a known AI image across JPEG compression, resizing, cropping, and a social-media-style resize/recompression:

```powershell
.\backend\venv\Scripts\python.exe .\training\check_image_robustness.py "C:\path\to\known-ai-image.png"
```

This is a regression check, not a substitute for a large generator-separated test set. Keep regression images out of training so the check remains honest.

## Step 6: Tell TruthShield To Use The Model

In the same terminal where you start the backend:

```powershell
$env:AI_IMAGE_DETECTOR_MODELS="C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai\training\models\truthshield-image-detector"
$env:ENABLE_LOCAL_AI_MODELS="true"
cd "C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai\backend"
.\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

Now upload images in TruthShield and check the Detector Opinions section.

TruthShield automatically prefers a valid local `truthshield-image-detector-v3`, then `v2`, then the original model. Setting the environment variable explicitly is still useful when comparing models.

For production, deploy from the repository root with the root `Dockerfile`. Deploying only `backend/` leaves `training/models/truthshield-image-detector-v2` out of the container and forces a weaker fallback. See `DEPLOYMENT.md`.

## Step 7: What Counts As A Good Model?

Do not trust training accuracy alone.

Test with images the model never saw:

- real phone photos
- real photos with text captions
- screenshots
- memes
- compressed social media images
- AI images from several generators

A useful model should not call every meme or screenshot AI.

## Simple Rule

Bad dataset equals bad model.

If your training images are too clean, the model learns clean-vs-messy instead of real-vs-AI. Add messy real images on purpose.

For the generator-isolated four-way pipeline, constrained calibration, promotion gates, and Lightning/Colab workflow, see [V4_ACCURACY_GUIDE.md](V4_ACCURACY_GUIDE.md).

The mask-supervised manipulation path is implemented by:

- `prepare_diffusion_manipulation_pairs.py`: resumable, split-isolated diffusion edits with exact masks and per-model license records.
- `train_manipulation_localizer.py`: a lightweight LR-ASPP MobileNetV3 pixel localizer that trains only on `train` and selects checkpoints only on `tuning`.
- `evaluate_manipulation_localizer.py`: full-image tuning/calibration scoring with localized connected-component support and controlled JPEG/resize stability checks. It refuses to read `locked_test` unless the final-gate flag is explicitly supplied.

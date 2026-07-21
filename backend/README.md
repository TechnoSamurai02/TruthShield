---
title: TruthShield AI Backend
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# TruthShield AI Backend

FastAPI backend for TruthShield AI.

## Endpoints

- `GET /api/health`
- `POST /api/analyze/image`
- `POST /api/analyze/video`
- `POST /api/analyze/text`

## Environment Variables

Set `FRONTEND_ORIGINS` to the comma-separated list of deployed frontend origins that may call this API.

Example:

```text
FRONTEND_ORIGINS=https://truthshield-ai-frontend.pages.dev,https://www.example.com
```

Cloudflare Pages preview and production domains ending in `.pages.dev` are allowed by default through `FRONTEND_ORIGIN_REGEX`.

Enhanced verification stays free-only and is controlled with:

```text
ENABLE_ENHANCED_ANALYSIS=true
BRAVE_SEARCH_API_KEY=
WEB_RESEARCH_PER_SCAN_LIMIT=2
WEB_RESEARCH_MONTHLY_LIMIT=150
GOOGLE_VISION_API_KEY=
GOOGLE_VISION_MAX_RESULTS=10
ENABLE_LOCAL_AI_MODELS=true
AI_IMAGE_DETECTOR_MODELS=Organika/sdxl-detector,dima806/deepfake_vs_real_image_detection
AI_MANIPULATION_DETECTOR_MODELS=
AI_MANIPULATION_LOCALIZER_PATH=
MEDIA_DECISION_POLICY_PATH=
IMAGE_TRANSFORMATION_CHECKS=true
IMAGE_TILE_ANALYSIS=true
IMAGE_TILE_SIZE=448
IMAGE_TILE_OVERLAP=0.15
AI_VIDEO_FRAME_DETECTOR_MODELS=
AI_VIDEO_TEMPORAL_MODEL_PATH=
VIDEO_ANALYSIS_MODE=adaptive
VIDEO_FRAME_STRIDE=1
VIDEO_MAX_FRAMES=0
VIDEO_KEYFRAME_MAX=64
VIDEO_WINDOW_MAX=8
VIDEO_TILE_ANALYSIS=true
COMMUNITY_FORENSICS_REPO_PATH=
COMMUNITY_FORENSICS_MODEL_ID=OwensLab/commfor-model-224
LOCAL_REASONING_BASE_URL=
```

`AI_MANIPULATION_LOCALIZER_PATH` accepts the directory containing the calibrated TruthShield `model.ts` and `preprocess.json` artifacts. The localizer reports a score and suspicious region, while the shared policy and controlled-view stability check remain responsible for the verdict. `IMAGE_TILE_ANALYSIS=true` runs configured classifier specialists over overlapping full-coverage tiles when an image is larger than one tile. Only calibrated high-scoring regions become suspicious evidence. `VIDEO_ANALYSIS_MODE=adaptive` decodes the complete video cheaply, selects up to 64 diverse/anomalous frames, and builds up to eight temporal windows. `exhaustive` remains available for rollback and debugging. `VIDEO_TILE_ANALYSIS=true` adds overlapping full-coverage model tiles and can be slow on a CPU. Image and video responses use the same four-way policy. Missing providers, metadata, manifests, scores, or web matches return neutral/unavailable signals and never default to AI.

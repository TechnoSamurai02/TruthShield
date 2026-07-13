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
AI_VIDEO_FRAME_DETECTOR_MODELS=
AI_VIDEO_TEMPORAL_MODEL_PATH=
VIDEO_ANALYSIS_MODE=exhaustive
VIDEO_FRAME_STRIDE=1
VIDEO_MAX_FRAMES=0
VIDEO_TILE_ANALYSIS=true
LOCAL_REASONING_BASE_URL=
```

`VIDEO_ANALYSIS_MODE=exhaustive`, `VIDEO_FRAME_STRIDE=1`, and `VIDEO_MAX_FRAMES=0` process every decoded frame. `VIDEO_TILE_ANALYSIS=true` adds overlapping full-coverage model tiles and can be very slow on a CPU. `GOOGLE_VISION_API_KEY`, `BRAVE_SEARCH_API_KEY`, trained model paths, C2PA tooling, and a local reasoning endpoint are optional. The API still returns a valid report when any of them are absent.

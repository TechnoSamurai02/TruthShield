---
title: TruthShield AI Backend
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# TruthShield AI

TruthShield AI helps people inspect suspicious images and videos before they trust or share them. Image reports combine available metadata, a dedicated detector, supporting pixel checks, provenance, and honest web context into a conservative three-way assessment.

Image outcomes are `Likely authentic`, `Inconclusive`, or `Likely AI-generated or manipulated`. Raw detector scores are model outputs, not proof or calibrated real-world probabilities.

## Product Experience

- Image analysis for `.jpg`, `.jpeg`, `.png`, and `.webp`
- Video analysis for `.mp4`, `.mov`, and `.webm`
- Drag-and-drop and keyboard-accessible file selection
- Explainable three-way image verdict with evidence on both sides and explicit limitations
- Practical safety recommendations and development-mode intermediate-signal diagnostics
- Packaged, held-out-tested image detector plus supporting pixel forensics
- Video frame coverage and suspicious-frame evidence
- Optional C2PA provenance checks and indexed web research
- Light and dark editorial themes with persisted preference
- Responsive layouts for desktop, tablet, and mobile

## Tech Stack

Frontend:

- React 18
- Vite
- TypeScript
- Tailwind CSS

Backend:

- Python
- FastAPI
- Pillow
- OpenCV
- NumPy
- Pydantic

## Run Locally

Start the backend:

```bash
cd backend
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Start the frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The frontend defaults to `http://localhost:8000`. Override it with:

```text
VITE_API_BASE_URL=https://your-api.example.com
```

## API

### `GET /api/health`

Returns the backend service status.

### `POST /api/analyze/image`

Accepts a multipart upload under the `file` field and returns:

- `content_type`
- `truth_score`
- `risk_level`
- `verdict`
- `summary`
- `warnings`
- `positive_signals`
- `recommendations`
- `evidence`
- `technical_details`
- Structured `assessment` with independent signals, evidence, reliability, and limitations
- Optional detector, provenance, research, citation, and feedback fields

### `POST /api/analyze/video`

Accepts a multipart upload under the `file` field. It temporarily stores the upload, analyzes decoded frames, aggregates temporal and forensic evidence, returns the standard analysis fields plus `frames_analyzed` and `suspicious_frames`, and then removes the temporary file.

## Optional Enhanced Verification

Enhanced analysis is enabled by default and degrades gracefully when optional tools are unavailable.

```text
ENABLE_ENHANCED_ANALYSIS=true
BRAVE_SEARCH_API_KEY=
WEB_RESEARCH_PER_SCAN_LIMIT=2
WEB_RESEARCH_MONTHLY_LIMIT=150
GOOGLE_VISION_API_KEY=
GOOGLE_VISION_MAX_RESULTS=10
ENABLE_LOCAL_AI_MODELS=true
AI_IMAGE_DETECTOR_MODELS=
AI_VIDEO_FRAME_DETECTOR_MODELS=
AI_VIDEO_TEMPORAL_MODEL_PATH=
VIDEO_ANALYSIS_MODE=exhaustive
VIDEO_FRAME_STRIDE=1
VIDEO_MAX_FRAMES=0
VIDEO_TILE_ANALYSIS=true
LOCAL_REASONING_BASE_URL=
```

- Google Cloud Vision Web Detection can provide full, partial, visually similar, and page-level image matches.
- Brave Search can provide indexed source leads and citations; when uploaded-image matching returns visual labels or web entities but no direct match, those clues are carried into the indexed search instead of relying only on the filename.
- The root Docker deployment packages `truthshield-image-detector-v2`; its raw score is evaluated through conservative abstention rules rather than used directly as the verdict.
- Missing metadata, absent C2PA credentials, missing web matches, timeouts, null scores, and provider errors are neutral or unavailable—not AI evidence.

### Image decision policy

- Dedicated-detector score `<= 0.15`: likely authentic when no independent positive AI evidence conflicts.
- Score between `0.15` and `0.95`: inconclusive unless camera metadata or verified provenance safely corroborates a detector score `<= 0.30`.
- Score `>= 0.95`: likely AI-generated/manipulated unless camera metadata or verified provenance creates an unresolved conflict.
- An explicit AI-generator software metadata tag is positive AI evidence; filenames and missing metadata are not.

The thresholds operate on the packaged model's raw AI-class score. They are not probability statements.

## Legacy context score

- `80–100`: High Trust
- `60–79`: Medium Trust
- `40–59`: Low Trust
- `0–39`: High Risk

This numeric field remains for API compatibility and video/context workflows. It does not determine the image authenticity verdict.

## File Safety

- Image extensions are validated and uploads are limited to 15 MB.
- Video extensions are validated and uploads are limited to 80 MB.
- Uploaded files are never executed.
- Temporary video files are removed after analysis.
- Backend failures return safe user-facing messages.

## Limitations

TruthShield AI is not a forensic laboratory. A wide abstention band is intentional because polished real photographs and realistic generated images can both fool pixel-only models. Missing metadata does not prove fabrication, normal metadata does not prove authenticity, and web matches provide context rather than proof. Use the report as a first check and verify important claims with trusted independent sources.

Deployment guidance is available in [DEPLOYMENT.md](./DEPLOYMENT.md).

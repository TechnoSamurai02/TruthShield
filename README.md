---
title: TruthShield AI Backend
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# TruthShield AI

TruthShield AI helps people inspect suspicious images and videos before they trust or share them. Image and video reports use a shared, conservative, abstention-first four-way assessment.

Media outcomes are `Likely authentic`, `Likely AI-generated`, `Likely AI-edited/manipulated`, or `Inconclusive`. Raw detector scores are model outputs, not proof or real-world probabilities.

## Product Experience

- Image analysis for `.jpg`, `.jpeg`, `.png`, and `.webp`
- Video analysis for `.mp4`, `.mov`, and `.webm`
- Drag-and-drop and keyboard-accessible file selection
- Explainable four-way image/video assessment with generation and manipulation evidence shown separately
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

Returns service status, policy/calibration IDs, model artifact checksums, and decisive-verdict capabilities.

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
AI_MANIPULATION_DETECTOR_MODELS=
MEDIA_DECISION_POLICY_PATH=
IMAGE_TRANSFORMATION_CHECKS=true
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

- Google Cloud Vision Web Detection can provide full, partial, visually similar, and page-level image matches.
- Brave Search can provide indexed source leads and citations; when uploaded-image matching returns visual labels or web entities but no direct match, those clues are carried into the indexed search instead of relying only on the filename.
- The root Docker deployment packages `truthshield-image-detector-v2`; its raw score is evaluated through conservative abstention rules rather than used directly as the verdict.
- Missing metadata, absent C2PA credentials, missing web matches, timeouts, null scores, and provider errors are neutral or unavailable—not AI evidence.

### Media decision policy

- A generated verdict requires the calibrated generation score to cross its upper threshold and remain stable across views/windows.
- A manipulated verdict requires a dedicated specialist plus localized or temporally persistent support.
- An authentic verdict requires low generation and manipulation scores, completed required checks, sufficient quality, and no conflict.
- All other cases are inconclusive. Explicit software metadata is separate positive evidence but cannot bypass a missing or sub-threshold generation specialist.

Web matches, missing metadata, compression, and the deprecated `truth_score` are never blended into the AI decision. See `training/V4_ACCURACY_GUIDE.md` for generator-isolated calibration and promotion.

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

# TruthShield AI

Tagline: "Don't trust blindly. Verify intelligently."

TruthShield AI is a Hoobit Hacks 2026 MVP for the theme **AI APOCALYPSE**. It helps users run a first-pass risk check on suspicious images, videos, and text posts before sharing them.

In the AI apocalypse, the biggest threat may not be robots. It may be losing the ability to know what is real.

## What The App Does

TruthShield AI accepts an uploaded image, uploaded video, or pasted text post. It returns a clear Truth Score from 0-100, a risk level, a verdict, warning signs, positive signals, evidence breakdown bars, and practical recommendations.

The app is intentionally honest: it does not claim to prove that something is fake or real. It provides a risk-based analysis from available heuristic signals.

## Features

- Image upload analysis for `.jpg`, `.jpeg`, `.png`, and `.webp`
- Video upload analysis for `.mp4`, `.mov`, and `.webm`
- Video frame extraction with up to 8 sampled frames
- Text/post analysis with demo examples for judges
- Truth Score, risk level, verdict, summary, and recommendations
- Evidence breakdown for metadata, visual consistency, compression, language risk, and claim risk
- Free enhanced verification with local synthetic-image signals, optional C2PA provenance checks, optional free-capped Brave Search research, citations, and custom feedback
- AI apocalypse command-center UI
- Local-only MVP with no paid API keys required
- Safe file handling with temporary video storage and cleanup
- Clear disclaimer on every result

## Tech Stack

Frontend:

- React
- Vite
- TypeScript
- Tailwind CSS
- lucide-react icons

Backend:

- Python
- FastAPI
- Pillow
- OpenCV
- NumPy
- python-multipart
- Pydantic

## Run The Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The backend runs at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/api/health
```

### Optional Free Enhanced Verification

Enhanced analysis is enabled by default and remains free. If optional tools are missing, TruthShield falls back to deterministic local signals and explains what was unavailable in the report.

Backend environment variables:

```text
ENABLE_ENHANCED_ANALYSIS=true
BRAVE_SEARCH_API_KEY=
WEB_RESEARCH_PER_SCAN_LIMIT=2
WEB_RESEARCH_MONTHLY_LIMIT=150
ENABLE_LOCAL_AI_MODELS=true
AI_IMAGE_DETECTOR_MODELS=dima806/deepfake_vs_real_image_detection
LOCAL_REASONING_BASE_URL=
```

Notes:

- `BRAVE_SEARCH_API_KEY` is optional. When present, TruthShield uses free-capped Brave Web/Image Search requests for source leads and citations. When absent or quota-limited, analysis stays local.
- Local Hugging Face detectors are optional and free, but require `transformers` and `torch` in the backend environment and may download model files. If they are unavailable, TruthShield uses a deterministic synthetic-likelihood fallback.
- C2PA provenance checks use `c2pa-python` or `c2patool` if available. Missing content credentials are treated as a risk signal, not proof that media is fake.
- Free web research is indexed web/image search from generated queries. It is not an unlimited exact reverse image search across the entire internet.

## Run The Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at:

```text
http://localhost:5173
```

The frontend assumes the backend is running at `http://localhost:8000`. To change this, create a frontend `.env` file with:

```text
VITE_API_BASE_URL=http://localhost:8000
```

## Deployment

Deployment notes for Cloudflare Pages and Hugging Face Spaces are in [DEPLOYMENT.md](./DEPLOYMENT.md).

## API Endpoints

### GET `/api/health`

Returns:

```json
{
  "status": "ok",
  "service": "TruthShield AI Backend"
}
```

### POST `/api/analyze/image`

Input: multipart file upload

Returns image analysis:

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
- `analysis_mode`
- `confidence`
- `detectors`
- `provenance`
- `web_research`
- `citations`
- `custom_feedback`
- `disclaimer`

### POST `/api/analyze/video`

Input: multipart file upload

Behavior:

- Saves upload temporarily
- Extracts up to 8 frames
- Runs frame-level image heuristics
- Averages frame results
- Deletes the temporary file

Also returns:

- `frames_analyzed`
- `suspicious_frames`

### POST `/api/analyze/text`

Input:

```json
{
  "text": "Suspicious post or message..."
}
```

Returns text analysis with source, language, claim, and manipulation risk signals.

## Scoring Model

Truth Score meaning:

- `80-100`: High Trust, likely trustworthy
- `60-79`: Medium Trust, needs light verification
- `40-59`: Low Trust, suspicious / verify before sharing
- `0-39`: High Risk, do not share until verified

Higher scores mean fewer risk signals. Lower scores mean more warning signs.

Image and video checks include:

- EXIF metadata presence
- Camera make/model metadata
- File extension and detected format consistency
- Image dimensions
- Entropy
- Blur / over-smoothing estimate
- Compression and texture consistency
- Local synthetic-image detector signals
- Optional C2PA content credentials check
- Optional free-capped indexed web/image research
- Sampled video frame scores

Text checks include:

- Source or link indicators
- Balanced wording
- Dates, locations, or verifiable details
- Optional free-capped indexed web research for extracted claims
- All-caps ratio
- Urgency and fear language
- Conspiracy-style phrases
- Excessive punctuation
- Strong claims without evidence

## Limitations

TruthShield AI is a hackathon MVP, not a forensic system. Its signals are explainable heuristics and can be wrong. Missing metadata does not prove an image is fake, and normal metadata does not prove an image is real. Video frame sampling can miss important moments. Text analysis can identify risky language patterns, but it cannot independently verify real-world facts.

Every result includes this disclaimer:

> TruthShield AI provides a risk-based analysis, not a final proof that content is real or fake. Use this tool as a first check and verify important claims with trusted sources.

## Security And File Handling

- Upload extensions are validated.
- Image uploads are limited to 15 MB.
- Video uploads are limited to 80 MB.
- Uploaded videos are saved only to a temporary file for analysis.
- Temporary files are deleted after analysis.
- Uploaded files are never executed.
- Backend errors return safe, user-friendly messages.

## Demo Script

1. Start the backend with `uvicorn main:app --reload --port 8000`.
2. Start the frontend with `npm run dev`.
3. Open `http://localhost:5173`.
4. Click **Text Scan**.
5. Load the suspicious viral post example and run analysis.
6. Point out the low score, urgency warnings, no-source warning, and recommendations.
7. Load the normal news-style post and compare the higher score.
8. Upload a sample image and show the metadata, entropy, visual consistency, and compression signals.
9. Upload a short video and show the sampled frame analysis.
10. Emphasize that the app recommends verification and never claims certainty.

## Future Improvements

- Real deepfake detection model integration
- C2PA Content Credentials verification
- Reverse image search integration
- Browser extension
- Source credibility database
- Audio deepfake detection
- Social media share warning plugin
- Educational mode for students and seniors

## Optional API Hooks

This MVP intentionally ships without paid AI API calls. Future integrations with OpenAI, Groq, Gemini, Hugging Face, or other providers should be disabled by default, documented clearly, and treated as additional signals rather than final proof.

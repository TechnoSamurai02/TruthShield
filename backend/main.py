from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from analyzers.image_analyzer import analyze_image_bytes
from analyzers.text_analyzer import analyze_text
from analyzers.video_analyzer import analyze_video_path
from models.schemas import AnalysisResponse, TextAnalysisRequest, VideoAnalysisResponse
from utils.file_utils import (
    IMAGE_EXTENSIONS,
    MAX_IMAGE_BYTES,
    MAX_VIDEO_BYTES,
    VIDEO_EXTENSIONS,
    read_upload_bytes,
    remove_temp_file,
    safe_filename,
    save_upload_to_temp_file,
    validate_extension,
)


app = FastAPI(
    title="TruthShield AI Backend",
    description="Risk-based heuristic analysis for images, videos, and text posts.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "TruthShield AI Backend"}


@app.post("/api/analyze/image", response_model=AnalysisResponse)
async def analyze_image_endpoint(file: UploadFile = File(...)) -> AnalysisResponse:
    validate_extension(file.filename, IMAGE_EXTENSIONS)
    try:
        data = await read_upload_bytes(file, MAX_IMAGE_BYTES)
        result = analyze_image_bytes(data, safe_filename(file.filename))
        result["content_type"] = "image"
        return AnalysisResponse(**result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="We could not analyze this image safely.") from exc


@app.post("/api/analyze/video", response_model=VideoAnalysisResponse)
async def analyze_video_endpoint(file: UploadFile = File(...)) -> VideoAnalysisResponse:
    suffix = validate_extension(file.filename, VIDEO_EXTENSIONS)
    temp_path: str | None = None
    try:
        temp_path = await save_upload_to_temp_file(file, suffix, MAX_VIDEO_BYTES)
        result = analyze_video_path(temp_path, safe_filename(file.filename))
        return VideoAnalysisResponse(**result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="We could not analyze this video safely.") from exc
    finally:
        remove_temp_file(temp_path)


@app.post("/api/analyze/text", response_model=AnalysisResponse)
async def analyze_text_endpoint(payload: TextAnalysisRequest) -> AnalysisResponse:
    try:
        result = analyze_text(payload.text)
        return AnalysisResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="We could not analyze this text safely.") from exc

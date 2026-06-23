from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ContentType = Literal["image", "video", "text"]


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=12000)


class SuspiciousFrame(BaseModel):
    frame_index: int
    timestamp_seconds: Optional[float] = None
    truth_score: int
    warnings: List[str]


class AnalysisResponse(BaseModel):
    content_type: ContentType
    truth_score: int = Field(..., ge=0, le=100)
    risk_level: str
    verdict: str
    summary: str
    warnings: List[str]
    positive_signals: List[str]
    recommendations: List[str]
    evidence: Dict[str, float]
    technical_details: Dict[str, Any]
    disclaimer: str


class VideoAnalysisResponse(AnalysisResponse):
    frames_analyzed: int
    suspicious_frames: List[SuspiciousFrame]

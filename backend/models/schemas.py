from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ContentType = Literal["image", "video", "text"]


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=12000)


class SuspiciousFrame(BaseModel):
    frame_index: int
    source_frame_number: Optional[int] = None
    timestamp_seconds: Optional[float] = None
    truth_score: int
    synthetic_probability: Optional[float] = None
    tile_synthetic_probability: Optional[float] = None
    warnings: List[str]


class DetectorResult(BaseModel):
    name: str
    status: str
    label: Optional[str] = None
    score: Optional[float] = None
    synthetic_probability: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceResult(BaseModel):
    status: str
    score: float = Field(..., ge=0, le=100)
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    title: str
    url: str
    source: Optional[str] = None
    snippet: Optional[str] = None


class WebResearchResult(BaseModel):
    status: str
    provider: str
    score: float = Field(..., ge=0, le=100)
    queries: List[str] = Field(default_factory=list)
    matches_found: int = 0
    summary: str
    citations: List[Citation] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class CustomFeedback(BaseModel):
    headline: str
    explanation: str
    evidence_notes: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


class EvidenceSignal(BaseModel):
    source: str
    status: str
    signal: Literal["authentic", "ai_generated_or_manipulated", "neutral", "unavailable"]
    confidence: float = Field(..., ge=0, le=1)
    reliability: float = Field(..., ge=0, le=1)
    raw_score: Optional[float] = None
    evidence: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class ImageAssessment(BaseModel):
    verdict: Literal["likely_authentic", "inconclusive", "likely_ai_generated_or_manipulated"]
    label: str
    confidence: Literal["low", "moderate", "high"]
    detector_score: Optional[float] = Field(None, ge=0, le=1)
    reason: str
    evidence_supporting_authenticity: List[str] = Field(default_factory=list)
    evidence_raising_concern: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    signals: List[EvidenceSignal] = Field(default_factory=list)


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
    analysis_mode: str = "local_heuristic"
    confidence: float = Field(0.55, ge=0, le=1)
    detectors: List[DetectorResult] = Field(default_factory=list)
    provenance: Optional[ProvenanceResult] = None
    web_research: Optional[WebResearchResult] = None
    citations: List[Citation] = Field(default_factory=list)
    custom_feedback: Optional[CustomFeedback] = None
    assessment: Optional[ImageAssessment] = None


class VideoAnalysisResponse(AnalysisResponse):
    frames_analyzed: int
    suspicious_frames: List[SuspiciousFrame]

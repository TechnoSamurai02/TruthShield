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
    manipulation_probability: Optional[float] = None
    tile_synthetic_probability: Optional[float] = None
    warnings: List[str]
    kind: Optional[Literal["generation", "manipulation", "temporal_anomaly"]] = None
    end_timestamp_seconds: Optional[float] = None


class DetectorResult(BaseModel):
    name: str
    status: str
    label: Optional[str] = None
    score: Optional[float] = None
    synthetic_probability: Optional[float] = None
    manipulation_probability: Optional[float] = None
    task: Optional[Literal["generation", "manipulation", "temporal", "provenance", "supporting"]] = None
    model_version: Optional[str] = None
    calibration_id: Optional[str] = None
    suspicious_regions: List[Dict[str, Any]] = Field(default_factory=list)
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
    plain_language_summary: Optional[str] = None
    reasons_it_might_be_ai: List[str] = Field(default_factory=list)
    reasons_it_might_be_generated: List[str] = Field(default_factory=list)
    reasons_it_might_be_manipulated: List[str] = Field(default_factory=list)
    reasons_it_might_not_be_ai: List[str] = Field(default_factory=list)
    uncertainty_note: Optional[str] = None
    evidence_notes: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


class EvidenceSignal(BaseModel):
    source: str
    status: str
    signal: Literal[
        "authentic",
        "ai_generated",
        "ai_manipulated",
        "ai_generated_or_manipulated",
        "neutral",
        "unavailable",
    ]
    confidence: float = Field(..., ge=0, le=1)
    reliability: float = Field(..., ge=0, le=1)
    raw_score: Optional[float] = None
    evidence: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class MediaAssessment(BaseModel):
    verdict: Literal["likely_authentic", "likely_ai_generated", "likely_ai_manipulated", "inconclusive"]
    label: str
    confidence: Literal["low", "moderate", "high"]
    generation_score: Optional[float] = Field(None, ge=0, le=1)
    manipulation_score: Optional[float] = Field(None, ge=0, le=1)
    decision_policy_version: str
    model_versions: List[str] = Field(default_factory=list)
    # Deprecated compatibility alias for generation_score.
    detector_score: Optional[float] = Field(None, ge=0, le=1)
    reason: str
    evidence_supporting_authenticity: List[str] = Field(default_factory=list)
    evidence_supporting_generation: List[str] = Field(default_factory=list)
    evidence_supporting_manipulation: List[str] = Field(default_factory=list)
    evidence_conflicting: List[str] = Field(default_factory=list)
    # Deprecated compatibility union of generation/manipulation concern evidence.
    evidence_raising_concern: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    signals: List[EvidenceSignal] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    content_type: ContentType
    truth_score: int = Field(..., ge=0, le=100, deprecated=True, description="Legacy context score; never use as a media-authenticity conclusion.")
    risk_level: str = Field(..., deprecated=True, description="Legacy compatibility field. Use assessment.verdict.")
    verdict: str = Field(..., deprecated=True, description="Legacy compatibility label. Use assessment.verdict.")
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
    assessment: Optional[MediaAssessment] = None


class VideoAnalysisResponse(AnalysisResponse):
    frames_analyzed: int
    suspicious_frames: List[SuspiciousFrame]

export type MediaType = "image" | "video";

export type RiskLevel = "High Trust" | "Medium Trust" | "Low Trust" | "High Risk";

export type Evidence = Record<string, number>;

export interface DetectorResult {
  name: string;
  status: string;
  label?: string | null;
  score?: number | null;
  synthetic_probability?: number | null;
  details?: Record<string, unknown>;
}

export interface ProvenanceResult {
  status: string;
  score: number;
  summary: string;
  details?: Record<string, unknown>;
}

export interface Citation {
  title: string;
  url: string;
  source?: string | null;
  snippet?: string | null;
}

export interface WebResearchResult {
  status: string;
  provider: string;
  score: number;
  queries: string[];
  matches_found: number;
  summary: string;
  citations: Citation[];
  details?: Record<string, unknown>;
}

export interface CustomFeedback {
  headline: string;
  explanation: string;
  evidence_notes: string[];
  next_steps: string[];
}

export type EvidenceSignalValue = "authentic" | "ai_generated_or_manipulated" | "neutral" | "unavailable";

export interface EvidenceSignal {
  source: string;
  status: string;
  signal: EvidenceSignalValue;
  confidence: number;
  reliability: number;
  raw_score?: number | null;
  evidence: string[];
  limitations: string[];
}

export interface ImageAssessment {
  verdict: "likely_authentic" | "inconclusive" | "likely_ai_generated_or_manipulated";
  label: string;
  confidence: "low" | "moderate" | "high";
  detector_score?: number | null;
  reason: string;
  evidence_supporting_authenticity: string[];
  evidence_raising_concern: string[];
  limitations: string[];
  signals: EvidenceSignal[];
}

export interface SuspiciousFrame {
  frame_index: number;
  source_frame_number?: number | null;
  timestamp_seconds?: number | null;
  truth_score: number;
  synthetic_probability?: number | null;
  tile_synthetic_probability?: number | null;
  warnings: string[];
}

export interface AnalysisResult {
  content_type: MediaType;
  truth_score: number;
  risk_level: RiskLevel | string;
  verdict: string;
  summary: string;
  warnings: string[];
  positive_signals: string[];
  recommendations: string[];
  evidence: Evidence;
  technical_details: Record<string, unknown>;
  disclaimer: string;
  analysis_mode?: string;
  confidence?: number;
  detectors?: DetectorResult[];
  provenance?: ProvenanceResult | null;
  web_research?: WebResearchResult | null;
  citations?: Citation[];
  custom_feedback?: CustomFeedback | null;
  assessment?: ImageAssessment | null;
  frames_analyzed?: number;
  suspicious_frames?: SuspiciousFrame[];
}

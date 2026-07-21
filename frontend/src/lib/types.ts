export type MediaType = "image" | "video";

export type RiskLevel = "High Trust" | "Medium Trust" | "Low Trust" | "High Risk";

export type Evidence = Record<string, number>;

export interface DetectorResult {
  name: string;
  status: string;
  label?: string | null;
  score?: number | null;
  synthetic_probability?: number | null;
  manipulation_probability?: number | null;
  task?: "generation" | "manipulation" | "temporal" | "provenance" | "supporting" | null;
  model_version?: string | null;
  calibration_id?: string | null;
  suspicious_regions?: Array<Record<string, unknown>>;
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
  plain_language_summary?: string | null;
  reasons_it_might_be_ai?: string[];
  reasons_it_might_be_generated?: string[];
  reasons_it_might_be_manipulated?: string[];
  reasons_it_might_not_be_ai?: string[];
  uncertainty_note?: string | null;
  evidence_notes: string[];
  next_steps: string[];
}

export type EvidenceSignalValue =
  | "authentic"
  | "ai_generated"
  | "ai_manipulated"
  | "ai_generated_or_manipulated"
  | "neutral"
  | "unavailable";

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

export interface MediaAssessment {
  verdict: "likely_authentic" | "likely_ai_generated" | "likely_ai_manipulated" | "inconclusive";
  label: string;
  confidence: "low" | "moderate" | "high";
  generation_score?: number | null;
  manipulation_score?: number | null;
  decision_policy_version: string;
  model_versions: string[];
  detector_score?: number | null;
  reason: string;
  evidence_supporting_authenticity: string[];
  evidence_supporting_generation: string[];
  evidence_supporting_manipulation: string[];
  evidence_conflicting: string[];
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
  manipulation_probability?: number | null;
  tile_synthetic_probability?: number | null;
  warnings: string[];
  kind?: "generation" | "manipulation" | "temporal_anomaly" | null;
  end_timestamp_seconds?: number | null;
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
  assessment?: MediaAssessment | null;
  frames_analyzed?: number;
  suspicious_frames?: SuspiciousFrame[];
}

export type ContentType = "image" | "video" | "text";

export type RiskLevel = "High Trust" | "Medium Trust" | "Low Trust" | "High Risk";

export type Evidence = Record<string, number>;

export interface SuspiciousFrame {
  frame_index: number;
  timestamp_seconds?: number | null;
  truth_score: number;
  warnings: string[];
}

export interface AnalysisResult {
  content_type: ContentType;
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
  frames_analyzed?: number;
  suspicious_frames?: SuspiciousFrame[];
}

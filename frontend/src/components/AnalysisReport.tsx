import { forwardRef, type ReactNode } from "react";
import type { AnalysisResult, Citation, DetectorResult } from "../lib/types";

interface AnalysisReportProps {
  result: AnalysisResult;
  completedAt: Date;
  onNewAnalysis: () => void;
}

interface ForensicAnalysis {
  score?: number;
  synthetic_artifact_probability?: number;
  manipulation_probability?: number;
  caption_overlay?: {
    is_likely?: boolean;
    confidence?: number;
    location?: string | null;
    explanation?: string;
  };
  noise_residual?: Record<string, unknown>;
  frequency_spectrum?: Record<string, unknown>;
  jpeg_blockiness?: Record<string, unknown>;
  error_level_analysis?: Record<string, unknown>;
  duplicate_patch_analysis?: Record<string, unknown>;
}

interface VideoCoverage {
  mode?: string;
  exhaustive?: boolean;
  frame_stride?: number;
  frames_analyzed?: number;
  coverage_percent?: number;
  native_pixels_examined?: number;
  tile_count?: number;
  model_input_note?: string;
}

interface AttachmentFingerprint {
  sha256?: string;
  perceptual_hashes?: Record<string, string>;
}

interface SourceMatch {
  status?: string;
  confidence?: number;
  matched_citations?: number;
  explanation?: string;
}

function objectRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function formatLabel(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

const EVIDENCE_LABELS: Record<string, string> = {
  ai_generation_score: "Dedicated detector AI-class score (not probability)",
  sampled_frame_ai_generation_score: "Sampled-frame AI likelihood",
  video_ai_generation_score: "Video AI-generated likelihood",
  video_manipulation_score: "Video manipulation-class score",
  metadata_score: "Metadata availability (not authenticity)",
  visual_consistency_score: "Basic image quality (not authenticity)",
  compression_score: "Compression consistency (not authenticity)",
  pixel_forensic_score: "Traditional forensic consistency",
  ai_artifact_score: "Handcrafted AI-artifact signal",
  source_score: "Source context",
  provenance_score: "Verifiable provenance",
  web_corroboration_score: "Web corroboration",
  overall_risk_score: "Legacy context risk (not verdict)"
};

function evidenceLabel(value: string): string {
  return EVIDENCE_LABELS[value] ?? formatLabel(value);
}

function readableStatus(value: string): string {
  return value.replace(/_/g, " ");
}

function percent(value?: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "Not available";
  return `${Math.round(value * 100)}%`;
}

function score(value?: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "Not available";
  return `${Math.round(value)}/100`;
}

function readableCount(value?: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "Not available";
  return Math.round(value).toLocaleString();
}

function shortHash(value?: string): string | null {
  if (!value) return null;
  if (value.length <= 22) return value;
  return `${value.slice(0, 12)}…${value.slice(-8)}`;
}

function detailInterpretation(record?: Record<string, unknown>): string | null {
  return typeof record?.interpretation === "string" ? record.interpretation : null;
}

function forensicFrom(result: AnalysisResult): ForensicAnalysis | null {
  const value = objectRecord(result.technical_details?.forensic_analysis);
  return value ? (value as ForensicAnalysis) : null;
}

function videoCoverageFrom(result: AnalysisResult): VideoCoverage | null {
  const value = objectRecord(result.technical_details?.analysis_coverage);
  return value ? (value as VideoCoverage) : null;
}

function fingerprintFrom(result: AnalysisResult): AttachmentFingerprint | null {
  const webDetails = objectRecord(result.web_research?.details);
  const value = objectRecord(result.technical_details?.attachment_fingerprint) ?? objectRecord(webDetails?.attachment_fingerprint);
  return value ? (value as AttachmentFingerprint) : null;
}

function sourceMatchFrom(result: AnalysisResult): SourceMatch | null {
  const details = objectRecord(result.web_research?.details);
  const value = objectRecord(details?.source_match);
  return value ? (value as SourceMatch) : null;
}

function detectorSummary(detector: DetectorResult): string {
  if (typeof detector.manipulation_probability === "number") {
    return `${percent(detector.manipulation_probability)} manipulation-class score`;
  }
  if (typeof detector.synthetic_probability === "number") {
    return `${percent(detector.synthetic_probability)} AI-class score`;
  }
  if (typeof detector.score === "number") return score(detector.score);
  return readableStatus(detector.status);
}

function detectorDisplayName(detector: DetectorResult): string {
  const normalized = detector.name.toLowerCase();
  if (normalized.includes("truthshield-image-detector")) return "TruthShield learned image detector";
  if (normalized === "local_heuristic_synthetic_likelihood") return "Local heuristic fallback";
  return detector.name;
}

function learnedDetectorAvailable(result: AnalysisResult): boolean {
  const summary = objectRecord(result.technical_details?.ai_detector_summary);
  if (typeof summary?.learned_model_available === "boolean") return summary.learned_model_available;
  return Boolean(result.detectors?.some((detector) => {
    const provider = detector.details?.model_provider;
    const name = detector.name.toLowerCase();
    return detector.status === "completed" && (
      provider === "huggingface_local"
      || name.includes("truthshield")
      || name.includes("trained_")
    );
  }));
}

function aiDetectorScore(result: AnalysisResult): number | null {
  const evidenceKeys = result.content_type === "video"
    ? ["video_ai_generation_score", "sampled_frame_ai_generation_score", "ai_generation_score"]
    : ["ai_generation_score"];
  for (const key of evidenceKeys) {
    const value = result.evidence?.[key];
    if (typeof value === "number" && Number.isFinite(value)) return Math.max(0, Math.min(100, value));
  }
  const learned = result.detectors?.find((detector) =>
    detector.status === "completed"
      && detector.name !== "local_heuristic_synthetic_likelihood"
      && typeof detector.synthetic_probability === "number"
  );
  return typeof learned?.synthetic_probability === "number"
    ? Math.max(0, Math.min(100, learned.synthetic_probability * 100))
    : null;
}

function generationVerdict(likelihood: number | null, learnedAvailable: boolean): { headline: string; detail: string } {
  if (!learnedAvailable) {
    return {
      headline: "Trained detector unavailable",
      detail: "Fallback estimate only — do not treat this as a reliable real-versus-AI verdict"
    };
  }
  if (likelihood === null) return { headline: "No AI verdict", detail: "The detector returned no valid score" };
  if (likelihood >= 90) return { headline: "Likely AI-generated", detail: "Strong video detector signal" };
  if (likelihood >= 70) return { headline: "Inconclusive / uncertain", detail: "Detector evidence is not strong enough for an accusation" };
  if (likelihood > 30) return { headline: "Mixed AI signals", detail: "The learned model is uncertain" };
  if (likelihood <= 15) return { headline: "Likely camera-made", detail: "Low learned-model AI signal" };
  return { headline: "Lower AI signal", detail: "The learned model leans away from AI generation" };
}

function generationRiskClass(result: AnalysisResult, likelihood: number | null, learnedAvailable: boolean): string {
  if (result.assessment?.verdict === "likely_authentic") return "risk-trust";
  if (result.assessment?.verdict === "likely_ai_generated" || result.assessment?.verdict === "likely_ai_manipulated") return "risk-high";
  if (result.assessment?.verdict === "inconclusive") return "risk-medium";
  if (!learnedAvailable || likelihood === null) return "risk-low";
  if (likelihood >= 70) return "risk-high";
  if (likelihood > 30) return "risk-medium";
  return "risk-trust";
}

function plainConfidence(value?: string): string {
  if (value === "high") return "High";
  if (value === "moderate") return "Medium";
  return "Low";
}

function feedbackReasons(result: AnalysisResult, side: "generated" | "manipulated" | "authentic"): string[] {
  const customReasons = side === "generated"
    ? result.custom_feedback?.reasons_it_might_be_generated ?? result.custom_feedback?.reasons_it_might_be_ai
    : side === "manipulated"
      ? result.custom_feedback?.reasons_it_might_be_manipulated
      : result.custom_feedback?.reasons_it_might_not_be_ai;
  if (customReasons) return customReasons;
  if (result.assessment) {
    if (side === "generated") return result.assessment.evidence_supporting_generation;
    if (side === "manipulated") return result.assessment.evidence_supporting_manipulation;
    return result.assessment.evidence_supporting_authenticity;
  }
  return side === "authentic" ? result.positive_signals : side === "generated" ? result.warnings : [];
}

function ReportSection({ index, title, tone, children }: { index: string; title: string; tone?: string; children: ReactNode }) {
  return (
    <section className={`report-section${tone ? ` ${tone}` : ""}`}>
      <div className="report-section-heading">
        <span className="section-number" aria-hidden="true">{index}</span>
        <h2>{title}</h2>
      </div>
      <div className="report-section-content">{children}</div>
    </section>
  );
}

function FindingList({ items, emptyMessage }: { items: string[]; emptyMessage: string }) {
  if (items.length === 0) return <p className="empty-finding">{emptyMessage}</p>;
  return (
    <ul className="finding-list">
      {items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
    </ul>
  );
}

function ExternalArrow() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M5 11 11 5m-5 0h5v5" />
    </svg>
  );
}

function CitationList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return <p className="technical-empty">No citations were returned.</p>;
  return (
    <ol className="citation-list">
      {citations.slice(0, 5).map((citation, index) => (
        <li key={`${citation.url}-${index}`}>
          <a href={citation.url} target="_blank" rel="noreferrer">
            <span>{citation.title}</span>
            <ExternalArrow />
          </a>
          {citation.source ? <span className="citation-source">{citation.source}</span> : null}
          {citation.snippet ? <p>{citation.snippet}</p> : null}
        </li>
      ))}
    </ol>
  );
}

function TechnicalEvidence({ result }: { result: AnalysisResult }) {
  const forensic = forensicFrom(result);
  const coverage = videoCoverageFrom(result);
  const fingerprint = fingerprintFrom(result);
  const sourceMatch = sourceMatchFrom(result);
  const citations = result.citations?.length ? result.citations : (result.web_research?.citations ?? []);
  const primitiveDetails = Object.entries(result.technical_details ?? {}).filter(([, value]) =>
    ["string", "number", "boolean"].includes(typeof value)
  );
  const evidenceEntries = Object.entries(result.evidence ?? {}).filter(
    ([key]) => !result.assessment || key !== "overall_risk_score"
  );
  const forensicNotes = forensic
    ? [
        detailInterpretation(forensic.noise_residual),
        detailInterpretation(forensic.frequency_spectrum),
        detailInterpretation(forensic.jpeg_blockiness),
        detailInterpretation(forensic.error_level_analysis),
        detailInterpretation(forensic.duplicate_patch_analysis)
      ].filter((note): note is string => Boolean(note))
    : [];

  return (
    <div className="technical-content">
      <section className="technical-group">
        <h3>Analysis overview</h3>
        <dl className="metric-list">
          <div><dt>Analysis mode</dt><dd>{readableStatus(result.analysis_mode ?? "local heuristic")}</dd></div>
          <div><dt>Evidence coverage</dt><dd>{percent(result.confidence)}</dd></div>
          {primitiveDetails.map(([key, value]) => (
            <div key={key}><dt>{formatLabel(key)}</dt><dd>{String(value)}</dd></div>
          ))}
        </dl>
      </section>

      {evidenceEntries.length ? (
        <section className="technical-group">
          <h3>Raw evidence metrics</h3>
          <p>These are internal model and file-check scores. They are not real-world probabilities.</p>
          <dl className="metric-list">
            {evidenceEntries.map(([key, rawValue]) => (
              <div key={key}><dt>{evidenceLabel(key)}</dt><dd>{Math.round(Number(rawValue) || 0)}</dd></div>
            ))}
          </dl>
        </section>
      ) : null}

      {result.assessment?.signals?.length ? (
        <section className="technical-group">
          <h3>Decision signals</h3>
          <div className="detector-list">
            {result.assessment.signals.map((signal) => (
              <div className="detector-row" key={signal.source}>
                <div><strong>{formatLabel(signal.source)}</strong><span>{readableStatus(signal.signal)}</span></div>
                <span>{readableStatus(signal.status)}</span>
                <span>{typeof signal.raw_score === "number" ? `raw ${signal.raw_score.toFixed(3)}` : `reliability ${percent(signal.reliability)}`}</span>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {forensic ? (
        <section className="technical-group">
          <h3>Pixel forensics</h3>
          <dl className="metric-list metric-list-compact">
            <div><dt>Forensic score</dt><dd>{score(forensic.score)}</dd></div>
            <div><dt>Handcrafted artifact score</dt><dd>{percent(forensic.synthetic_artifact_probability)}</dd></div>
            <div><dt>Manipulation probability</dt><dd>{percent(forensic.manipulation_probability)}</dd></div>
            {forensic.caption_overlay?.is_likely ? (
              <div>
                <dt>Graphic overlay</dt>
                <dd>
                  Likely{typeof forensic.caption_overlay.confidence === "number" ? ` · ${percent(forensic.caption_overlay.confidence)}` : ""}
                </dd>
              </div>
            ) : null}
          </dl>
          {forensicNotes.length ? <FindingList items={forensicNotes} emptyMessage="" /> : null}
        </section>
      ) : null}

      {coverage || typeof result.frames_analyzed === "number" || result.suspicious_frames?.length ? (
        <section className="technical-group">
          <h3>Video and frames</h3>
          <dl className="metric-list metric-list-compact">
            <div><dt>Frames analyzed</dt><dd>{readableCount(coverage?.frames_analyzed ?? result.frames_analyzed)}</dd></div>
            {typeof coverage?.coverage_percent === "number" ? <div><dt>Coverage</dt><dd>{coverage.coverage_percent.toFixed(1)}%</dd></div> : null}
            {typeof coverage?.frame_stride === "number" ? <div><dt>Frame stride</dt><dd>{coverage.frame_stride}</dd></div> : null}
            {typeof coverage?.native_pixels_examined === "number" ? <div><dt>Native pixels</dt><dd>{readableCount(coverage.native_pixels_examined)}</dd></div> : null}
            {typeof coverage?.tile_count === "number" ? <div><dt>Model tiles</dt><dd>{readableCount(coverage.tile_count)}</dd></div> : null}
          </dl>
          {coverage?.model_input_note ? <p>{coverage.model_input_note}</p> : null}
          {result.suspicious_frames?.length ? (
            <div className="frame-list" aria-label="Suspicious frame samples">
              {result.suspicious_frames.map((frame) => (
                <div className="frame-row" key={`${frame.frame_index}-${frame.timestamp_seconds ?? "unknown"}`}>
                  <div><span>Frame</span><strong>{frame.frame_index}</strong></div>
                  <div><span>Time</span><strong>{typeof frame.timestamp_seconds === "number" ? `${frame.timestamp_seconds.toFixed(2)}s` : "—"}</strong></div>
                  <div><span>Truth Score</span><strong>{frame.truth_score}/100</strong></div>
                  <div><span>Synthetic signal</span><strong>{percent(frame.synthetic_probability)}</strong></div>
                  <div><span>Manipulation signal</span><strong>{percent(frame.manipulation_probability)}</strong></div>
                  <p>{frame.warnings.join(" · ")}</p>
                </div>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}

      <section className="technical-group">
        <h3>Detector outputs</h3>
        {result.detectors?.length ? (
          <div className="detector-list">
            {result.detectors.map((detector, index) => (
              <div className="detector-row" key={`${detector.name}-${index}`}>
                <div><strong>{detectorDisplayName(detector)}</strong>{detector.label ? <span>{readableStatus(detector.label)}</span> : null}</div>
                <span>{readableStatus(detector.status)}</span>
                <span>{detectorSummary(detector)}</span>
              </div>
            ))}
          </div>
        ) : <p className="technical-empty">No detector outputs were returned.</p>}
      </section>

      <section className="technical-group">
        <h3>Provenance</h3>
        {result.provenance ? (
          <div className="provenance-row">
            <div><strong>{readableStatus(result.provenance.status)}</strong><span>{score(result.provenance.score)}</span></div>
            <p>{result.provenance.summary}</p>
          </div>
        ) : <p className="technical-empty">No provenance result was returned.</p>}
        {fingerprint ? (
          <dl className="metric-list hash-list">
            {shortHash(fingerprint.sha256) ? <div><dt>SHA-256</dt><dd>{shortHash(fingerprint.sha256)}</dd></div> : null}
            {Object.entries(fingerprint.perceptual_hashes ?? {}).map(([key, value]) => (
              <div key={key}><dt>{formatLabel(key)}</dt><dd>{shortHash(value)}</dd></div>
            ))}
          </dl>
        ) : null}
      </section>

      <section className="technical-group">
        <h3>Web research</h3>
        {result.web_research ? (
          <>
            <div className="research-summary">
              <div><strong>{readableStatus(result.web_research.status)}</strong><span>{score(result.web_research.score)}</span></div>
              <p>{result.web_research.summary}</p>
              {result.web_research.queries.length ? <p className="query-line">Queries: {result.web_research.queries.join(" · ")}</p> : null}
            </div>
            {sourceMatch ? (
              <div className="source-match">
                <strong>Attachment match: {readableStatus(sourceMatch.status ?? "not checked")}</strong>
                {typeof sourceMatch.confidence === "number" ? <span>{percent(sourceMatch.confidence)} confidence</span> : null}
                {sourceMatch.explanation ? <p>{sourceMatch.explanation}</p> : null}
              </div>
            ) : null}
          </>
        ) : <p className="technical-empty">No web research was returned.</p>}
        <CitationList citations={citations} />
      </section>
    </div>
  );
}

const AnalysisReport = forwardRef<HTMLElement, AnalysisReportProps>(function AnalysisReport(
  { result, completedAt, onNewAnalysis },
  ref
) {
  const generationLikelihood = aiDetectorScore(result);
  const learnedAvailable = learnedDetectorAvailable(result);
  const generationResult = generationVerdict(generationLikelihood, learnedAvailable);
  const assessment = result.assessment ?? null;
  const headline = result.custom_feedback?.headline ?? assessment?.label ?? generationResult.headline;
  const plainSummary = result.custom_feedback?.plain_language_summary
    ?? result.custom_feedback?.explanation
    ?? assessment?.reason
    ?? result.summary;
  const generationReasons = feedbackReasons(result, "generated");
  const manipulationReasons = feedbackReasons(result, "manipulated");
  const authenticityReasons = feedbackReasons(result, "authentic");
  const uncertaintyNote = result.custom_feedback?.uncertainty_note
    ?? "This result is an estimate, not proof. Editing, compression, screenshots, and unfamiliar AI tools can change the clues the system uses.";
  const nextSteps = result.custom_feedback?.next_steps?.length
    ? result.custom_feedback.next_steps
    : result.recommendations;
  const showModelScore = learnedAvailable && generationLikelihood !== null;
  const verdictDetail = assessment
    ? `${plainConfidence(assessment.confidence)} result strength · based on the available checks`
    : showModelScore
      ? "Model estimate only · not a percent chance or proof"
      : generationResult.detail;
  const technicalPreview = [
    typeof assessment?.generation_score === "number"
      ? `${Math.round(assessment.generation_score * 100)}% generation-class score`
      : null,
    typeof assessment?.manipulation_score === "number"
      ? `${Math.round(assessment.manipulation_score * 100)}% manipulation-class score`
      : null,
    generationLikelihood !== null
      && typeof assessment?.generation_score !== "number"
      ? `${Math.round(generationLikelihood)}% ${showModelScore ? "raw AI-class score" : "fallback signal"}`
      : null,
    learnedAvailable ? "learned detector active" : "fallback only",
    typeof result.frames_analyzed === "number" ? `${result.frames_analyzed.toLocaleString()} frames` : null,
    result.suspicious_frames?.length ? `${result.suspicious_frames.length} suspicious samples` : null,
    result.detectors?.length ? `${result.detectors.length} detector outputs` : null
  ].filter((item): item is string => Boolean(item));
  const analyzedAt = new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(completedAt);

  return (
    <section
      ref={ref}
      id="report"
      className={`analysis-report ${generationRiskClass(result, generationLikelihood, learnedAvailable)}`}
      aria-labelledby="report-heading"
      tabIndex={-1}
    >
      <div className="report-meta">
        <span>{result.content_type === "image" ? "Image" : "Video"} analysis</span>
        <time dateTime={completedAt.toISOString()}>Analyzed {analyzedAt}</time>
      </div>

      <div className="score-summary">
        <div className="score-block">
          <span>{assessment ? "Result strength" : (showModelScore ? "AI model signal" : "Model status")}</span>
          <div className={`score-number${assessment ? " decision-word" : ""}`}>
            <strong>{assessment ? plainConfidence(assessment.confidence) : (showModelScore ? Math.round(generationLikelihood ?? 0) : "—")}</strong>
            {assessment || !showModelScore ? null : <span>%</span>}
          </div>
          <small>{assessment ? "How strongly the checks support this result" : (showModelScore ? "Raw model score — not probability" : "Fallback checks only")}</small>
        </div>
        <div className="verdict-block">
          <h1 id="report-heading">{headline}</h1>
          <p className="verdict">{verdictDetail}</p>
          {assessment ? (
            <p className="report-summary">
              Generation score: {percent(assessment.generation_score)} · Manipulation score: {percent(assessment.manipulation_score)}
              {` · Policy ${assessment.decision_policy_version}`}
            </p>
          ) : null}
          <p className="report-summary">Review both sides below before making an important decision.</p>
        </div>
      </div>

      <ReportSection index="01" title="What this result means">
        <p className="plain-language-summary">{plainSummary}</p>
        <p className="plain-language-note">
          AI detectors compare patterns. They do not know for certain who or what made the file, and a model score is not the percent chance that the result is correct.
        </p>
      </ReportSection>

      <ReportSection index="02" title="Evidence of AI generation" tone="warning-section">
        <FindingList
          items={generationReasons}
          emptyMessage="No strong calibrated generation evidence was available. That alone does not establish authenticity."
        />
      </ReportSection>

      <ReportSection index="03" title="Evidence of AI editing or manipulation" tone="warning-section">
        <FindingList
          items={manipulationReasons}
          emptyMessage="No dedicated manipulation evidence was found, or the specialist was unavailable."
        />
      </ReportSection>

      <ReportSection index="04" title="Evidence supporting authenticity" tone="positive-section">
        <FindingList
          items={authenticityReasons}
          emptyMessage="No strong, reliable authenticity support was available. This does not mean the content is AI-generated."
        />
      </ReportSection>

      <ReportSection index="05" title="What could make this result wrong">
        <p className="uncertainty-note">{uncertaintyNote}</p>
      </ReportSection>

      <ReportSection index="06" title="What to do next">
        {nextSteps.length > 0 ? (
          <div className="recommendation-copy">
            <ol>{nextSteps.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ol>
          </div>
        ) : <p className="empty-finding">Verify important claims with trusted, independent sources.</p>}
      </ReportSection>

      <p className="report-disclaimer">{result.disclaimer}</p>

      <details className="technical-evidence">
        <summary>
          <span className="technical-summary-heading"><span className="section-number" aria-hidden="true">07</span><span>Technical evidence</span></span>
          <span className="technical-preview">{technicalPreview.length > 0 ? technicalPreview.join(" · ") : "Forensics · provenance · research"}</span>
          <svg viewBox="0 0 20 20" aria-hidden="true" focusable="false"><path d="m5 8 5 5 5-5" /></svg>
        </summary>
        <TechnicalEvidence result={result} />
      </details>

      <button type="button" className="new-analysis" onClick={onNewAnalysis}>
        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M19 12H5m0 0 5-5m-5 5 5 5" /></svg>
        <span>Back to new analysis</span>
      </button>
    </section>
  );
});

export default AnalysisReport;

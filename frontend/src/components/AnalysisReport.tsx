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

function riskClass(scoreValue: number): string {
  if (scoreValue >= 80) return "risk-trust";
  if (scoreValue >= 60) return "risk-medium";
  if (scoreValue >= 40) return "risk-low";
  return "risk-high";
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
  if (typeof detector.synthetic_probability === "number") {
    return `${percent(detector.synthetic_probability)} synthetic likelihood`;
  }
  if (typeof detector.score === "number") return score(detector.score);
  return readableStatus(detector.status);
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
      {citations.slice(0, 5).map((citation) => (
        <li key={citation.url}>
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
          <div><dt>Mode</dt><dd>{readableStatus(result.analysis_mode ?? "local heuristic")}</dd></div>
          <div><dt>Confidence</dt><dd>{percent(result.confidence)}</dd></div>
          {primitiveDetails.map(([key, value]) => (
            <div key={key}><dt>{formatLabel(key)}</dt><dd>{String(value)}</dd></div>
          ))}
        </dl>
      </section>

      {result.custom_feedback ? (
        <section className="technical-group">
          <h3>Analysis notes</h3>
          <p className="technical-lead">{result.custom_feedback.headline}</p>
          <p>{result.custom_feedback.explanation}</p>
          {result.custom_feedback.next_steps.length ? (
            <FindingList items={result.custom_feedback.next_steps} emptyMessage="" />
          ) : null}
        </section>
      ) : null}

      {forensic ? (
        <section className="technical-group">
          <h3>Pixel forensics</h3>
          <dl className="metric-list metric-list-compact">
            <div><dt>Forensic score</dt><dd>{score(forensic.score)}</dd></div>
            <div><dt>Synthetic artifacts</dt><dd>{percent(forensic.synthetic_artifact_probability)}</dd></div>
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
                  <p>{frame.warnings.join(" · ")}</p>
                </div>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}

      <section className="technical-group">
        <h3>Detector opinions</h3>
        {result.detectors?.length ? (
          <div className="detector-list">
            {result.detectors.map((detector, index) => (
              <div className="detector-row" key={`${detector.name}-${index}`}>
                <div><strong>{detector.name}</strong>{detector.label ? <span>{readableStatus(detector.label)}</span> : null}</div>
                <span>{readableStatus(detector.status)}</span>
                <span>{detectorSummary(detector)}</span>
              </div>
            ))}
          </div>
        ) : <p className="technical-empty">No detector opinions were returned.</p>}
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
  const evidenceEntries = Object.entries(result.evidence ?? {});
  const technicalPreview = [
    typeof result.frames_analyzed === "number" ? `${result.frames_analyzed.toLocaleString()} frames` : null,
    result.suspicious_frames?.length ? `${result.suspicious_frames.length} suspicious samples` : null,
    result.detectors?.length ? `${result.detectors.length} detector opinions` : null
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
      className={`analysis-report ${riskClass(result.truth_score)}`}
      aria-labelledby="report-heading"
      tabIndex={-1}
    >
      <div className="report-meta">
        <span>{result.content_type === "image" ? "Image" : "Video"} analysis</span>
        <time dateTime={completedAt.toISOString()}>Analyzed {analyzedAt}</time>
      </div>

      <div className="score-summary">
        <div className="score-block">
          <span>Truth Score</span>
          <strong>{Math.round(result.truth_score)}</strong>
        </div>
        <div className="verdict-block">
          <h1 id="report-heading">{result.risk_level}</h1>
          <p className="verdict">{result.verdict}</p>
          <p className="report-summary">{result.summary}</p>
        </div>
      </div>

      <ReportSection index="01" title="Key evidence">
        {evidenceEntries.length ? (
          <dl className="evidence-list">
            {evidenceEntries.map(([key, rawValue]) => (
              <div key={key}>
                <dt>{formatLabel(key)}</dt>
                <dd>{Math.round(Number(rawValue) || 0)}</dd>
              </div>
            ))}
          </dl>
        ) : <p className="empty-finding">No evidence metrics were returned.</p>}
        {result.custom_feedback?.evidence_notes.length ? (
          <FindingList items={result.custom_feedback.evidence_notes} emptyMessage="" />
        ) : null}
      </ReportSection>

      <ReportSection index="02" title="Warning signs" tone="warning-section">
        <FindingList items={result.warnings} emptyMessage="No strong warning signs were returned." />
      </ReportSection>

      <ReportSection index="03" title="Reassuring signals" tone="positive-section">
        <FindingList items={result.positive_signals} emptyMessage="No strong reassuring signals were returned." />
      </ReportSection>

      <ReportSection index="04" title="Safety recommendation">
        {result.recommendations.length ? (
          <div className="recommendation-copy">
            <p>{result.recommendations[0]}</p>
            {result.recommendations.length > 1 ? (
              <ol>{result.recommendations.slice(1).map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ol>
            ) : null}
          </div>
        ) : <p className="empty-finding">Verify important claims with trusted, independent sources.</p>}
      </ReportSection>

      <p className="report-disclaimer">{result.disclaimer}</p>

      <details className="technical-evidence">
        <summary>
          <span className="technical-summary-heading"><span className="section-number" aria-hidden="true">05</span><span>Technical evidence</span></span>
          <span className="technical-preview">{technicalPreview.length ? technicalPreview.join(" · ") : "Forensics · provenance · research"}</span>
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

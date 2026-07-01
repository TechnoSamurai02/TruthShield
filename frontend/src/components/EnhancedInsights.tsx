import { BadgeCheck, Bot, ExternalLink, FileSearch, Fingerprint, ScanSearch, Sparkles } from "lucide-react";
import type { AnalysisResult, Citation, DetectorResult } from "../lib/types";

interface EnhancedInsightsProps {
  result: AnalysisResult;
}

function percent(value?: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return `${Math.round(value * 100)}%`;
}

function score(value?: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return `${Math.round(value)}/100`;
}

function statusTone(status: string): string {
  const lowered = status.toLowerCase();
  if (["verified", "completed"].includes(lowered)) return "border-emerald-300/25 bg-emerald-300/10 text-emerald-50";
  if (["no_results", "no_manifest", "not_configured", "tool_unavailable", "skipped"].includes(lowered)) {
    return "border-yellow-300/25 bg-yellow-300/10 text-yellow-50";
  }
  return "border-orange-300/25 bg-orange-300/10 text-orange-50";
}

function readableStatus(status: string): string {
  return status.replace(/_/g, " ");
}

interface SourceMatch {
  status?: string;
  confidence?: number;
  matched_citations?: number;
  explanation?: string;
}

interface AttachmentFingerprint {
  sha256?: string;
  perceptual_hashes?: Record<string, string>;
}

function objectRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function sourceMatchFrom(result: AnalysisResult): SourceMatch | null {
  const details = objectRecord(result.web_research?.details);
  const sourceMatch = objectRecord(details?.source_match);
  return sourceMatch ? (sourceMatch as SourceMatch) : null;
}

function fingerprintFrom(result: AnalysisResult): AttachmentFingerprint | null {
  const technical = objectRecord(result.technical_details);
  const webDetails = objectRecord(result.web_research?.details);
  const fingerprint = objectRecord(technical?.attachment_fingerprint) ?? objectRecord(webDetails?.attachment_fingerprint);
  return fingerprint ? (fingerprint as AttachmentFingerprint) : null;
}

function shortHash(value?: string): string | null {
  if (!value) return null;
  if (value.length <= 18) return value;
  return `${value.slice(0, 10)}...${value.slice(-8)}`;
}

function detectorSummary(detector: DetectorResult): string {
  if (typeof detector.synthetic_probability === "number") {
    return `${percent(detector.synthetic_probability)} synthetic likelihood`;
  }
  if (typeof detector.score === "number") {
    return score(detector.score);
  }
  return readableStatus(detector.status);
}

function CitationList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) {
    return <p className="text-sm leading-6 text-white/55">No citations returned for this scan.</p>;
  }
  return (
    <div className="grid gap-3">
      {citations.slice(0, 5).map((citation) => (
        <a
          key={citation.url}
          href={citation.url}
          target="_blank"
          rel="noreferrer"
          className="rounded-lg border border-white/10 bg-white/5 p-3 text-sm text-white/75 transition hover:border-cyan-300/30 hover:text-white"
        >
          <div className="flex items-start justify-between gap-3">
            <span className="font-bold text-white">{citation.title}</span>
            <ExternalLink className="mt-1 h-4 w-4 shrink-0 text-cyan-200" />
          </div>
          {citation.source && <div className="mt-1 text-xs uppercase text-cyan-100/70">{citation.source}</div>}
          {citation.snippet && <p className="mt-2 leading-6 text-white/60">{citation.snippet}</p>}
        </a>
      ))}
    </div>
  );
}

export default function EnhancedInsights({ result }: EnhancedInsightsProps) {
  const detectors = result.detectors ?? [];
  const citations = result.citations ?? result.web_research?.citations ?? [];
  const confidence = typeof result.confidence === "number" ? Math.round(result.confidence * 100) : null;
  const sourceMatch = sourceMatchFrom(result);
  const fingerprint = fingerprintFrom(result);
  const phash = fingerprint?.perceptual_hashes?.phash;

  return (
    <div className="mt-4 grid gap-4 lg:grid-cols-2">
      {result.custom_feedback && (
        <section className="panel lg:col-span-2">
          <div className="mb-4 flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-cyan-300" />
            <h3 className="text-base font-bold text-white">Custom Feedback</h3>
          </div>
          <h4 className="text-lg font-black text-white">{result.custom_feedback.headline}</h4>
          <p className="mt-3 text-sm leading-7 text-white/70">{result.custom_feedback.explanation}</p>
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <div className="rounded-lg border border-white/10 bg-white/5 p-3">
              <div className="mb-2 text-xs font-bold uppercase text-white/45">Evidence Notes</div>
              <ul className="space-y-2 text-sm leading-6 text-white/70">
                {result.custom_feedback.evidence_notes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </div>
            <div className="rounded-lg border border-white/10 bg-white/5 p-3">
              <div className="mb-2 text-xs font-bold uppercase text-white/45">Next Steps</div>
              <ul className="space-y-2 text-sm leading-6 text-white/70">
                {result.custom_feedback.next_steps.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ul>
            </div>
          </div>
        </section>
      )}

      <section className="panel">
        <div className="mb-4 flex items-center gap-2">
          <Bot className="h-5 w-5 text-violet-200" />
          <h3 className="text-base font-bold text-white">Detector Opinions</h3>
        </div>
        {detectors.length > 0 ? (
          <div className="grid gap-3">
            {detectors.map((detector) => (
              <div key={`${detector.name}-${detector.status}`} className={`rounded-lg border p-3 ${statusTone(detector.status)}`}>
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="break-words text-sm font-bold">{detector.name}</div>
                    {detector.label && <div className="mt-1 text-xs uppercase opacity-70">{detector.label.replace(/_/g, " ")}</div>}
                  </div>
                  <span className="shrink-0 rounded-full border border-white/10 bg-black/20 px-2 py-1 text-xs font-bold uppercase">
                    {readableStatus(detector.status)}
                  </span>
                </div>
                <div className="mt-3 font-mono text-sm">{detectorSummary(detector)}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="rounded-lg border border-white/10 bg-white/5 p-3 text-sm text-white/60">No detector results were returned.</p>
        )}
      </section>

      <section className="panel">
        <div className="mb-4 flex items-center gap-2">
          <Fingerprint className="h-5 w-5 text-emerald-200" />
          <h3 className="text-base font-bold text-white">Provenance</h3>
        </div>
        {result.provenance ? (
          <div className={`rounded-lg border p-3 ${statusTone(result.provenance.status)}`}>
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm font-bold capitalize">{readableStatus(result.provenance.status)}</span>
              <span className="font-mono text-sm">{score(result.provenance.score)}</span>
            </div>
            <p className="mt-3 text-sm leading-6 opacity-80">{result.provenance.summary}</p>
          </div>
        ) : (
          <p className="rounded-lg border border-white/10 bg-white/5 p-3 text-sm text-white/60">
            Provenance checks are not available for this content type.
          </p>
        )}
      </section>

      <section className="panel lg:col-span-2">
        <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-2">
            <FileSearch className="h-5 w-5 text-sky-200" />
            <h3 className="text-base font-bold text-white">Web Research</h3>
          </div>
          <div className="flex flex-wrap gap-2 text-xs font-bold uppercase">
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-white/55">
              {result.analysis_mode ?? "local heuristic"}
            </span>
            {confidence !== null && (
              <span className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-cyan-50">
                {confidence}% confidence
              </span>
            )}
          </div>
        </div>
        {result.web_research ? (
          <div className="grid gap-4">
            <div className={`rounded-lg border p-3 ${statusTone(result.web_research.status)}`}>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <span className="text-sm font-bold capitalize">{readableStatus(result.web_research.status)}</span>
                <span className="font-mono text-sm">{score(result.web_research.score)}</span>
              </div>
              <p className="mt-3 text-sm leading-6 opacity-80">{result.web_research.summary}</p>
              {result.web_research.queries.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {result.web_research.queries.map((query) => (
                    <span key={query} className="rounded-full border border-white/10 bg-black/20 px-3 py-1 text-xs text-white/70">
                      {query}
                    </span>
                  ))}
                </div>
              )}
            </div>
            {(sourceMatch || fingerprint) && (
              <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                <div className="mb-3 flex items-center gap-2">
                  <ScanSearch className="h-4 w-4 text-sky-200" />
                  <h4 className="text-sm font-bold text-white">Attachment Match</h4>
                </div>
                {sourceMatch && (
                  <div className={`rounded-lg border p-3 ${statusTone(sourceMatch.status ?? "not_checked")}`}>
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <span className="text-sm font-bold capitalize">{readableStatus(sourceMatch.status ?? "not checked")}</span>
                      {typeof sourceMatch.confidence === "number" && (
                        <span className="font-mono text-sm">{percent(sourceMatch.confidence)}</span>
                      )}
                    </div>
                    {sourceMatch.explanation && <p className="mt-3 text-sm leading-6 opacity-80">{sourceMatch.explanation}</p>}
                  </div>
                )}
                {fingerprint && (
                  <div className="mt-3 grid gap-2 text-xs text-white/55 sm:grid-cols-2">
                    {shortHash(fingerprint.sha256) && (
                      <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2">
                        <span className="font-bold uppercase text-white/35">SHA-256</span>
                        <div className="mt-1 font-mono text-white/70">{shortHash(fingerprint.sha256)}</div>
                      </div>
                    )}
                    {shortHash(phash) && (
                      <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2">
                        <span className="font-bold uppercase text-white/35">pHash</span>
                        <div className="mt-1 font-mono text-white/70">{shortHash(phash)}</div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            <CitationList citations={citations} />
            <div className="flex items-start gap-2 rounded-lg border border-cyan-300/15 bg-cyan-300/10 p-3 text-sm leading-6 text-cyan-50/75">
              <BadgeCheck className="mt-1 h-4 w-4 shrink-0 text-cyan-200" />
              <span>
                Free web research uses indexed web/image search and generated queries. It is not an unlimited exact reverse image search
                across the entire internet.
              </span>
            </div>
          </div>
        ) : (
          <p className="rounded-lg border border-white/10 bg-white/5 p-3 text-sm text-white/60">No web research was returned.</p>
        )}
      </section>
    </div>
  );
}

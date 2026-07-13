import { FileWarning, ScanLine } from "lucide-react";
import type { AnalysisResult } from "../lib/types";
import EnhancedInsights from "./EnhancedInsights";
import EvidenceBreakdown from "./EvidenceBreakdown";
import RecommendationList from "./RecommendationList";
import ScoreGauge from "./ScoreGauge";
import WarningList from "./WarningList";

interface ResultCardProps {
  result: AnalysisResult;
}

const badgeTone = (riskLevel: string) => {
  if (riskLevel === "High Trust") return "border-emerald-300/30 bg-emerald-300/10 text-emerald-100";
  if (riskLevel === "Medium Trust") return "border-yellow-300/30 bg-yellow-300/10 text-yellow-100";
  if (riskLevel === "Low Trust") return "border-orange-300/30 bg-orange-300/10 text-orange-100";
  return "border-red-300/30 bg-red-300/10 text-red-100";
};

export default function ResultCard({ result }: ResultCardProps) {
  return (
    <section id="report" className="rounded-lg border border-cyan-300/20 bg-[#070910]/90 p-4 shadow-neon md:p-6">
      <div className="mb-6 flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase text-cyan-200">
            <ScanLine className="h-4 w-4" />
            Combined Report
          </div>
          <h2 className="text-2xl font-black text-white md:text-3xl">{result.verdict}</h2>
          <p className="mt-3 max-w-3xl text-sm leading-7 text-white/70 md:text-base">{result.summary}</p>
          <div className="mt-4 flex flex-wrap gap-3">
            <span className={`rounded-full border px-3 py-1 text-xs font-bold uppercase ${badgeTone(result.risk_level)}`}>
              {result.risk_level}
            </span>
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold uppercase text-white/60">
              {result.content_type} scan
            </span>
            {typeof result.frames_analyzed === "number" && (
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold uppercase text-white/60">
                {result.frames_analyzed} frames
              </span>
            )}
          </div>
        </div>
        <ScoreGauge score={result.truth_score} riskLevel={result.risk_level} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <EvidenceBreakdown evidence={result.evidence} />
        <RecommendationList recommendations={result.recommendations} />
        <WarningList title="Warning Signs" items={result.warnings} variant="warning" />
        <WarningList title="Positive Signals" items={result.positive_signals} variant="positive" />
      </div>

      {result.suspicious_frames && result.suspicious_frames.length > 0 && (
        <section className="panel mt-4">
          <div className="mb-4 flex items-center gap-2">
            <FileWarning className="h-5 w-5 text-red-300" />
            <h3 className="text-base font-bold text-white">Suspicious Frame Samples</h3>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {result.suspicious_frames.map((frame) => (
              <div key={`${frame.frame_index}-${frame.timestamp_seconds}`} className="rounded-lg border border-red-300/20 bg-red-300/10 p-3">
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="font-semibold text-red-100">Frame {frame.frame_index}</span>
                  <span className="font-mono text-red-100">{frame.truth_score}/100</span>
                </div>
                {typeof frame.timestamp_seconds === "number" && (
                  <p className="mt-1 text-xs text-white/50">{frame.timestamp_seconds.toFixed(2)} seconds</p>
                )}
                {typeof frame.synthetic_probability === "number" && (
                  <p className="mt-1 text-xs text-red-100/70">
                    {Math.round(frame.synthetic_probability * 100)}% synthetic signal
                  </p>
                )}
                <ul className="mt-3 space-y-2 text-sm text-white/70">
                  {frame.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>
      )}

      <EnhancedInsights result={result} />

      <div className="mt-4 rounded-lg border border-white/10 bg-white/5 p-4 text-sm leading-6 text-white/70">
        <strong className="text-white">Disclaimer: </strong>
        {result.disclaimer}
      </div>
    </section>
  );
}

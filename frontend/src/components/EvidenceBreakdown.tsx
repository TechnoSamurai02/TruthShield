import { Activity } from "lucide-react";
import type { Evidence } from "../lib/types";

interface EvidenceBreakdownProps {
  evidence: Evidence;
}

function formatLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function metricTone(key: string, value: number): string {
  const isRisk =
    key.includes("risk") ||
    key.includes("manipulation") ||
    key.includes("artifact") ||
    key.includes("synthetic") ||
    key.includes("ai_generation");
  if (isRisk) {
    if (value >= 70) return "bg-red-400";
    if (value >= 40) return "bg-orange-400";
    return "bg-emerald-400";
  }
  if (value >= 70) return "bg-emerald-400";
  if (value >= 40) return "bg-yellow-300";
  return "bg-red-400";
}

export default function EvidenceBreakdown({ evidence }: EvidenceBreakdownProps) {
  const entries = Object.entries(evidence);

  return (
    <section className="panel">
      <div className="mb-4 flex items-center gap-2">
        <Activity className="h-5 w-5 text-cyan-300" />
        <h3 className="text-base font-bold text-white">Evidence Breakdown</h3>
      </div>
      <div className="space-y-4">
        {entries.map(([key, rawValue]) => {
          const value = Math.max(0, Math.min(100, Number(rawValue) || 0));
          return (
            <div key={key}>
              <div className="mb-2 flex items-center justify-between gap-3 text-sm">
                <span className="text-white/70">{formatLabel(key)}</span>
                <span className="font-mono text-white">{Math.round(value)}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-white/10">
                <div className={`h-full rounded-full ${metricTone(key, value)}`} style={{ width: `${value}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

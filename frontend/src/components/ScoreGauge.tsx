import { ShieldAlert, ShieldCheck } from "lucide-react";

interface ScoreGaugeProps {
  score: number;
  riskLevel: string;
}

const scoreTone = (score: number) => {
  if (score >= 80) return { color: "#34d399", label: "High trust" };
  if (score >= 60) return { color: "#facc15", label: "Medium trust" };
  if (score >= 40) return { color: "#fb923c", label: "Low trust" };
  return { color: "#f87171", label: "High risk" };
};

export default function ScoreGauge({ score, riskLevel }: ScoreGaugeProps) {
  const tone = scoreTone(score);
  const degrees = Math.max(0, Math.min(100, score)) * 3.6;

  return (
    <div className="flex flex-col items-center gap-4">
      <div
        className="relative grid h-48 w-48 place-items-center rounded-full border border-white/10 shadow-neon"
        style={{
          background: `conic-gradient(${tone.color} ${degrees}deg, rgba(255,255,255,0.08) ${degrees}deg 360deg)`
        }}
        aria-label={`Truth Score ${score} out of 100`}
      >
        <div className="absolute inset-4 rounded-full border border-white/10 bg-[#070910]" />
        <div className="relative text-center">
          <div className="mx-auto mb-2 grid h-9 w-9 place-items-center rounded-full border border-white/10 bg-white/5">
            {score >= 60 ? <ShieldCheck className="h-5 w-5 text-emerald-300" /> : <ShieldAlert className="h-5 w-5 text-red-300" />}
          </div>
          <div className="text-5xl font-black text-white">{score}</div>
          <div className="mt-1 text-xs font-semibold uppercase text-white/60">Truth Score</div>
        </div>
      </div>
      <div className="text-center">
        <div className="text-sm font-semibold text-white">{riskLevel}</div>
        <div className="text-xs uppercase text-white/50">{tone.label}</div>
      </div>
    </div>
  );
}

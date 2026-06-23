import { AlertTriangle, CheckCircle2 } from "lucide-react";

interface WarningListProps {
  title: string;
  items: string[];
  variant: "warning" | "positive";
}

export default function WarningList({ title, items, variant }: WarningListProps) {
  const isPositive = variant === "positive";
  const Icon = isPositive ? CheckCircle2 : AlertTriangle;
  const tone = isPositive
    ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-100"
    : "border-orange-400/25 bg-orange-400/10 text-orange-100";
  const iconTone = isPositive ? "text-emerald-300" : "text-orange-300";

  return (
    <section className="panel">
      <div className="mb-4 flex items-center gap-2">
        <Icon className={`h-5 w-5 ${iconTone}`} />
        <h3 className="text-base font-bold text-white">{title}</h3>
      </div>
      {items.length > 0 ? (
        <div className="grid gap-3">
          {items.map((item) => (
            <div key={item} className={`rounded-lg border p-3 text-sm leading-6 ${tone}`}>
              {item}
            </div>
          ))}
        </div>
      ) : (
        <p className="rounded-lg border border-white/10 bg-white/5 p-3 text-sm text-white/60">
          No strong signals in this category were detected by the current heuristic checks.
        </p>
      )}
    </section>
  );
}

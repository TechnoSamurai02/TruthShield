import { Compass } from "lucide-react";

interface RecommendationListProps {
  recommendations: string[];
}

export default function RecommendationList({ recommendations }: RecommendationListProps) {
  return (
    <section className="panel">
      <div className="mb-4 flex items-center gap-2">
        <Compass className="h-5 w-5 text-sky-300" />
        <h3 className="text-base font-bold text-white">Recommended Next Steps</h3>
      </div>
      <div className="grid gap-3">
        {recommendations.map((recommendation, index) => (
          <div
            key={recommendation}
            className="flex gap-3 rounded-lg border border-sky-300/20 bg-sky-300/10 p-3 text-sm leading-6 text-sky-50"
          >
            <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full border border-sky-200/20 bg-sky-200/10 font-mono text-xs text-sky-200">
              {index + 1}
            </span>
            <span>{recommendation}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

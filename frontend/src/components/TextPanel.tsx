import { FileText, Loader2, Radar } from "lucide-react";

interface TextPanelProps {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  loading: boolean;
}

const examples = [
  {
    label: "Load suspicious viral post",
    text: "BREAKING!!! AI ROBOTS have secretly taken over hospitals in New York. The government does not want you to know. Share this before it gets deleted!!!"
  },
  {
    label: "Load normal news-style post",
    text: "Researchers are studying how artificial intelligence may affect healthcare jobs over the next decade, according to recent reports from universities and policy groups."
  },
  {
    label: "Load scam-style message",
    text: "URGENT: Your account will be deleted in 10 minutes. Click this link now to verify your identity and avoid permanent suspension!"
  }
];

export default function TextPanel({ value, onChange, onAnalyze, loading }: TextPanelProps) {
  return (
    <div className="panel">
      <div className="mb-4 flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-lg border border-cyan-300/20 bg-cyan-300/10">
          <FileText className="h-5 w-5 text-cyan-200" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-white">Text Scan</h2>
          <p className="text-sm text-white/50">Post, headline, message, or article excerpt</p>
        </div>
      </div>

      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="min-h-56 w-full resize-y rounded-lg border border-white/10 bg-[#05070c] p-4 text-sm leading-7 text-white outline-none transition placeholder:text-white/30 focus:border-cyan-300/60 focus:ring-2 focus:ring-cyan-300/20"
        placeholder="Paste suspicious text here..."
      />

      <div className="mt-4 grid gap-2 lg:grid-cols-3">
        {examples.map((example) => (
          <button
            type="button"
            key={example.label}
            onClick={() => onChange(example.text)}
            className="min-h-11 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-left text-xs font-bold text-white/70 transition hover:border-cyan-300/40 hover:bg-cyan-300/10"
          >
            {example.label}
          </button>
        ))}
      </div>

      <button
        type="button"
        onClick={onAnalyze}
        disabled={!value.trim() || loading}
        className="mt-4 flex h-12 w-full items-center justify-center gap-2 rounded-lg border border-emerald-300/30 bg-emerald-300/10 px-4 text-sm font-black uppercase text-emerald-50 transition hover:bg-emerald-300/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Radar className="h-5 w-5" />}
        Analyze Text
      </button>
    </div>
  );
}

import { AlertCircle, FileImage, FileText, FileVideo, Radar, Shield, type LucideIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import ResultCard from "./components/ResultCard";
import TextPanel from "./components/TextPanel";
import UploadPanel from "./components/UploadPanel";
import { analyzeImage, analyzeText, analyzeVideo } from "./lib/api";
import type { AnalysisResult } from "./lib/types";

type Tab = "image" | "video" | "text";

const scanMessages = [
  "Scanning for manipulation signals...",
  "Checking metadata...",
  "Calculating risk score..."
];

const tabs: Array<{ id: Tab; label: string; icon: LucideIcon }> = [
  { id: "image", label: "Image Upload", icon: FileImage },
  { id: "video", label: "Video Upload", icon: FileVideo },
  { id: "text", label: "Text Scan", icon: FileText }
];

function App() {
  const [activeTab, setActiveTab] = useState<Tab>("image");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    if (!loading) {
      setMessageIndex(0);
      return;
    }
    const interval = window.setInterval(() => {
      setMessageIndex((current) => (current + 1) % scanMessages.length);
    }, 1150);
    return () => window.clearInterval(interval);
  }, [loading]);

  const activeLoadingMessage = useMemo(() => scanMessages[messageIndex], [messageIndex]);

  const runAnalysis = async (task: () => Promise<AnalysisResult>) => {
    setLoading(true);
    setError(null);
    try {
      const response = await task();
      setResult(response);
      window.setTimeout(() => {
        document.getElementById("report")?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 80);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "We could not analyze this file. Try another file type or smaller file.";
      setError(message || "We could not analyze this file. Try another file type or smaller file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="apocalypse-bg min-h-screen text-white">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-5 md:px-6 lg:px-8">
        <section className="grid gap-5 rounded-lg border border-cyan-300/20 bg-[#070910]/90 p-5 shadow-neon lg:grid-cols-[1.1fr_0.9fr] lg:items-end lg:p-7">
          <div>
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-red-300/25 bg-red-300/10 px-3 py-1 text-xs font-bold uppercase text-red-100">
              <Shield className="h-4 w-4" />
              Hoobit Hacks 2026: AI Apocalypse
            </div>
            <h1 className="text-4xl font-black text-white md:text-6xl">TruthShield AI</h1>
            <p className="mt-3 text-lg font-semibold text-cyan-100 md:text-xl">
              Your emergency truth scanner for the AI apocalypse.
            </p>
            <p className="mt-4 max-w-3xl text-sm leading-7 text-white/70 md:text-base">
              Upload an image, scan a video, or paste suspicious text. TruthShield AI gives a risk-based Truth Score
              and explains what warning signs were found.
            </p>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-white/60">
              In the AI apocalypse, the biggest threat may not be robots. It may be losing the ability to know what is real.
            </p>
          </div>
          <div className="relative min-h-44 overflow-hidden rounded-lg border border-white/10 bg-[#05070c] p-4">
            <div className="absolute inset-0 command-grid" />
            <div className="scan-sweep absolute inset-x-0 top-0 h-20" />
            <div className="relative flex h-full flex-col justify-between gap-8">
              <div className="flex items-center justify-between gap-3">
                <span className="text-xs font-bold uppercase text-white/50">Threat Signal Map</span>
                <span className="rounded-full border border-emerald-300/30 bg-emerald-300/10 px-2 py-1 text-xs font-bold text-emerald-100">
                  Local MVP
                </span>
              </div>
              <div className="grid grid-cols-3 gap-3">
                {["Metadata", "Frames", "Language"].map((label, index) => (
                  <div key={label} className="rounded-lg border border-white/10 bg-white/5 p-3">
                    <div className="mb-3 h-1.5 rounded-full bg-white/10">
                      <div
                        className={`h-full rounded-full ${index === 0 ? "bg-cyan-300" : index === 1 ? "bg-orange-300" : "bg-emerald-300"}`}
                        style={{ width: `${72 - index * 14}%` }}
                      />
                    </div>
                    <div className="text-xs font-bold uppercase text-white/60">{label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[0.88fr_1.12fr]">
          <div className="rounded-lg border border-white/10 bg-[#070910]/80 p-4">
            <div className="grid grid-cols-3 gap-2">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    type="button"
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex min-h-14 items-center justify-center gap-2 rounded-lg border px-2 text-xs font-black uppercase transition md:text-sm ${
                      isActive
                        ? "border-cyan-300/40 bg-cyan-300/10 text-cyan-50 shadow-neon"
                        : "border-white/10 bg-white/5 text-white/60 hover:border-cyan-300/25 hover:text-white"
                    }`}
                    aria-pressed={isActive}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    <span className="hidden sm:inline">{tab.label}</span>
                  </button>
                );
              })}
            </div>

            <div className="mt-4">
              {activeTab === "image" && (
                <UploadPanel
                  kind="image"
                  accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
                  file={imageFile}
                  onFileChange={setImageFile}
                  loading={loading}
                  onAnalyze={() => imageFile && runAnalysis(() => analyzeImage(imageFile))}
                />
              )}
              {activeTab === "video" && (
                <UploadPanel
                  kind="video"
                  accept=".mp4,.mov,.webm,video/mp4,video/quicktime,video/webm"
                  file={videoFile}
                  onFileChange={setVideoFile}
                  loading={loading}
                  onAnalyze={() => videoFile && runAnalysis(() => analyzeVideo(videoFile))}
                />
              )}
              {activeTab === "text" && (
                <TextPanel value={text} onChange={setText} loading={loading} onAnalyze={() => runAnalysis(() => analyzeText(text))} />
              )}
            </div>
          </div>

          <div className="min-h-[34rem]">
            {loading && (
              <div className="grid h-full min-h-[34rem] place-items-center rounded-lg border border-cyan-300/20 bg-[#070910]/90 p-6 shadow-neon">
                <div className="w-full max-w-md text-center">
                  <div className="mx-auto mb-6 grid h-24 w-24 place-items-center rounded-full border border-cyan-300/25 bg-cyan-300/10">
                    <Radar className="h-10 w-10 animate-pulse text-cyan-200" />
                  </div>
                  <div className="scanner-bars mb-5" />
                  <p className="text-lg font-black text-white">{activeLoadingMessage}</p>
                  <p className="mt-3 text-sm leading-6 text-white/60">
                    TruthShield AI is checking available signals. This can take longer for videos.
                  </p>
                </div>
              </div>
            )}

            {!loading && error && (
              <div className="rounded-lg border border-red-300/30 bg-red-300/10 p-5 text-red-50 shadow-alert">
                <div className="flex items-start gap-3">
                  <AlertCircle className="mt-1 h-5 w-5 shrink-0 text-red-200" />
                  <div>
                    <h2 className="font-bold">Analysis failed</h2>
                    <p className="mt-2 text-sm leading-6 text-red-50/80">
                      {error || "We could not analyze this file. Try another file type or smaller file."}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {!loading && !result && !error && (
              <div className="grid h-full min-h-[34rem] place-items-center rounded-lg border border-white/10 bg-[#070910]/80 p-6 text-center">
                <div className="max-w-md">
                  <div className="mx-auto mb-5 grid h-16 w-16 place-items-center rounded-lg border border-cyan-300/20 bg-cyan-300/10">
                    <Shield className="h-8 w-8 text-cyan-200" />
                  </div>
                  <h2 className="text-2xl font-black text-white">Awaiting content</h2>
                  <p className="mt-3 text-sm leading-7 text-white/60">
                    Select a tab, submit content, and the combined report will appear here with the Truth Score,
                    evidence bars, warnings, positive signals, and safety recommendations.
                  </p>
                </div>
              </div>
            )}

            {!loading && result && <ResultCard result={result} />}
          </div>
        </section>
      </div>
    </main>
  );
}

export default App;

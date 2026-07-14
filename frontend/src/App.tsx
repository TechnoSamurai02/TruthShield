import { useEffect, useRef, useState } from "react";
import AnalysisReport from "./components/AnalysisReport";
import Header from "./components/Header";
import MediaTabs from "./components/MediaTabs";
import UploadPanel from "./components/UploadPanel";
import { analyzeImage, analyzeVideo } from "./lib/api";
import type { AnalysisResult, MediaType } from "./lib/types";

const SCAN_MESSAGES = [
  "Examining visual signals…",
  "Checking file and camera metadata…",
  "Comparing forensic evidence…",
  "Reviewing provenance and source leads…",
  "Preparing your Truth Score…"
];

const EMPTY_FILES: Record<MediaType, File | null> = {
  image: null,
  video: null
};

function App() {
  const [activeMode, setActiveMode] = useState<MediaType>("image");
  const [files, setFiles] = useState<Record<MediaType, File | null>>(EMPTY_FILES);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [completedAt, setCompletedAt] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [messageIndex, setMessageIndex] = useState(0);
  const reportRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!loading) {
      setMessageIndex(0);
      return;
    }

    const interval = window.setInterval(() => {
      setMessageIndex((current) => (current + 1) % SCAN_MESSAGES.length);
    }, 1600);
    return () => window.clearInterval(interval);
  }, [loading]);

  useEffect(() => {
    if (!result) return;
    const frame = window.requestAnimationFrame(() => {
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      reportRef.current?.focus({ preventScroll: true });
      reportRef.current?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [result]);

  const handleModeChange = (mode: MediaType) => {
    if (loading || mode === activeMode) return;
    setActiveMode(mode);
    setResult(null);
    setCompletedAt(null);
    setError(null);
  };

  const handleFileChange = (mode: MediaType, file: File | null) => {
    setFiles((current) => ({ ...current, [mode]: file }));
    setResult(null);
    setCompletedAt(null);
    setError(null);
  };

  const runAnalysis = async () => {
    const file = files[activeMode];
    if (!file || loading) return;

    setLoading(true);
    setError(null);
    try {
      const response = activeMode === "image" ? await analyzeImage(file) : await analyzeVideo(file);
      setCompletedAt(new Date());
      setResult(response);
    } catch (caught) {
      const fallback = "We could not analyze this file. Try another supported file or try again shortly.";
      setError(caught instanceof Error && caught.message ? caught.message : fallback);
    } finally {
      setLoading(false);
    }
  };

  const startNewAnalysis = () => {
    setFiles((current) => ({ ...current, [activeMode]: null }));
    setResult(null);
    setCompletedAt(null);
    setError(null);
  };

  return (
    <main id="top" className="app-shell">
      <div className="page-frame">
        <Header />

        {!result ? (
          <section className="hero" aria-labelledby="hero-title">
            <h1 id="hero-title">TruthShield AI</h1>
            <p className="hero-line">Read between the pixels.</p>
            <p className="hero-copy">
              Analyze suspicious images and videos for warning signs, then receive an explainable, risk-based Truth Score.
            </p>
          </section>
        ) : null}

        <section className={`analysis-stage${result ? " has-result" : ""}`} aria-label="Media analysis">
          <MediaTabs activeMode={activeMode} disabled={loading} onChange={handleModeChange} />

          {result && completedAt ? (
            <AnalysisReport ref={reportRef} result={result} completedAt={completedAt} onNewAnalysis={startNewAnalysis} />
          ) : (
            <UploadPanel
              key={activeMode}
              kind={activeMode}
              file={files[activeMode]}
              error={error}
              loading={loading}
              loadingMessage={SCAN_MESSAGES[messageIndex]}
              onFileChange={(file) => handleFileChange(activeMode, file)}
              onValidationError={setError}
              onAnalyze={runAnalysis}
            />
          )}
        </section>
      </div>
    </main>
  );
}

export default App;

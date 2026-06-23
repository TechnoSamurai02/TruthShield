import { FileImage, FileVideo, Loader2, UploadCloud } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";

interface UploadPanelProps {
  kind: "image" | "video";
  accept: string;
  file: File | null;
  onFileChange: (file: File | null) => void;
  onAnalyze: () => void;
  loading: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

export default function UploadPanel({ kind, accept, file, onFileChange, onAnalyze, loading }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const Icon = kind === "image" ? FileImage : FileVideo;

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const nextFile = event.dataTransfer.files?.[0];
    if (nextFile) onFileChange(nextFile);
  };

  return (
    <div className="panel">
      <div className="mb-4 flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-lg border border-cyan-300/20 bg-cyan-300/10">
          <Icon className="h-5 w-5 text-cyan-200" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-white">{kind === "image" ? "Image Upload" : "Video Upload"}</h2>
          <p className="text-sm text-white/50">
            {kind === "image" ? ".jpg, .jpeg, .png, .webp" : ".mp4, .mov, .webm"}
          </p>
        </div>
      </div>

      <div
        className={`grid min-h-52 place-items-center rounded-lg border border-dashed p-5 text-center transition ${
          dragging ? "border-cyan-300 bg-cyan-300/10" : "border-white/20 bg-white/[0.03]"
        }`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={accept}
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
        />
        <div>
          <UploadCloud className="mx-auto mb-4 h-10 w-10 text-cyan-200" />
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="rounded-lg border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-sm font-bold text-cyan-50 transition hover:bg-cyan-300/20"
          >
            Choose File
          </button>
          <p className="mt-3 text-sm text-white/50">Drag and drop a file here</p>
        </div>
      </div>

      {file && (
        <div className="mt-4 rounded-lg border border-white/10 bg-white/5 p-3">
          <div className="text-sm font-semibold text-white">{file.name}</div>
          <div className="mt-1 text-xs text-white/50">{formatBytes(file.size)}</div>
        </div>
      )}

      <button
        type="button"
        onClick={onAnalyze}
        disabled={!file || loading}
        className="mt-4 flex h-12 w-full items-center justify-center gap-2 rounded-lg border border-emerald-300/30 bg-emerald-300/10 px-4 text-sm font-black uppercase text-emerald-50 transition hover:bg-emerald-300/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <UploadCloud className="h-5 w-5" />}
        Analyze Content
      </button>
    </div>
  );
}

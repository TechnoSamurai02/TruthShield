import { useEffect, useRef, useState, type DragEvent } from "react";
import type { MediaType } from "../lib/types";

interface UploadPanelProps {
  kind: MediaType;
  file: File | null;
  error: string | null;
  errorTitle: string;
  loading: boolean;
  loadingMessage: string;
  onFileChange: (file: File | null) => void;
  onValidationError: (message: string | null) => void;
  onAnalyze: () => void;
}

const FILE_RULES = {
  image: {
    extensions: [".jpg", ".jpeg", ".png", ".webp"],
    accept: ".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp",
    formats: "JPG, JPEG, PNG, WEBP",
    maxBytes: 15 * 1024 * 1024,
    maxLabel: "15 MB"
  },
  video: {
    extensions: [".mp4", ".mov", ".webm"],
    accept: ".mp4,.mov,.webm,video/mp4,video/quicktime,video/webm",
    formats: "MP4, MOV, WEBM",
    maxBytes: 80 * 1024 * 1024,
    maxLabel: "80 MB"
  }
} as const;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kilobytes = bytes / 1024;
  if (kilobytes < 1024) return `${kilobytes.toFixed(1)} KB`;
  return `${(kilobytes / 1024).toFixed(1)} MB`;
}

function formatDuration(seconds: number): string {
  const totalSeconds = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(totalSeconds / 60);
  const remainder = totalSeconds % 60;
  return `${minutes}:${remainder.toString().padStart(2, "0")}`;
}

function fileExtension(filename: string): string {
  const dotIndex = filename.lastIndexOf(".");
  return dotIndex >= 0 ? filename.slice(dotIndex).toLowerCase() : "";
}

function validateFile(file: File, kind: MediaType): string | null {
  const rules = FILE_RULES[kind];
  const extension = fileExtension(file.name);
  if (file.size === 0) return "This file is empty. Choose a file that contains media.";
  if (!rules.extensions.some((allowedExtension) => allowedExtension === extension)) {
    return `Unsupported file type. Choose ${rules.formats.toLowerCase()}.`;
  }
  if (file.size > rules.maxBytes) {
    return `This file is too large. ${kind === "image" ? "Images" : "Videos"} must be ${rules.maxLabel} or smaller.`;
  }
  return null;
}

function useMediaPreview(file: File | null, kind: MediaType) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<string | null>(null);

  useEffect(() => {
    setPreviewUrl(null);
    setMetadata(null);
    if (!file) return;

    let disposed = false;
    let image: HTMLImageElement | null = null;
    let video: HTMLVideoElement | null = null;
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    if (kind === "image") {
      image = new Image();
      image.onload = () => {
        if (!disposed && image) setMetadata(`${image.naturalWidth} × ${image.naturalHeight}`);
      };
      image.src = objectUrl;
    } else {
      video = document.createElement("video");
      video.preload = "metadata";
      video.onloadedmetadata = () => {
        if (!disposed && video && Number.isFinite(video.duration)) {
          setMetadata(formatDuration(video.duration));
        }
      };
      video.src = objectUrl;
    }

    return () => {
      disposed = true;
      if (image) image.onload = null;
      if (video) video.onloadedmetadata = null;
      URL.revokeObjectURL(objectUrl);
    };
  }, [file, kind]);

  return { previewUrl, metadata };
}

function UploadArrow() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M12 18V5m0 0-5 5m5-5 5 5" />
    </svg>
  );
}

function ForwardArrow() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M5 12h14m0 0-5-5m5 5-5 5" />
    </svg>
  );
}

export default function UploadPanel({
  kind,
  file,
  error,
  errorTitle,
  loading,
  loadingMessage,
  onFileChange,
  onValidationError,
  onAnalyze
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);
  const { previewUrl, metadata } = useMediaPreview(file, kind);
  const rules = FILE_RULES[kind];
  const noun = kind === "image" ? "image" : "video";

  const chooseFile = () => inputRef.current?.click();

  const acceptCandidate = (candidate: File | null) => {
    if (!candidate) return;
    const validationMessage = validateFile(candidate, kind);
    if (validationMessage) {
      onFileChange(null);
      onValidationError(validationMessage);
      if (inputRef.current) inputRef.current.value = "";
      return;
    }

    onValidationError(null);
    onFileChange(candidate);
  };

  const handleDragEnter = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (loading) return;
    dragDepth.current += 1;
    setDragging(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    dragDepth.current -= 1;
    if (dragDepth.current <= 0) {
      dragDepth.current = 0;
      setDragging(false);
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    if (!loading) acceptCandidate(event.dataTransfer.files?.[0] ?? null);
  };

  const removeFile = () => {
    onFileChange(null);
    onValidationError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div
      id={`panel-${kind}`}
      className="upload-panel"
      role="tabpanel"
      aria-labelledby={`tab-${kind}`}
      aria-busy={loading}
    >
      <div className={`upload-boundary${dragging ? " is-dragging" : ""}`}>
        <div
          className="drop-surface"
          onDragEnter={handleDragEnter}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept={rules.accept}
            aria-label={`Choose ${noun} file`}
            aria-describedby={`${kind}-file-help`}
            disabled={loading}
            onChange={(event) => {
              acceptCandidate(event.currentTarget.files?.[0] ?? null);
              event.currentTarget.value = "";
            }}
          />
          <span className="upload-glyph">
            <UploadArrow />
          </span>
          <p className="drop-instruction">Drag and drop {noun === "image" ? "an" : "a"} {noun} here</p>
          <span className="drop-or">or</span>
          <button type="button" className="choose-button" onClick={chooseFile} disabled={loading}>
            Choose a file
          </button>
          <p id={`${kind}-file-help`} className="file-help">
            {rules.formats} <span aria-hidden="true">·</span> up to {rules.maxLabel}
          </p>
        </div>

        {file ? (
          <div className="selected-file" aria-live="polite">
            <div className="file-preview" aria-hidden="true">
              {previewUrl ? (
                kind === "image" ? (
                  <img src={previewUrl} alt="" />
                ) : (
                  <video src={previewUrl} muted playsInline preload="metadata" />
                )
              ) : null}
            </div>
            <div className="file-copy">
              <p className="file-name">{file.name}</p>
              <p className="file-meta">
                {formatBytes(file.size)}{metadata ? ` · ${metadata}` : ""}
              </p>
            </div>
            <button
              type="button"
              className="remove-file"
              onClick={removeFile}
              disabled={loading}
              aria-label={`Remove ${file.name}`}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <path d="m6 6 12 12M18 6 6 18" />
              </svg>
            </button>
          </div>
        ) : null}
      </div>

      <button type="button" className="analyze-button" onClick={onAnalyze} disabled={!file || loading}>
        <span>{loading ? `Analyzing ${noun}…` : `Analyze ${noun}`}</span>
        {!loading ? <ForwardArrow /> : null}
      </button>

      {loading ? (
        <div className="loading-status" aria-live="polite">
          <span className="loading-line" aria-hidden="true" />
          <span>{loadingMessage}</span>
        </div>
      ) : null}

      {error ? (
        <div className="inline-error" role="alert">
          <div>
            <strong>{errorTitle}</strong>
            <p>{error}</p>
          </div>
          <button type="button" onClick={chooseFile}>
            Try another file
          </button>
        </div>
      ) : null}

      <p className="report-note">Your report will appear here after analysis.</p>
    </div>
  );
}

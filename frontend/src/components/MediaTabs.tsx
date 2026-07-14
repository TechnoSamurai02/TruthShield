import { useRef, type KeyboardEvent } from "react";
import type { MediaType } from "../lib/types";

interface MediaTabsProps {
  activeMode: MediaType;
  disabled?: boolean;
  onChange: (mode: MediaType) => void;
}

const MODES: Array<{ id: MediaType; label: string }> = [
  { id: "image", label: "Image" },
  { id: "video", label: "Video" }
];

export default function MediaTabs({ activeMode, disabled = false, onChange }: MediaTabsProps) {
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (disabled) return;

    let nextIndex = index;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % MODES.length;
    else if (event.key === "ArrowLeft") nextIndex = (index - 1 + MODES.length) % MODES.length;
    else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = MODES.length - 1;
    else return;

    event.preventDefault();
    const nextMode = MODES[nextIndex].id;
    onChange(nextMode);
    tabRefs.current[nextIndex]?.focus();
  };

  return (
    <div className="media-tabs" role="tablist" aria-label="Analysis type">
      {MODES.map((mode, index) => {
        const selected = activeMode === mode.id;
        return (
          <button
            key={mode.id}
            ref={(node) => {
              tabRefs.current[index] = node;
            }}
            type="button"
            id={`tab-${mode.id}`}
            role="tab"
            aria-selected={selected}
            aria-controls={`panel-${mode.id}`}
            tabIndex={selected ? 0 : -1}
            disabled={disabled}
            onClick={() => onChange(mode.id)}
            onKeyDown={(event) => handleKeyDown(event, index)}
          >
            {mode.label}
          </button>
        );
      })}
    </div>
  );
}

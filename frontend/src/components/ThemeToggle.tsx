import { useEffect, useState } from "react";

type Theme = "light" | "dark";

const THEME_STORAGE_KEY = "truthshield-theme";

function themeFromDocument(): Theme {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function hasStoredTheme(): boolean {
  try {
    return Boolean(window.localStorage.getItem(THEME_STORAGE_KEY));
  } catch {
    return false;
  }
}

function storeTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // The visible theme can still change when storage is unavailable.
  }
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(themeFromDocument);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const followSystemTheme = (event: MediaQueryListEvent) => {
      if (hasStoredTheme()) return;
      const nextTheme: Theme = event.matches ? "dark" : "light";
      document.documentElement.dataset.theme = nextTheme;
      document.documentElement.style.colorScheme = nextTheme;
      setTheme(nextTheme);
    };

    media.addEventListener("change", followSystemTheme);
    return () => media.removeEventListener("change", followSystemTheme);
  }, []);

  const toggleTheme = () => {
    const nextTheme: Theme = theme === "light" ? "dark" : "light";
    document.documentElement.dataset.themeTransition = "true";
    document.documentElement.dataset.theme = nextTheme;
    document.documentElement.style.colorScheme = nextTheme;
    storeTheme(nextTheme);
    setTheme(nextTheme);

    window.setTimeout(() => {
      delete document.documentElement.dataset.themeTransition;
    }, 300);
  };

  const targetTheme = theme === "light" ? "dark" : "light";

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggleTheme}
      aria-label={`Switch to ${targetTheme} mode`}
    >
      <span>{targetTheme === "dark" ? "Dark" : "Light"}</span>
      <span className="theme-mark" aria-hidden="true" />
    </button>
  );
}

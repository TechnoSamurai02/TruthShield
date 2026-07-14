import ThemeToggle from "./ThemeToggle";

export default function Header() {
  return (
    <header className="site-header">
      <a className="wordmark" href="#top" aria-label="TruthShield AI home">
        TruthShield AI
      </a>
      <ThemeToggle />
    </header>
  );
}

import { useEffect, useRef, useState } from "react";

const DELIVERABLES = [
  {
    number: "01",
    title: "A clear assessment",
    description:
      "A conservative result that can also remain inconclusive."
  },
  {
    number: "02",
    title: "Evidence, in context",
    description:
      "Warnings and supporting signals explaining what affected the result."
  },
  {
    number: "03",
    title: "Uncertainty and next steps",
    description:
      "Limitations, evidence coverage, and practical verification guidance."
  }
] as const;

function shouldRevealImmediately(): boolean {
  if (typeof window === "undefined") return true;
  return (
    window.matchMedia("(prefers-reduced-motion: reduce)").matches ||
    !("IntersectionObserver" in window)
  );
}

export default function WhatYouReceive() {
  const sectionRef = useRef<HTMLElement | null>(null);
  const [isVisible, setIsVisible] = useState(shouldRevealImmediately);

  useEffect(() => {
    if (isVisible || !sectionRef.current) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        setIsVisible(true);
        observer.disconnect();
      },
      { threshold: 0.16, rootMargin: "0px 0px -8% 0px" }
    );

    observer.observe(sectionRef.current);
    return () => observer.disconnect();
  }, [isVisible]);

  return (
    <section
      ref={sectionRef}
      className={`what-you-receive${isVisible ? " is-visible" : ""}`}
      aria-labelledby="what-you-receive-title"
    >
      <div className="receive-intro">
        <h2 id="what-you-receive-title">What you’ll receive</h2>
        <p>TruthShield turns technical warning signs into a clear, explainable assessment.</p>
      </div>

      <ol className="receive-list">
        {DELIVERABLES.map((item) => (
          <li key={item.number}>
            <span className="receive-number" aria-hidden="true">
              {item.number}
            </span>
            <h3>{item.title}</h3>
            <p>{item.description}</p>
          </li>
        ))}
      </ol>

      <p className="receive-disclaimer">
        TruthShield provides risk-based indicators, not proof. Consider each report alongside source context and
        independent verification.
      </p>
    </section>
  );
}

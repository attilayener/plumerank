"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { StreamState } from "@/lib/types";

/**
 * 56 px of chrome: a wordmark, the three routes, and a live/idle dot in --signal.
 *
 * No theme toggle (there is one theme), no sidebar, no hamburger, no search. Six web
 * routes became three, so a navigation component that could hide anything would be
 * hiding most of a three-item list.
 */
const LINKS: { href: string; label: string }[] = [
  { href: "/", label: "The Wall" },
  { href: "/results", label: "Results" },
  { href: "/judge", label: "For judges" },
];

const DOT: Record<StreamState, { cls: string; text: string; short: string }> = {
  connecting: { cls: "dot--idle", text: "connecting", short: "…" },
  connected: { cls: "dot--idle", text: "connected", short: "on" },
  live: { cls: "dot--live", text: "live", short: "live" },
  idle: { cls: "dot--idle", text: "idle", short: "idle" },
  down: { cls: "dot--down", text: "stream down", short: "down" },
  unavailable: { cls: "dot--down", text: "pipeline unavailable", short: "off" },
};

export function TopBar({ stream }: { stream?: StreamState }) {
  const pathname = usePathname();
  const dot = stream ? DOT[stream] : null;

  return (
    <header className="topbar">
      <Link href="/" style={{ color: "var(--ink-0)", fontWeight: 600, letterSpacing: "-0.01em" }}>
        PlumeRank
      </Link>
      <nav style={{ display: "flex", gap: 18, flexWrap: "wrap", rowGap: 0 }}>
        {LINKS.map((l) => {
          const active = pathname === l.href;
          return (
            <Link
              key={l.href}
              href={l.href}
              style={{
                color: active ? "var(--ink-0)" : "var(--ink-2)",
                fontSize: "var(--size-label)",
                borderBottom: active ? "1px solid var(--signal)" : "1px solid transparent",
                paddingBottom: 2,
              }}
            >
              {l.label}
            </Link>
          );
        })}
      </nav>
      {dot ? (
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <span className={`dot ${dot.cls}`} aria-hidden="true" title={dot.text} />
          <span className="label topbar-status" style={{ letterSpacing: "0.06em" }}>
            {dot.text}
          </span>
          {/* At phone width the full word does not fit beside the nav; a short visible
              word keeps the state readable instead of leaving a bare dot. */}
          <span className="label topbar-status-short" aria-hidden="true">
            {dot.short}
          </span>
        </span>
      ) : null}
    </header>
  );
}

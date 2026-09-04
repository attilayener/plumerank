"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { StreamState } from "@/lib/types";

/**
 * 56 px of chrome: a wordmark, the two routes, and a live/idle dot in --signal.
 *
 * No theme toggle (there is one theme), no sidebar, no hamburger, no search. Six web
 * routes became two, so a navigation component that could hide anything would be
 * hiding half of a two-item list.
 */
const LINKS: { href: string; label: string }[] = [
  { href: "/", label: "The Wall" },
  { href: "/results", label: "Results" },
];

const DOT: Record<StreamState, { cls: string; text: string }> = {
  connecting: { cls: "dot--idle", text: "connecting" },
  live: { cls: "dot--live", text: "live" },
  idle: { cls: "dot--idle", text: "idle" },
  down: { cls: "dot--down", text: "stream down" },
};

export function TopBar({ stream }: { stream?: StreamState }) {
  const pathname = usePathname();
  const dot = stream ? DOT[stream] : null;

  return (
    <header className="topbar">
      <Link href="/" style={{ color: "var(--ink-0)", fontWeight: 600, letterSpacing: "-0.01em" }}>
        PlumeRank
      </Link>
      <nav style={{ display: "flex", gap: 18 }}>
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
          <span className={`dot ${dot.cls}`} aria-hidden="true" />
          <span className="label" style={{ letterSpacing: "0.06em" }}>
            {dot.text}
          </span>
        </span>
      ) : null}
    </header>
  );
}

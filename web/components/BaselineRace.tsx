"use client";

import { minutes, ABSENT } from "@/lib/format";
import type { Benchmark } from "@/lib/types";

/**
 * Three lanes on ONE clock: PlumeRank, B1 (the same CNN on every frame) and B2 (frame
 * differencing), all calibrated to the same alert budget on the same confusion set.
 *
 * One clock is the argument. Three systems each with its own axis is three claims a
 * reader has to reconcile; three markers on one axis is a distance, and a distance is
 * the thing a fire agency is buying. The gaps are drawn and labelled in minutes for the
 * same reason.
 *
 * There is no baseline switcher and no budget slider. Both were controls that existed
 * to be demoed: the lanes are already all three systems, and the operating curve
 * already carries budget-vs-TTD for every one of them.
 */

const LANE_COLOURS: Record<string, string> = {
  plumerank: "var(--signal)",
  b1: "var(--ink-2)",
  b2: "var(--fog)",
};

export function BaselineRace({ data }: { data: Benchmark }) {
  const lanes = data.systems.filter((s) => s.p50_minutes !== null);
  if (lanes.length === 0) {
    return (
      <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)" }}>
        per-system results not published in results/bench_&lt;date&gt;.json
      </p>
    );
  }

  const slowest = Math.max(...lanes.map((l) => l.p50_minutes ?? 0));
  const fastest = Math.min(...lanes.map((l) => l.p50_minutes ?? 0));
  const scale = (v: number) => `${(v / (slowest || 1)) * 100}%`;

  return (
    <div>
      {lanes.map((lane) => {
        const v = lane.p50_minutes ?? 0;
        const gap = v - fastest;
        return (
          <div key={lane.key} style={{ marginBottom: 14 }}>
            <div
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: 10,
                fontSize: "var(--size-label)",
                color: "var(--ink-2)",
                marginBottom: 4,
              }}
            >
              <span style={{ color: "var(--ink-0)" }}>{lane.label}</span>
              <span className="mono num" style={{ color: "var(--ink-1)" }}>
                {minutes(lane.p50_minutes)} min
              </span>
              <span>
                {gap > 0 ? `+${minutes(gap)} min behind` : "first"}
                {lane.n_missed !== null ? ` - ${lane.n_missed} missed` : ""}
              </span>
            </div>
            <div style={{ position: "relative", height: 14, background: "var(--surface-raised)", borderRadius: 2 }}>
              <div
                style={{
                  position: "absolute",
                  left: 0,
                  top: 0,
                  bottom: 0,
                  width: scale(v),
                  background: LANE_COLOURS[lane.key] ?? "var(--ink-3)",
                  opacity: 0.35,
                }}
              />
              <div
                style={{
                  position: "absolute",
                  left: `calc(${scale(v)} - 2px)`,
                  top: -3,
                  bottom: -3,
                  width: 4,
                  background: LANE_COLOURS[lane.key] ?? "var(--ink-3)",
                }}
                aria-hidden="true"
              />
            </div>
          </div>
        );
      })}
      <p style={{ fontSize: "var(--size-label)", color: "var(--ink-3)", margin: 0 }}>
        one clock, minutes after ignition. Every lane is calibrated to the same alert
        budget on the same confusion set, and the learned component is IDENTICAL in
        PlumeRank and in B1 -- so the gap is the ranking primitive and not a bigger
        model. Slowest lane: {minutes(slowest)} min
        {lanes.length < 3 ? ` (only ${lanes.length} of 3 lanes published)` : ""}.
        {slowest === 0 ? ` scale ${ABSENT}` : ""}
      </p>
    </div>
  );
}

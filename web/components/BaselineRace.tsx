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
  // Keyed by the system ids bench.py actually writes. Until 2026-10-04 these were `b1`
  // and `b2`, matched nothing, and both baseline lanes fell through to --ink-3.
  b1_cnn_grid: "var(--ink-2)",
  b2_frame_diff: "var(--fog)",
};

export function BaselineRace({ data }: { data: Benchmark }) {
  // Every published system is a lane. A null p50 with a detection count is not a missing
  // number: it is a system that detected fewer than half the arm's fires, so its median
  // is infinite by metrics.py's own definition (a miss is an infinitely late detection).
  // Until 2026-09-17 such lanes were FILTERED OUT and the caption read "only 2 of 3 lanes
  // published" -- which, on the real DEV arm where B1 finds 1 of 29 and B2 finds 3 of 29,
  // hid the most important comparison on the page behind a publishing apology.
  const lanes = data.systems.filter((s) => s.p50_minutes !== null || s.n_detected !== null);
  if (lanes.length === 0) {
    return (
      <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)" }}>
        per-system results not published yet — it appears here after the held-out run
      </p>
    );
  }

  const timed = lanes.filter((l) => l.p50_minutes !== null);
  const slowest = timed.length ? Math.max(...timed.map((l) => l.p50_minutes ?? 0)) : 0;
  const fastest = timed.length ? Math.min(...timed.map((l) => l.p50_minutes ?? 0)) : 0;
  const scale = (v: number) => `${(v / (slowest || 1)) * 100}%`;

  return (
    <div>
      {lanes.map((lane) => {
        const noMedian = lane.p50_minutes === null;
        const v = lane.p50_minutes ?? 0;
        const gap = v - fastest;
        const colour = LANE_COLOURS[lane.key] ?? "var(--ink-3)";
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
              {noMedian ? (
                <span className="mono num" style={{ color: "var(--ink-1)" }}>
                  no median
                </span>
              ) : (
                <span className="mono num" style={{ color: "var(--ink-1)" }}>
                  {minutes(lane.p50_minutes)} min
                </span>
              )}
              <span>
                {noMedian
                  ? `fewer than half detected${
                      lane.n_detected !== null && lane.n !== null
                        ? ` - ${lane.n_detected} of ${lane.n}`
                        : ""
                    }`
                  : `${gap > 0 ? `+${minutes(gap)} min behind` : "first"}${
                      lane.n_missed !== null ? ` - ${lane.n_missed} missed` : ""
                    }`}
              </span>
            </div>
            <div style={{ position: "relative", height: 14, background: "var(--surface-raised)", borderRadius: 2 }}>
              {noMedian ? (
                // A lane that never reaches its median runs off the right edge of the
                // clock: full width, hatched, no marker -- there is no minute to mark.
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    backgroundImage: `repeating-linear-gradient(135deg, ${colour} 0 3px, transparent 3px 8px)`,
                    opacity: 0.35,
                  }}
                  aria-label={`${lane.label}: no median, fewer than half detected`}
                />
              ) : (
                <>
                  <div
                    style={{
                      position: "absolute",
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: scale(v),
                      background: colour,
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
                      background: colour,
                    }}
                    aria-hidden="true"
                  />
                </>
              )}
            </div>
          </div>
        );
      })}
      <p style={{ fontSize: "var(--size-label)", color: "var(--ink-3)", margin: 0 }}>
        one clock, minutes after ignition. Every lane&apos;s threshold is solved to the same
        alert budget on the same confusion set, and the learned component is IDENTICAL in
        PlumeRank and in B1 -- so the gap is not a bigger model. It is everything around
        the model: the classical statistics, the per-camera baseline (PlumeRank only) and
        the budgeted ranking. A hatched lane detected fewer than half the fires: its median is off the
        clock, which is a result and not a gap in the file.
        {timed.length ? ` Slowest timed lane: ${minutes(slowest)} min.` : ""}
        {lanes.length < 3 ? ` (only ${lanes.length} of 3 lanes published)` : ""}
        {timed.length && slowest === 0 ? ` scale ${ABSENT}` : ""}
      </p>
    </div>
  );
}

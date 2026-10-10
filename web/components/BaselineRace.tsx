"use client";

import { minutes, rate, ABSENT } from "@/lib/format";
import type { Benchmark, SystemResult } from "@/lib/types";

/**
 * Lanes on ONE clock: PlumeRank, and B1 (the same CNN on every frame) and B2 (frame
 * differencing) twice -- through PlumeRank's own per-camera baseline (the like-for-like
 * comparison, led with) and as deployed. Every threshold is solved to the same alert
 * budget on the same confusion set.
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
  b1_cnn_grid_rebased: "var(--ink-2)",
  b2_frame_diff_rebased: "var(--fog)",
};

function Lane({
  lane,
  fastest,
  slowest,
}: {
  lane: SystemResult;
  fastest: number;
  slowest: number;
}) {
  const noMedian = lane.p50_minutes === null;
  const v = lane.p50_minutes ?? 0;
  const gap = v - fastest;
  const colour = LANE_COLOURS[lane.key] ?? "var(--ink-3)";
  const scale = (x: number) => `${(x / (slowest || 1)) * 100}%`;
  return (
    <div style={{ marginBottom: 14 }}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 10,
          flexWrap: "wrap",
          fontSize: "var(--size-label)",
          color: "var(--ink-2)",
          marginBottom: 4,
        }}
      >
        <span style={{ color: "var(--ink-0)" }}>{lane.label}</span>
        <span className="mono num" style={{ color: "var(--ink-1)" }}>
          {noMedian ? "no median" : `${minutes(lane.p50_minutes)} min`}
        </span>
        <span>
          {lane.n_detected !== null && lane.n !== null ? `${lane.n_detected} of ${lane.n} detected` : ""}
          {noMedian
            ? ""
            : gap > 0
              ? ` — +${minutes(gap)} min behind`
              : " — first"}
        </span>
        {lane.arm_alerts_per_camera_night !== null ? (
          <span className="num" style={{ color: "var(--ink-3)", fontWeight: 400 }}>
            {rate(lane.arm_alerts_per_camera_night)} realised on these fires
          </span>
        ) : null}
      </div>
      <div style={{ position: "relative", height: 14, background: "var(--surface-raised)", borderRadius: 2 }}>
        {noMedian ? (
          // A lane that never reaches its median runs off the right edge of the clock:
          // full width, hatched, no marker -- there is no minute to mark.
          <div
            style={{
              position: "absolute",
              inset: 0,
              backgroundImage: `repeating-linear-gradient(135deg, ${colour} 0 3px, transparent 3px 8px)`,
              opacity: 0.35,
            }}
            role="img"
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
}

function Group({ title, lanes, fastest, slowest }: { title: string; lanes: SystemResult[]; fastest: number; slowest: number }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <p className="label" style={{ margin: "0 0 8px" }}>{title}</p>
      {lanes.map((lane) => (
        <Lane key={lane.key} lane={lane} fastest={fastest} slowest={slowest} />
      ))}
    </div>
  );
}

export function BaselineRace({ data }: { data: Benchmark }) {
  // Every published system is a lane. A null p50 with a detection count is not a missing
  // number: it is a system that detected fewer than half the arm's fires, so its median
  // is infinite by metrics.py's own definition (a miss is an infinitely late detection).
  const published = (s: SystemResult) => s.p50_minutes !== null || s.n_detected !== null;
  const lanes = data.systems.filter(published);
  const rebased = data.rebased.filter(published);
  if (lanes.length === 0) {
    return (
      <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)" }}>
        per-system results not published yet — they appear here after the held-out run
      </p>
    );
  }

  // One clock across both groups, so a distance in one group reads the same in the other.
  const timed = [...lanes, ...rebased].filter((l) => l.p50_minutes !== null);
  const slowest = timed.length ? Math.max(...timed.map((l) => l.p50_minutes ?? 0)) : 0;
  const fastest = timed.length ? Math.min(...timed.map((l) => l.p50_minutes ?? 0)) : 0;

  const headline = lanes.filter((l) => l.key === "plumerank");
  const asDeployed = lanes.filter((l) => l.key !== "plumerank");

  return (
    <div>
      {rebased.length > 0 ? (
        <>
          <Group
            title="like-for-like — every system through the same per-camera baseline"
            lanes={[...headline, ...rebased]}
            fastest={fastest}
            slowest={slowest}
          />
          <Group
            title="baselines as deployed — no per-camera baseline"
            lanes={asDeployed}
            fastest={fastest}
            slowest={slowest}
          />
        </>
      ) : (
        <Group title="all systems" lanes={lanes} fastest={fastest} slowest={slowest} />
      )}
      <p style={{ fontSize: "var(--size-label)", color: "var(--ink-3)", margin: 0 }}>
        One clock, minutes after ignition. Every threshold is solved to the same alert
        budget on the same fire-free confusion set (every system lands at the same rate
        there); on the fire footage each system then realises its own rate, shown per lane.
        The learned component is identical in PlumeRank and in B1, so the like-for-like gap
        is not a bigger model. B1 as deployed scores with saturated CNN probabilities, so
        its threshold sits at the very top of the scale. A hatched lane detected fewer than
        half the fires, so it has no median — a result, not a gap in the file.
        {timed.length && slowest === 0 ? ` scale ${ABSENT}` : ""}
      </p>
    </div>
  );
}

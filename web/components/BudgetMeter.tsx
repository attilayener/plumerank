"use client";

import { useEffect, useRef } from "react";
import { rate, ABSENT } from "@/lib/format";
import type { Benchmark, WatchMeta, WatchRow } from "@/lib/types";

/**
 * 80 DISCRETE COUNTABLE SLOTS -- 2 per camera, grouped by camera. Never a percentage,
 * never a gauge, never an arc.
 *
 * A dispatcher's night contains a countable number of interruptions. "63% of budget
 * used" does not say that; forty little pairs of boxes, of which eleven are filled,
 * says it without arithmetic and says it from across a control room. The grouping by
 * camera is the second half of the idea: the budget is spent PER CAMERA, so a meter
 * that pooled 80 slots into one bar would show a number the system does not actually
 * enforce.
 *
 * Two rates share one name in this project and this component publishes both, labelled,
 * because publishing either one unlabelled is how they get confused:
 *   - the SLOTS are budget-capped -- they are alerts that were actually spent tonight;
 *   - "realised (uncapped)" comes from /api/benchmark and counts threshold crossings
 *     WITHOUT the cap, because that is the quantity rank.solve_threshold was calibrated
 *     on. Capped counting makes every threshold satisfy the target and the solver
 *     degenerates.
 */

/** Slots per camera. The operating design is 2 (constants.ALERTS_PER_CAMERA_PER_NIGHT)
 *  and that is a shape, not a measurement -- but the server may still override it, so
 *  the meta value wins whenever the stream publishes one. */
const DEFAULT_SLOTS_PER_CAMERA = 2;

function Slot({ filled, justFilled }: { filled: boolean; justFilled: boolean }) {
  return (
    <span
      className={`slot${filled ? " slot--filled" : ""}${justFilled ? " slot--just-filled" : ""}`}
      aria-hidden="true"
    />
  );
}

export function BudgetMeter({
  meta,
  rows,
  benchmark,
}: {
  meta: WatchMeta | null;
  rows: WatchRow[];
  benchmark: Benchmark | null;
}) {
  const perCamera = meta?.budget_per_camera ?? DEFAULT_SLOTS_PER_CAMERA;
  const cameras = meta?.cameras ?? rows.map((r) => r.camera_id);

  const spentByCamera = new Map<string, number>();
  for (const r of rows) spentByCamera.set(r.camera_id, r.budget_spent);

  // Which slots filled since the previous tick -- the only thing the 160 ms scale-in is
  // allowed to fire on. Animating every filled slot on every tick would make the second
  // of the product's two animations mean "a tick happened", which is not worth a
  // channel.
  const previous = useRef<Map<string, number>>(new Map());
  // The first tick a viewer sees is not a fill. A judge who opens the page at 04:50 is
  // joining a night already in progress, so every slot the roster has already spent
  // would otherwise scale in at once -- which would make the product's second and last
  // animation mean "the page loaded", the exact thing the two-animation budget forbids.
  const seeded = useRef<boolean>(false);
  const justFilled = new Map<string, number>();
  if (seeded.current) {
    for (const cam of cameras) {
      const before = previous.current.get(cam) ?? 0;
      const now = spentByCamera.get(cam) ?? 0;
      if (now > before) justFilled.set(cam, before);
    }
  }
  useEffect(() => {
    previous.current = new Map(spentByCamera);
    if (rows.length > 0) seeded.current = true;
  });

  const totalSlots = cameras.length * perCamera;
  const totalSpent = cameras.reduce((acc, c) => acc + (spentByCamera.get(c) ?? 0), 0);

  const target = benchmark?.budget.target ?? meta?.budget_target ?? null;
  const realised = benchmark?.budget.realised_uncapped ?? null;
  const tolerance = benchmark?.budget.drift_tolerance ?? meta?.drift_tolerance ?? null;

  // The drift pill is coloured only when the file publishes the tolerance that defines
  // drift. A green pill computed against a guessed band would be an assurance the
  // project has not earned.
  const drift = realised !== null && target !== null ? Math.abs(realised - target) : null;
  const driftClass =
    drift === null || tolerance === null
      ? "badge--spent"
      : drift <= tolerance
        ? "badge--ok"
        : "badge--alert";
  const driftText =
    drift === null || tolerance === null
      ? "drift tolerance not published"
      : drift <= tolerance
        ? "drift OK"
        : "drift alarm";

  return (
    <section className="card" id="budget" style={{ padding: "14px 16px" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
        <span className="label">alert budget</span>
        <span className="mono num" style={{ color: "var(--ink-0)", fontSize: "var(--size-lead)" }}>
          {totalSpent} / {totalSlots}
        </span>
        <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
          slots spent tonight, {perCamera} per camera (budget-capped)
        </span>
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 8,
          marginTop: 12,
        }}
      >
        {cameras.length === 0 ? (
          <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
            no camera roster yet -- the meter draws one pair of slots per camera the
            stream publishes, and never a placeholder pair.
          </span>
        ) : (
          cameras.map((cam) => {
            const spent = spentByCamera.get(cam) ?? 0;
            const firstNew = justFilled.get(cam);
            return (
              <span
                key={cam}
                title={`${cam}: ${spent} of ${perCamera} spent`}
                style={{ display: "flex", gap: 2 }}
              >
                {Array.from({ length: perCamera }, (_, i) => (
                  <Slot
                    key={i}
                    filled={i < spent}
                    justFilled={firstNew !== undefined && i >= firstNew && i < spent}
                  />
                ))}
              </span>
            );
          })
        )}
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginTop: 14,
          flexWrap: "wrap",
          fontSize: "var(--size-label)",
          color: "var(--ink-2)",
        }}
      >
        <span className="num">
          realised (uncapped) {rate(realised)} / target {target === null ? ABSENT : rate(target)}{" "}
          alerts per camera-night
        </span>
        <a href="/results#budget" className={`pill ${driftClass}`}>
          {driftText}
        </a>
      </div>
    </section>
  );
}

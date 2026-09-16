"use client";

import { minutes, rate, ABSENT } from "@/lib/format";
import type { Benchmark } from "@/lib/types";

/**
 * Budget (x) against median time-to-detection (y), one line per system.
 *
 * HAND-ROLLED INLINE SVG. There is no chart library in this repo, and that is a
 * decision with two halves. The visible half: about forty lines of path arithmetic
 * cost less than a dependency whose bundle a t4g.small has to serve to every judge.
 * The load-bearing half: this curve is the component that REPLACED the budget slider.
 * A dispatcher at 04:50 has no spare attention for a control, so the answer at every
 * budget is precomputed and drawn, and the one operating point the project is
 * calibrated to is the only annotated point on it.
 */

const W = 660;
const H = 280;
/* top padding leaves a clear band for the y-axis label, so it never lands on the
 * topmost gridline tick -- an axis label sitting on a number is a chart that has to be
 * decoded before it can be read. */
const PAD = { top: 34, right: 18, bottom: 40, left: 54 };

const STROKES = ["var(--signal)", "var(--ink-2)", "var(--fog)"];

export function OperatingCurve({ data }: { data: Benchmark }) {
  const series = data.operating.series;
  if (series.length === 0) {
    return (
      <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)" }}>
        operating curve not published in results/bench_&lt;date&gt;.json
      </p>
    );
  }

  const pts = series.flatMap((s) => s.points);
  const xs = pts.map((p) => p.budget);
  const ys = pts.map((p) => p.p50_minutes);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  // The y axis starts at zero, always. A truncated axis is how a two-minute difference
  // is made to look like a rout, and this chart's whole job is to be believed.
  const y1 = Math.max(...ys);

  const sx = (v: number) =>
    PAD.left + ((v - x0) / (x1 - x0 || 1)) * (W - PAD.left - PAD.right);
  const sy = (v: number) => H - PAD.bottom - (v / (y1 || 1)) * (H - PAD.top - PAD.bottom);

  const op = data.operating.operating_point;
  // An operating point outside the published budget range cannot be drawn as a rule on
  // this axis, and silently omitting the rule would leave a chart whose caption says it
  // has one annotated point and whose picture has none. The caption says which case it
  // is instead.
  const opInRange = op !== null && op >= x0 && op <= x1;

  return (
    <figure style={{ margin: 0 }}>
      <div className="scroll-x">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 440, display: "block" }} role="img"
             aria-label="median time to detection against alert budget, one line per system">
          <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom}
                stroke="var(--hairline)" />
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--hairline)" />

          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <text x={PAD.left - 8} y={sy(y1 * f) + 4} textAnchor="end"
                    fill="var(--ink-3)" fontSize="12" fontFamily="var(--font-mono)">
                {(y1 * f).toFixed(0)}
              </text>
              <line x1={PAD.left} y1={sy(y1 * f)} x2={W - PAD.right} y2={sy(y1 * f)}
                    stroke="var(--hairline)" strokeDasharray="2 5" />
            </g>
          ))}

          {opInRange && op !== null ? (
            <g>
              <line x1={sx(op)} y1={PAD.top} x2={sx(op)} y2={H - PAD.bottom}
                    stroke="var(--alert)" strokeWidth="1.5" />
              <text x={sx(op) + 6} y={PAD.top + 12} fill="var(--alert)" fontSize="12">
                operating point {rate(op, 1)} alerts / camera-night
              </text>
            </g>
          ) : null}

          {series.map((s, i) => (
            <polyline
              key={s.key}
              fill="none"
              stroke={STROKES[i % STROKES.length]}
              strokeWidth={s.key === "plumerank" ? 2.5 : 1.5}
              points={s.points.map((p) => `${sx(p.budget)},${sy(p.p50_minutes)}`).join(" ")}
            />
          ))}

          <text x={(W + PAD.left) / 2} y={H - 8} textAnchor="middle" fill="var(--ink-3)" fontSize="12">
            alerts per camera-night
          </text>
          <text x={8} y={16} fill="var(--ink-3)" fontSize="12">
            median TTD (min)
          </text>
        </svg>
      </div>
      <figcaption style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 6 }}>
        {series.map((s, i) => (
          <span key={s.key} style={{ fontSize: "var(--size-label)", color: "var(--ink-2)" }}>
            <span aria-hidden="true" style={{
              display: "inline-block", width: 14, height: 2, marginRight: 6, verticalAlign: "middle",
              background: STROKES[i % STROKES.length],
            }} />
            {s.label}
          </span>
        ))}
        {data.systems
          .filter((sys) => !series.some((s) => s.key === sys.key))
          .map((sys) => (
            <span key={sys.key} style={{ fontSize: "var(--size-label)", color: "var(--ink-3)" }}>
              {sys.label}: no median at any published budget
            </span>
          ))}
        <span style={{ fontSize: "var(--size-label)", color: "var(--ink-3)" }}>
          y axis starts at 0; peak {minutes(y1)} min
          {op === null
            ? ` - operating point ${ABSENT}`
            : opInRange
              ? ""
              : ` - operating point ${rate(op, 1)} lies outside the published budget range ` +
                `${rate(x0, 1)} to ${rate(x1, 1)} and is not marked`}
        </span>
      </figcaption>
    </figure>
  );
}

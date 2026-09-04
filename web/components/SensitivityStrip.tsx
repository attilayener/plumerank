"use client";

import { minutes } from "@/lib/format";
import type { Benchmark } from "@/lib/types";

/**
 * The +/-5 min t0 sensitivity curve, 11 offsets, as one wide sparkline with a shaded
 * band -- and it sits directly under the headline rather than in an appendix.
 *
 * Also hand-rolled SVG, for the same reason as the operating curve, plus one specific
 * to this chart: the shaded band is the whole content. The offset moves the GROUND
 * TRUTH clock, not the detector, so this strip answers the first question a sceptical
 * reader has about a time-to-detection headline -- "how much of that number is the
 * label?" -- and it answers it as a visible width. A library would draw a prettier line
 * and the same width.
 */

const W = 660;
const H = 132;
const PAD = { top: 14, right: 16, bottom: 28, left: 46 };

export function SensitivityStrip({ data }: { data: Benchmark }) {
  const pts = data.sensitivity;
  if (pts.length < 2) {
    return (
      <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)" }}>
        sensitivity curve not published in results/bench_&lt;date&gt;.json
      </p>
    );
  }

  const xs = pts.map((p) => p.offset_minutes);
  const ys = pts.map((p) => p.p50_minutes);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  // Padded by a tenth of the spread so a perfectly flat curve -- the good outcome --
  // still draws as a line rather than collapsing onto the axis and looking like no data.
  const pad = Math.max((hi - lo) * 0.35, 0.5);
  const yLo = Math.max(0, lo - pad);
  const yHi = hi + pad;

  const sx = (v: number) => PAD.left + ((v - x0) / (x1 - x0 || 1)) * (W - PAD.left - PAD.right);
  const sy = (v: number) =>
    H - PAD.bottom - ((v - yLo) / (yHi - yLo || 1)) * (H - PAD.top - PAD.bottom);

  const centre = pts.find((p) => p.offset_minutes === 0) ?? null;

  return (
    <figure style={{ margin: 0 }}>
      <div className="scroll-x">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 420, display: "block" }} role="img"
             aria-label="median time to detection across eleven ignition-clock offsets from minus five to plus five minutes">
          <rect x={PAD.left} y={sy(hi)} width={W - PAD.left - PAD.right} height={Math.max(1, sy(lo) - sy(hi))}
                fill="var(--signal)" opacity="0.14" />
          <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--hairline)" />
          <line x1={sx(0)} y1={PAD.top} x2={sx(0)} y2={H - PAD.bottom} stroke="var(--hairline)" />

          <polyline fill="none" stroke="var(--signal)" strokeWidth="2"
                    points={pts.map((p) => `${sx(p.offset_minutes)},${sy(p.p50_minutes)}`).join(" ")} />
          {pts.map((p) => (
            <circle key={p.offset_minutes} cx={sx(p.offset_minutes)} cy={sy(p.p50_minutes)} r="2.5"
                    fill="var(--signal)" />
          ))}

          <text x={PAD.left - 6} y={sy(hi) + 4} textAnchor="end" fill="var(--ink-3)" fontSize="11"
                fontFamily="var(--font-mono)">{hi.toFixed(1)}</text>
          <text x={PAD.left - 6} y={sy(lo) + 4} textAnchor="end" fill="var(--ink-3)" fontSize="11"
                fontFamily="var(--font-mono)">{lo.toFixed(1)}</text>
          <text x={sx(x0)} y={H - 9} fill="var(--ink-3)" fontSize="11">{x0} min</text>
          <text x={sx(0)} y={H - 9} textAnchor="middle" fill="var(--ink-3)" fontSize="11">published t0</text>
          <text x={sx(x1)} y={H - 9} textAnchor="end" fill="var(--ink-3)" fontSize="11">+{x1} min</text>
        </svg>
      </div>
      <figcaption style={{ marginTop: 6, fontSize: "var(--size-label)", color: "var(--ink-2)" }}>
        median TTD across {pts.length} ignition-clock offsets: {minutes(lo)} to {minutes(hi)} min
        {centre ? `, ${minutes(centre.p50_minutes)} min at the published t0` : ""}. The offset moves
        the ground-truth clock, not the detector -- the band is how much of the headline is the label.
      </figcaption>
    </figure>
  );
}

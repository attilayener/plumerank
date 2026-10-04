"use client";

import type { Benchmark } from "@/lib/types";

/**
 * {MOG2, LSBP, GSOC} x {S4 on/off} x {escalation on/off} on the DEV arm, as a plain
 * table.
 *
 * A plain table on purpose: this is what the freed hours bought, it sits inside the
 * criterion that carries 30% and breaks ties, and a reader comparing twelve cells wants
 * to read twelve numbers rather than decode twelve bars. The columns are whatever the
 * results file publishes -- the frontend does not know the grid's shape and has no
 * business asserting one.
 */
export function AblationTable({ data }: { data: Benchmark }) {
  const grid = data.ablation;
  if (!grid || grid.rows.length === 0) {
    return (
      <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)" }}>
        ablation grid not published yet — it is a separate twelve-cell run on the
        development set, quoted here by the held-out file once both exist
      </p>
    );
  }

  return (
    <div className="scroll-x card">
      <table>
        <thead>
          <tr>
            {grid.columns.map((c) => (
              <th key={c}>{c.replace(/_/g, " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {grid.rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j} className={j === 0 ? undefined : "mono"} style={j === 0 ? undefined : { fontWeight: 400 }}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

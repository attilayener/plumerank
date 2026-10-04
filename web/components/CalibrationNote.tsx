import type { CalibrationKind, WatchMeta } from "@/lib/types";

/** Which committed run the wall's threshold came from, read off the stream. */
export function calibrationKindOf(meta: WatchMeta | null): CalibrationKind {
  return meta?.calibration_kind ?? meta?.live_pipeline?.calibration_kind ?? null;
}

/**
 * Says, wherever the wall or its budget is drawn, where the alert threshold came from.
 *
 * Before the held-out run exists the wall runs on the threshold the development run
 * solved on the fire-free confusion set. That threshold is real and is the one the
 * held-out run will solve on the same nights -- but a judge must never have to infer
 * which run they are looking at, so it is stated in words, not implied.
 */
export function CalibrationNote({ kind }: { kind: CalibrationKind }) {
  if (kind === null) return null;
  return (
    <p
      className="calibration-note"
      style={{ color: kind === "development" ? "var(--fog)" : "var(--ink-2)", fontSize: "var(--size-label)", margin: "6px 0 0" }}
    >
      {kind === "development"
        ? "Threshold solved on the fire-free confusion set (development run). The held-out result is pending."
        : "Threshold solved on the fire-free confusion set by the held-out run."}
    </p>
  );
}

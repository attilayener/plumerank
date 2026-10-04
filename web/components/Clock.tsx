"use client";

import { signedClock, wallClock, ABSENT } from "@/lib/format";
import type { WatchMeta } from "@/lib/types";

/**
 * The replay's simulated wall time, and -- underneath it, in --ink-3 -- how long it has
 * been since ignition.
 *
 * The second line is the honesty element. `t0_index` is ground truth: it comes from
 * FIgLib's record of when the fire started, metrics.py is the only module that ever
 * sees it, and the pipeline never does. Putting it on screen with the label
 * "ground truth -- not visible to the system" is what stops a viewer from reading the
 * countdown as something the detector knows. The alternative -- not showing it -- would
 * make the wall unreadable, because "seven minutes after ignition" is the entire claim.
 */
export function Clock({
  meta,
  frameIndex,
}: {
  meta: WatchMeta | null;
  frameIndex: number | null;
}) {
  const spf = meta?.seconds_per_frame ?? null;
  const t0Index = meta?.t0_index ?? null;
  const startEpoch =
    typeof (meta as { start_epoch?: number } | null)?.start_epoch === "number"
      ? (meta as unknown as { start_epoch: number }).start_epoch
      : null;
  const ignitionEpoch = typeof meta?.t0 === "number" ? meta.t0 : null;

  const sinceIgnitionSeconds =
    spf !== null && t0Index !== null && frameIndex !== null
      ? (frameIndex - t0Index) * spf
      : null;

  // Two accepted spellings for the replay's clock origin, because either is a natural
  // thing for api/main.py to publish and neither is guessed at: `start_epoch` is the
  // first frame, `t0` is the ignition frame. When only `t0` is present the wall time is
  // reconstructed backwards through the ignition offset rather than assumed to be it.
  const wallEpoch =
    startEpoch !== null && spf !== null && frameIndex !== null
      ? startEpoch + frameIndex * spf
      : ignitionEpoch !== null && sinceIgnitionSeconds !== null
        ? ignitionEpoch + sinceIgnitionSeconds
        : null;

  // The ignition offset is promoted to the 46 px slot whenever the stream publishes no
  // wall-clock origin, which today is always: api/main.py's `meta` carries the corpus
  // cadence and the ignition frame but no epoch, because FIgLib's clock belongs to the
  // imagery rather than to the replay. Rendering "--:--:--" at 46 px would read as a
  // broken component; rendering the offset there reads as the thing the wall is
  // actually about. The moment a wall-clock origin IS published the two swap back.
  const leadIsOffset = wallEpoch === null;

  // Nothing is replaying (no tick yet, or the pipeline is unavailable): say that in words
  // rather than drawing a clock face of dashes, which reads as a rendering bug.
  if (frameIndex === null) {
    return (
      <div className="card" style={{ padding: "14px 16px" }}>
        <div className="label">since ignition</div>
        <p style={{ color: "var(--ink-2)", margin: "6px 0 0" }}>
          the clock starts with the first replayed frame.
        </p>
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: "14px 16px" }}>
      <div className="label">{leadIsOffset ? "since ignition" : "simulated wall time"}</div>
      <div
        className="mono"
        style={{ fontSize: "var(--size-metric-m)", color: "var(--ink-0)", lineHeight: 1.1 }}
      >
        {leadIsOffset
          ? sinceIgnitionSeconds === null
            ? `T+${ABSENT}:${ABSENT}`
            : signedClock(sinceIgnitionSeconds)
          : wallClock(wallEpoch)}
      </div>
      {leadIsOffset ? null : (
        <div className="mono" style={{ fontSize: "var(--size-body)", color: "var(--ink-3)", marginTop: 6 }}>
          {sinceIgnitionSeconds === null
            ? `T+${ABSENT}:${ABSENT}`
            : signedClock(sinceIgnitionSeconds)}{" "}
          since ignition
        </div>
      )}
      <div className="label" style={{ marginTop: 2, textTransform: "none", letterSpacing: 0 }}>
        ground truth -- not visible to the system
      </div>
      <div className="label" style={{ marginTop: 8, textTransform: "none", letterSpacing: 0 }}>
        frame {frameIndex === null ? ABSENT : frameIndex}
        {meta?.frame_count ? ` of ${meta.frame_count}` : ""}
        {leadIsOffset ? " - replay time, not wall-clock time" : ""}
      </div>
    </div>
  );
}

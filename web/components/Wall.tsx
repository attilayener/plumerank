"use client";

import { apiUrl } from "@/lib/api";
import type { StreamState, WatchMeta, WatchRow } from "@/lib/types";

/**
 * The full camera roster, 8 x 5, and it never gets shorter.
 *
 * Low-ranked tiles are dimmed to 45%. They are NOT removed, and that is the whole
 * argument the wall exists to make: PlumeRank reorders attention, it does not suppress
 * a camera. A wall that dropped the quiet 36 would be showing a different system from
 * the one that runs, and the one safeguard a fire agency will ask about first is
 * exactly the one that removal would visually contradict.
 *
 * Below 1280 px the same tiles become a horizontally scrollable strip of the top 12;
 * below 768 px the wall is hidden by one CSS rule and the watch list IS the product.
 */

/** How many tiles the sub-1280 strip carries. A layout number, not a measurement. */
const STRIP_TILES = 12;

export function Wall({
  meta,
  rows,
  topK,
  stream,
  warm,
  framesSeen,
  compact,
  holdSequences,
}: {
  meta: WatchMeta | null;
  rows: WatchRow[];
  topK: number | null;
  stream: StreamState;
  warm: boolean | null;
  framesSeen: number | null;
  /** true for the < 1280 px strip. Set by the page, not sniffed here: the page owns
   *  the layout and a component that measured the viewport itself would be a second
   *  source of truth for the same breakpoint. */
  compact: boolean;
  /** How many held-out sequences the published headline was measured on, straight from
   *  /api/benchmark. The provenance caption names it rather than carrying a literal:
   *  the wall's whole job here is to say the demo is NOT the measurement, and a number
   *  typed into the caption could disagree with the one on /results. */
  holdSequences: number | null;
}) {
  const byCamera = new Map(rows.map((r) => [r.camera_id, r]));
  const roster = meta?.cameras ?? rows.map((r) => r.camera_id);

  const ordered = compact
    ? [...roster]
        .sort((a, b) => (byCamera.get(a)?.rank ?? 1e9) - (byCamera.get(b)?.rank ?? 1e9))
        .slice(0, STRIP_TILES)
    : roster;

  const warmupFrames = meta?.warmup_frames ?? null;
  // framesSeen and NOT frameIndex. The replay wraps every window while the background
  // models stay warm across the wrap, so a warm-up read off the cyclic frame index
  // re-announces "warming background model" on every lap -- a caption about forty
  // models that have been warm for an hour. useWatchlist publishes null rather than a
  // count whenever it cannot honestly claim the loop started while this client watched.
  const seen = framesSeen;
  const warming =
    warm === false ||
    (warm === null && warmupFrames !== null && seen !== null && seen < warmupFrames);
  const warmProgress =
    warmupFrames !== null && warmupFrames > 0 && seen !== null
      ? Math.min(1, Math.max(0, seen / warmupFrames))
      : 0;

  // The three ways the wall can have nothing to draw are three DIFFERENT statements and
  // the page is not allowed to collapse them: the meta event has not arrived yet; it
  // arrived carrying api/main.py's own "live pipeline unavailable" flag; or the loop is
  // simply between ticks. Only the last one is "replay paused".
  const live = meta?.live_pipeline ?? null;
  const pipelineDown = live !== null && live.available === false;
  const paused = stream === "idle" && !pipelineDown;

  return (
    <section>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 10,
          marginBottom: 8,
          flexWrap: "wrap",
        }}
      >
        <span className="label">the wall</span>
        <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
          {roster.length} cameras, all of them, always
        </span>
      </div>

      {roster.length === 0 ? (
        <p style={{ color: "var(--ink-2)", margin: 0 }}>
          {pipelineDown
            ? live?.detail ??
              "live pipeline unavailable -- the stream published an empty roster and said why."
            : meta === null
              ? "no camera roster: the `meta` event of /api/watchlist has not arrived."
              : "the `meta` event arrived carrying no cameras."}{" "}
          No placeholder tiles are drawn -- a wall of invented cameras would be the one
          thing on this screen a judge could not tell from a real one.
        </p>
      ) : (
        <div
          className={compact ? "wall-strip" : "wall-grid"}
          style={paused ? { opacity: 0.45 } : undefined}
        >
          {ordered.map((cam) => {
            const row = byCamera.get(cam);
            const src = row?.frame_url ?? row?.evidence_url ?? null;
            // topK null means the stream did not publish the cutoff, and the wall then
            // dims NOTHING rather than dimming by a guessed one. See the note below.
            const dim = topK !== null && (row === undefined || row.rank > topK);
            return (
              <div key={cam} className={`tile${dim ? " tile--dim" : ""}`}>
                {src ? (
                  <img src={apiUrl(src)} alt="" />
                ) : (
                  <span className="tile-empty">no frame published</span>
                )}
                {row ? <span className="tile-rank">{row.rank}</span> : null}
                <span className="tile-name">{cam}</span>
                {warming ? (
                  <span className="tile-warm" style={{ width: `${warmProgress * 100}%` }} />
                ) : null}
              </div>
            );
          })}
        </div>
      )}

      {/* No second degraded box here: the empty-roster paragraph above already carries
          live_pipeline.detail, and the page renders the designed state-7 strip. Saying
          the same sentence three times is how a designed state turns into noise. */}
      {paused ? (
        <p style={{ color: "var(--ink-2)", fontSize: "var(--size-label)", marginTop: 8 }}>
          replay paused -- no viewers. The loop is on-demand: it starts on the first SSE
          connection and stops about two minutes after the last one drops, because a
          burstable t4g.small that ran the wall continuously would throttle the endpoint
          mid-judging. It restarts on the next connection.
        </p>
      ) : null}

      {warming ? (
        <p style={{ color: "var(--ink-2)", fontSize: "var(--size-label)", marginTop: 8 }}>
          warming background model -- this is what a real camera start looks like
          {warmupFrames !== null && seen !== null ? ` (${seen} of ${warmupFrames} frames)` : ""}
        </p>
      ) : null}

      {topK === null && roster.length > 0 ? (
        <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)", marginTop: 8 }}>
          top-k was not published by the stream, so no tile is dimmed. The wall dims by
          the pipeline&apos;s own cutoff or not at all -- dimming by a guessed one would
          be the frontend drawing a mechanism the server never described.
        </p>
      ) : null}

      {/* Provenance. The wall runs the DEV arm and says so, unprompted, right under the
          thing it is showing -- because the strongest attack on this project is
          "you demoed on the fire you tuned on", and answering it before it is asked is
          cheaper than answering it afterwards. The wall never reads HOLD. */}
      <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)", marginTop: 10 }}>
        demo sequence: DEV arm -- constants were fit on this fire. The headline is
        measured on{" "}
        {holdSequences === null ? "the" : `${holdSequences}`} HOLD sequences the system
        has never seen.
      </p>
    </section>
  );
}

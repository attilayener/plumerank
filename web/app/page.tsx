"use client";

import Link from "next/link";
import { useState } from "react";
import { BudgetMeter } from "@/components/BudgetMeter";
import { Clock } from "@/components/Clock";
import { DegradedStrip } from "@/components/DegradedStrip";
import { EvidenceCard } from "@/components/EvidenceCard";
import { ThemeNote } from "@/components/ThemeNote";
import { TopBar } from "@/components/TopBar";
import { Wall } from "@/components/Wall";
import { WatchList } from "@/components/WatchList";
import { minutes, ABSENT } from "@/lib/format";
import { useBenchmark } from "@/lib/useBenchmark";
import { useMediaQuery } from "@/lib/useMediaQuery";
import { useWatchlist } from "@/lib/useWatchlist";
import type { WatchRow } from "@/lib/types";

/**
 * "/" -- The Wall. The thirty-second demo.
 *
 * Everything on this screen is live: the roster, the ranks, the crops, the budget
 * slots. There is no pre-rendered ranking anywhere on the judged path -- the replay
 * loop starts when you connect and the background model warms from the pre-ignition
 * frames FIgLib supplies, on screen, which is exactly what a real deployment does at
 * camera start.
 */

/** Slots per camera when the stream does not publish its own. The operating design is
 *  2 (constants.ALERTS_PER_CAMERA_PER_NIGHT); it decides a badge, never a measurement. */
const DEFAULT_SLOTS_PER_CAMERA = 2;

export default function WallPage() {
  const { meta, rows, frameIndex, warm, framesSeen, state, stale, error } = useWatchlist();
  const { data: benchmark } = useBenchmark();
  const [selected, setSelected] = useState<WatchRow | null>(null);

  const compact = useMediaQuery("(max-width: 1279px)");
  const topK = meta?.top_k ?? null;
  const perCamera = meta?.budget_per_camera ?? DEFAULT_SLOTS_PER_CAMERA;
  const threshold = typeof meta?.threshold === "number" ? meta.threshold : null;
  // api/main.py publishes its degraded flag ON the meta event, not only on /health, and
  // the stream stays open when it fires. Reading it is the only way this page can tell
  // an outage apart from an idle loop -- without it the wall captions a dead pipeline
  // "replay paused -- no viewers" while a viewer is looking at it.
  const pipelineDown = meta?.live_pipeline?.available === false;
  // The provenance caption names the held-out sequence count only when the published
  // run really is the held-out one. bench.py can write a dev-arm file (--arm dev), and
  // captioning the wall "measured on N HOLD sequences" off a DEV run's count would be
  // the exact confusion the caption exists to prevent.
  const holdSequences =
    benchmark !== null && (benchmark.run.arm === null || benchmark.run.arm === "hold")
      ? benchmark.run.n_sequences
      : null;

  return (
    <>
      <TopBar stream={state} />

      <main style={{ padding: "20px 16px 0" }}>
        {/* The landing band is deliberately SHORT. This route is the thirty-second
            demo, so the wall and the rank digit have to be on the first screen at
            1280 px: a hook that pushed them under the fold would be a landing page
            wearing the product as a footer. The two long-form notes -- the declined
            awards and the one-theme refusal -- are real content and they are kept, at
            the bottom, where a reader who wants them will look. */}
        <section className="landing">
          <div>
            <h1 className="display">
              Forty cameras. Two alerts each. Which one do you look at?
            </h1>
            <p style={{ color: "var(--ink-1)", margin: 0 }}>
              PlumeRank ranks a camera network by an anchored plume-growth statistic that
              OpenCV does not ship -- a plume is tied to the ground while its top climbs,
              and its boundary is soft where a cloud shadow&apos;s is hard. The ranking runs
              live below, under the same hard alert budget a dispatcher actually lives with.
            </p>
          </div>

          <div>
            <p style={{ margin: "0 0 8px" }}>
              <span className="mono num" style={{ fontSize: "var(--size-metric-s)", color: "var(--ink-0)" }}>
                {minutes(benchmark?.headline.p50_minutes ?? null)}
              </span>{" "}
              <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
                min median time-to-detection at{" "}
                {benchmark === null || benchmark.budget.target === null
                  ? ABSENT
                  : benchmark.budget.target.toFixed(1)}{" "}
                alerts per camera-night, on{" "}
                {benchmark === null || benchmark.run.n_sequences === null
                  ? ABSENT
                  : benchmark.run.n_sequences}{" "}
                held-out sequences
              </span>
            </p>
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <code className="code">make verify</code>
              <span className="label" style={{ textTransform: "none", letterSpacing: 0, flex: "1 1 200px" }}>
                about 3 min -- recomputes the micro-corpus metrics. It does not reproduce
                the headline; <Link href="/results#reproduce">the 95-minute pair</Link> does.
              </span>
            </div>
            <p style={{ margin: "8px 0 0", fontSize: "var(--size-label)" }}>
              <Link href="/results">the full results, the operating curve and the build string</Link>
            </p>
          </div>
        </section>

        {state === "down" || pipelineDown ? (
          <div style={{ marginTop: 16 }}>
            <DegradedStrip
              commit={benchmark?.run.commit ?? null}
              detail={
                pipelineDown
                  ? (meta?.live_pipeline?.detail ??
                      "the stream reports live_pipeline.available = false") +
                    ". The stream is connected and the roster is empty because the box " +
                    "cannot run the pipeline, not because nobody is watching."
                  : (error ?? "the live stream is not delivering") +
                    (stale
                      ? ". The rows below are the last tick received and are NOT current."
                      : ". No rows are being drawn.")
              }
            />
          </div>
        ) : null}

        <div className="wall-page" style={{ padding: "16px 0 0" }}>
          <div className="area-wall">
            <Wall
              meta={meta}
                rows={rows}
                topK={topK}
                stream={state}
              warm={warm}
              framesSeen={framesSeen}
              compact={compact}
              holdSequences={holdSequences}
            />
          </div>

          <div className="area-rail rail">
            <div className="rail-clock">
              <Clock meta={meta} frameIndex={frameIndex} />
            </div>
            <div className="rail-list">
              <WatchList
                rows={rows}
                topK={topK}
                perCamera={perCamera}
                threshold={threshold}
                stream={state}
                onSelect={setSelected}
              />
            </div>
          </div>

          <div className="area-budget">
            <BudgetMeter meta={meta} rows={rows} benchmark={benchmark} />
          </div>
        </div>

        <footer style={{ marginTop: 22, display: "grid", gap: 10, maxWidth: 900 }}>
          <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)", margin: 0 }}>
            Two special awards are deliberately declined and neither box is ticked. Best
            Use of COOL: this hot loop is background modelling and optical flow, not
            resize / adaptive threshold / contours, so a COOL lane would benchmark flat
            against that award&apos;s own measured-performance criterion. Agentic Vision:
            there is no orchestration loop, because the dispatcher being human-in-command
            IS the safety argument.
          </p>
          <ThemeNote />
          <p style={{ color: "var(--ink-3)", fontSize: "var(--size-label)", margin: 0 }}>
            {meta?.credit ?? "Imagery: HPWREN, http://hpwren.ucsd.edu/"}
          </p>
        </footer>
      </main>

      {selected ? <EvidenceCard row={selected} onClose={() => setSelected(null)} /> : null}
    </>
  );
}

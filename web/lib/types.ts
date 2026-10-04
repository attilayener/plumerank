/**
 * The wire shapes, written down once.
 *
 * Two of these are contracts with api/main.py (W5) and one is a contract with
 * scripts/bench.py (W6). All three are read defensively -- see lib/read.ts -- for a
 * reason that is worth stating rather than apologising for: the frontend is not
 * allowed to invent a number, so the only two honest reactions to a field it did not
 * receive are "render the designed empty state" and "render the designed degraded
 * state". Neither of those needs the field to be typed as present, and typing it as
 * present would push the failure to runtime in the one place the product is judged.
 */

/* --------------------------------------------------------------------------------
 * GET /api/watchlist -- SSE. Event `meta` once, then event `tick` per replay minute.
 * ----------------------------------------------------------------------------- */

/** One row of `WatchEntry.to_json()` (plumerank/types.py). */
export interface WatchRow {
  rank: number;
  camera_id: string;
  hazard: number;
  s1: number;
  s2: number;
  s3: number;
  s4: number;
  escalated: boolean;
  alerted: boolean;
  /** Alerts spent on this camera so far tonight, 0..2. Drawn as countable slots. */
  budget_spent: number;
  /** Server-rendered crop with the anchoring overlay burnt in by rank.draw_evidence.
   *  The browser does no geometry; it lays out and captions an image. */
  evidence_url: string | null;

  /** Optional, and the wall degrades gracefully without them.
   *  `frame_url`: the live tile image for this camera, if the server publishes one.
   *  `confusion_class` / `fog`: the CONFUSE label. Fog-class rows render in --fog with
   *  no colour of their own -- the frontend NEVER derives this from S3, because
   *  deriving a class from a statistic is exactly the answer the gate exists to
   *  measure rather than assume. */
  frame_url?: string | null;
  confusion_class?: string | null;
  fog?: boolean;
  county?: string | null;
}

/** The `meta` event: everything constant for one replay. */
export interface WatchMeta {
  replay_id?: string;
  /** The camera roster. The wall draws EVERY one of these, always. */
  cameras?: string[];
  credit?: string;
  frame_count?: number;
  /** Ignition frame index. Ground truth -- shown, labelled as invisible to the system. */
  t0_index?: number | null;
  /** Replay start as an epoch second, if the server simulates a wall clock. */
  t0?: number | null;
  seconds_per_frame?: number;
  /** Configured replay pace, seconds per tick (one frame = one minute of footage). */
  tick_seconds?: number;
  /** Measured pace (EWMA of tick starts), when the server has one. Preferred for copy. */
  tick_seconds_measured?: number | null;
  /** Frames the background model needs before its output means anything. Drives the
   *  warm-up state; absent means the wall does not claim to know. */
  warmup_frames?: number;
  budget_target?: number;
  budget_per_camera?: number;
  threshold?: number;
  threshold_hash?: string;
  /** api/main.py's degraded flag, published ON the meta event and not only on /health.
   *  `available: false` is the designed "live pipeline unavailable" state: the stream
   *  stays open, the roster is empty and no tick ever arrives. Without reading this the
   *  page cannot tell that case apart from an idle replay, and it would caption an
   *  outage "replay paused -- no viewers" while a viewer was looking at it. */
  live_pipeline?: {
    available?: boolean;
    detail?: string | null;
    cameras?: number;
    threshold?: number | null;
    threshold_source?: string | null;
    calibration_kind?: CalibrationKind;
  } | null;
  /** Which committed run the wall's threshold came from -- see CalibrationKind. */
  calibration_kind?: CalibrationKind;
  threshold_source?: string | null;
  /** The watch list a human actually reads (plumerank/constants.py TOP_K). The wall
   *  dims every tile outside it and NEVER guesses the cutoff -- see Wall.tsx. */
  top_k?: number;
  drift_tolerance?: number;
  /** One record per camera on the wall, straight from the manifests. `arm` is which
   *  corpus the window came from; `confuse` windows are fire-free confusion-set nights,
   *  and that manifest label -- never a statistic -- is what styles a row as --fog. */
  sequences?: { camera_id: string; sequence_id?: string; arm?: string }[];
}

/** `held-out`: the threshold in results/bench_<date>.json. `development`: before that
 *  run exists, the threshold in the newest shipped results/arm_dev_<date>.json -- solved
 *  on the same fire-free confusion set by the same code. `null`: no threshold at all. */
export type CalibrationKind = "held-out" | "development" | null;

/** The `tick` event. Either a bare array of rows or an object carrying them. */
export interface WatchTick {
  frame_index?: number;
  minute?: number;
  warm?: boolean;
  frames_seen?: number;
  rows: WatchRow[];
}

/** `connected`: the `meta` event arrived and the first tick has not yet.
 *  `unavailable`: `meta` arrived carrying live_pipeline.available = false -- the stream
 *  is open and healthy, the box simply cannot run the pipeline, and no tick will come. */
export type StreamState = "connecting" | "connected" | "live" | "idle" | "down" | "unavailable";

/* --------------------------------------------------------------------------------
 * GET /api/benchmark -- results/bench_<date>.json, verbatim from disk.
 * ----------------------------------------------------------------------------- */

export interface SystemResult {
  key: string;
  label: string;
  p50_minutes: number | null;
  p95_minutes: number | null;
  /** How many of the arm's fires this system detected at all. A lane with a null p50 and
   *  a detection count is a system that MISSED MORE THAN HALF -- a result, not an absence.
   *  On the real DEV arm both baselines are in that state; only PlumeRank has a median. */
  n_detected: number | null;
  n: number | null;
  n_missed: number | null;
  alerts_per_camera_night: number | null;
  /** The rate realised on the arm itself (uncapped). On a fire arm this is legitimately
   *  above the calibration rate -- a fire SHOULD alert -- and it is published beside it,
   *  labelled, never in its place. */
  arm_alerts_per_camera_night: number | null;
  model_invocations_per_camera_night: number | null;
}

export interface CurvePoint {
  budget: number;
  p50_minutes: number;
}

export interface CurveSeries {
  key: string;
  label: string;
  points: CurvePoint[];
  /** Budgets the file publishes at which this system has no median (fewer than half
   *  the fires detected). Drawn as "no median" ticks rather than silently skipped. */
  missing: number[];
}

export interface SensitivityPoint {
  offset_minutes: number;
  /** Null at an offset where fewer than half the fires were detected, so the median is
   *  off the clock. Kept as an explicit gap rather than dropped from the strip. */
  p50_minutes: number | null;
  p95_minutes: number | null;
}

export interface RunMeta {
  commit: string | null;
  instance: string | null;
  date: string | null;
  wall_clock_seconds: number | null;
  arm: string | null;
  n_sequences: number | null;
  constants_sha256: string | null;
}

export interface AblationGrid {
  columns: string[];
  rows: string[][];
}

export interface Exclusions {
  count: number | null;
  contact_sheet_url: string | null;
  items: { sequence_id: string; reason: string }[];
}

/** The shape lib/read.ts normalises /api/benchmark into. Every field is nullable and
 *  every null has a designed rendering. */
export interface Benchmark {
  run: RunMeta;
  headline: { p50_minutes: number | null; p95_minutes: number | null; n_missed: number | null };
  budget: {
    target: number | null;
    realised_uncapped: number | null;
    threshold: number | null;
    drift_tolerance: number | null;
  };
  systems: SystemResult[];
  /** B1 and B2 through PlumeRank's own per-camera baseline (bench.py baselines_rebased):
   *  the like-for-like comparison. Empty when the file predates them. */
  rebased: SystemResult[];
  /** `budgets` is every budget the file publishes, so the x axis can tick all of them. */
  operating: { series: CurveSeries[]; operating_point: number | null; budgets: number[] };
  sensitivity: SensitivityPoint[];
  ablation: AblationGrid | null;
  escalation: { fraction: number | null; max: number | null };
  classifier: { precision: number | null; recall: number | null; dataset: string | null };
  exclusions: Exclusions;
  /** The parsed document, so BuildInfo-style verbatim inspection stays possible. */
  raw: Record<string, unknown>;
}

/* --------------------------------------------------------------------------------
 * GET /build-info
 * ----------------------------------------------------------------------------- */

/** GET /build-info, in the shape api/main.py's build_info() returns -- nested, not flat.
 *  An earlier flat spelling (opencv_version, dnn_engine, modules) matched no field the
 *  server has ever sent, so the section rendered ABSENT for every value on the live box. */
export interface BuildInfo {
  opencv?: {
    version?: string;
    modules?: Record<string, { ok?: boolean; symbols?: Record<string, boolean> }>;
    build_information?: string;
  };
  dnn?: {
    engine_requested?: string | null;
    engine_selected?: string | null;
    loaded?: boolean;
    detail?: string | null;
  };
  constants?: { sha256?: string | null; committed_sha256?: string | null; matches?: boolean | null };
  instance_id?: string | null;
  ami_id?: string | null;
  commit?: string | null;
  onnx_sha256?: string | null;
  [key: string]: unknown;
}

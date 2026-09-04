/**
 * Reading /api/benchmark without inventing anything.
 *
 * The endpoint serves results/bench_<date>.json VERBATIM from disk, which is what
 * makes the page, the endpoint and the repo unable to disagree about the headline. The
 * price of "verbatim" is that the frontend does not get to specify the file's exact
 * key names -- scripts/bench.py owns those -- so this module looks each quantity up
 * under the handful of spellings the file could plausibly use and returns `null` when
 * it finds none.
 *
 * `null` is a load-bearing value here, not a shrug. Every consumer of a null renders
 * "not published in results/bench_<date>.json" in --ink-3. That is a worse-looking
 * page than a zero and an infinitely better one: a zero p50 would read as "detects
 * instantly", which is the single most attackable wrong number this project could put
 * on a screen.
 */

import type {
  AblationGrid,
  Benchmark,
  CurveSeries,
  Exclusions,
  SensitivityPoint,
  SystemResult,
} from "./types";

type Dict = Record<string, unknown>;

function isDict(v: unknown): v is Dict {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Dotted lookup: `at(doc, "budget.target")`. */
function at(doc: unknown, path: string): unknown {
  let cur: unknown = doc;
  for (const key of path.split(".")) {
    if (!isDict(cur)) return undefined;
    cur = cur[key];
  }
  return cur;
}

/** First path that yields a finite number. NaN and Infinity are rejected on purpose:
 *  metrics.py encodes a missed sequence as +inf, and an infinity rendered as a p50
 *  would be read as a formatting bug rather than as the honest answer it is. The miss
 *  count is published beside every percentile instead. */
export function num(doc: unknown, ...paths: string[]): number | null {
  for (const p of paths) {
    const v = at(doc, p);
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) {
      return Number(v);
    }
  }
  return null;
}

export function str(doc: unknown, ...paths: string[]): string | null {
  for (const p of paths) {
    const v = at(doc, p);
    if (typeof v === "string" && v.trim() !== "") return v;
  }
  return null;
}

function list(doc: unknown, ...paths: string[]): unknown[] | null {
  for (const p of paths) {
    const v = at(doc, p);
    if (Array.isArray(v)) return v;
  }
  return null;
}

/* -------------------------------------------------------------------------------- */

const SYSTEM_LABELS: Record<string, string> = {
  plumerank: "PlumeRank",
  b1: "B1 -- the same CNN on every frame",
  b2: "B2 -- frame differencing",
};

function readSystems(doc: unknown): SystemResult[] {
  // Two shapes are accepted because both are natural for bench.py to write: a list of
  // per-system records, or an object keyed by system. Neither is guessed at -- a
  // document carrying neither yields an empty list and the table says so.
  const raw = list(doc, "systems", "arms", "results");
  const entries: [string, unknown][] = raw
    ? raw.map((r) => [str(r, "key", "system", "name") ?? "", r])
    : Object.entries((at(doc, "systems") as Dict) ?? {});

  return entries
    .filter(([key]) => key !== "")
    .map(([key, r]) => ({
      key,
      label: str(r, "label") ?? SYSTEM_LABELS[key] ?? key,
      p50_minutes: num(r, "p50_minutes", "p50", "ttd.p50_minutes", "ttd_p50_minutes"),
      p95_minutes: num(r, "p95_minutes", "p95", "ttd.p95_minutes", "ttd_p95_minutes"),
      n_missed: num(r, "n_missed", "missed", "ttd.n_missed"),
      alerts_per_camera_night: num(
        r,
        "alerts_per_camera_night",
        "realised_alerts_per_camera_night",
        "realised_rate",
      ),
      model_invocations_per_camera_night: num(
        r,
        "model_invocations_per_camera_night",
        "invocations_per_camera_night",
        "model_invocations",
      ),
    }));
}

function readOperating(doc: unknown): CurveSeries[] {
  const raw = list(doc, "operating_curve.series", "operating_curve", "operating.series");
  if (!raw) return [];
  return raw
    .map((s) => {
      const key = str(s, "key", "system", "name") ?? "";
      const pts = list(s, "points", "curve") ?? [];
      return {
        key,
        label: str(s, "label") ?? SYSTEM_LABELS[key] ?? key,
        points: pts
          .map((p) => ({
            budget: num(p, "budget", "alerts_per_camera_night", "x"),
            p50_minutes: num(p, "p50_minutes", "p50", "y"),
          }))
          .filter((p): p is { budget: number; p50_minutes: number } =>
            p.budget !== null && p.p50_minutes !== null,
          )
          .sort((a, b) => a.budget - b.budget),
      };
    })
    .filter((s) => s.key !== "" && s.points.length > 1);
}

function readSensitivity(doc: unknown): SensitivityPoint[] {
  const raw = list(doc, "sensitivity.points", "sensitivity", "sensitivity_curve");
  if (!raw) return [];
  return raw
    .map((p) => ({
      offset_minutes: num(p, "offset_minutes", "offset", "t0_offset_minutes", "x"),
      p50_minutes: num(p, "p50_minutes", "p50", "y"),
      p95_minutes: num(p, "p95_minutes", "p95"),
    }))
    .filter((p): p is SensitivityPoint => p.offset_minutes !== null && p.p50_minutes !== null)
    .sort((a, b) => a.offset_minutes - b.offset_minutes);
}

/** The ablation grid, rendered as whatever table the CSV-turned-JSON actually is.
 *  Columns are derived from the union of the row keys rather than declared here: the
 *  grid is {MOG2, LSBP, GSOC} x {S4 on/off} x {escalation on/off} today and the point
 *  of shipping it is that a judge reads the cells, not that the frontend knows their
 *  names. */
function readAblation(doc: unknown): AblationGrid | null {
  const declaredCols = list(doc, "ablation.columns");
  const declaredRows = list(doc, "ablation.rows");
  if (declaredCols && declaredRows) {
    return {
      columns: declaredCols.map((c) => String(c)),
      rows: declaredRows.map((r) => (Array.isArray(r) ? r.map((c) => String(c)) : [String(r)])),
    };
  }

  const records = list(doc, "ablation.cells", "ablation", "ablation_grid");
  if (!records || records.length === 0) return null;

  const columns: string[] = [];
  for (const rec of records) {
    if (!isDict(rec)) continue;
    for (const k of Object.keys(rec)) if (!columns.includes(k)) columns.push(k);
  }
  if (columns.length === 0) return null;

  const rows = records.map((rec) =>
    columns.map((c) => {
      const v = isDict(rec) ? rec[c] : undefined;
      if (v === undefined || v === null) return "--";
      return typeof v === "number" ? String(v) : String(v);
    }),
  );
  return { columns, rows };
}

function readExclusions(doc: unknown): Exclusions {
  const items = list(doc, "exclusions.items", "exclusions.sequences", "excluded") ?? [];
  return {
    count: num(doc, "exclusions.count", "exclusions.n", "n_excluded"),
    contact_sheet_url: str(doc, "exclusions.contact_sheet_url", "exclusions.contact_sheet"),
    items: items
      .map((i) => ({
        sequence_id: str(i, "sequence_id", "id") ?? "",
        reason: str(i, "reason", "why") ?? "",
      }))
      .filter((i) => i.sequence_id !== ""),
  };
}

/* -------------------------------------------------------------------------------- */

export function readBenchmark(doc: unknown): Benchmark {
  return {
    run: {
      commit: str(doc, "run.commit", "commit", "git_commit"),
      instance: str(doc, "run.instance", "instance", "instance_type"),
      date: str(doc, "run.date", "date", "generated", "generated_at"),
      wall_clock_seconds: num(doc, "run.wall_clock_seconds", "wall_clock_seconds", "elapsed_seconds"),
      arm: str(doc, "run.arm", "arm"),
      n_sequences: num(doc, "run.n_sequences", "n_sequences", "n_windows", "headline.n_sequences"),
      constants_sha256: str(doc, "run.constants_sha256", "constants_sha256", "budget.constants_sha256"),
    },
    headline: {
      p50_minutes: num(doc, "headline.p50_minutes", "headline.p50", "p50_minutes"),
      p95_minutes: num(doc, "headline.p95_minutes", "headline.p95", "p95_minutes"),
      n_missed: num(doc, "headline.n_missed", "n_missed"),
    },
    budget: {
      target: num(
        doc,
        "budget.target_alerts_per_camera_night",
        "budget.target",
        "budget_target",
      ),
      // The UNCAPPED rate, and the JSON must label it: rank.realised_rate counts
      // threshold crossings without the budget cap, because a capped count makes every
      // threshold satisfy 2.0/camera-night and the solver degenerates. That is the
      // quantity solve_threshold was calibrated on, so it is the one shown here.
      realised_uncapped: num(
        doc,
        "budget.realised_alerts_per_camera_night_uncapped",
        "budget.realised_uncapped",
        "budget.realised_alerts_per_camera_night",
        "realised_alerts_per_camera_night",
      ),
      threshold: num(doc, "budget.threshold", "threshold"),
      drift_tolerance: num(doc, "budget.drift_tolerance", "drift_tolerance"),
    },
    systems: readSystems(doc),
    operating: {
      series: readOperating(doc),
      operating_point: num(
        doc,
        "operating_curve.operating_point",
        "operating_point",
        "budget.target_alerts_per_camera_night",
        "budget.target",
      ),
    },
    sensitivity: readSensitivity(doc),
    ablation: readAblation(doc),
    escalation: {
      fraction: num(doc, "escalation.fraction", "escalation_fraction"),
      max: num(doc, "escalation.max", "escalation_fraction_max"),
    },
    classifier: {
      precision: num(doc, "classifier.val_precision", "classifier.precision", "val_precision"),
      recall: num(doc, "classifier.val_recall", "classifier.recall", "val_recall"),
      dataset: str(doc, "classifier.dataset", "classifier.corpus"),
    },
    exclusions: readExclusions(doc),
    raw: isDict(doc) ? doc : {},
  };
}

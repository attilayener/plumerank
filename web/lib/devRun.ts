/**
 * The development-set run, compiled in -- the ONE bundled result, and only for the
 * pending state.
 *
 * /results renders /api/benchmark at runtime and never a bundled copy of the headline.
 * Before the held-out run exists there is no headline to disagree with, and the
 * committed DEV payload is real evidence a judge should not have to clone the repo to
 * see. So it is shown -- read through the same readBenchmark() as the live file, always
 * labelled "development set, in-sample, not the headline" -- and disappears the moment
 * /api/benchmark answers. scripts/sync-dev-run.mjs refreshes lib/generated/dev-run.json
 * from the newest shipped results/arm_dev_<date>.json before every build.
 */

import devDoc from "./generated/dev-run.json";
import { readBenchmark, str } from "./read";

export const DEV_RUN = readBenchmark(devDoc);
export const DEV_RUN_FILE = `results/${str(devDoc, "results_file") ?? "arm_dev_<date>.json"}`;

const headline = DEV_RUN.systems.find((s) => s.key === "plumerank") ?? null;

/** The handful of DEV figures the home page and /judge quote, read from the file. */
export const DEV_SUMMARY = {
  detected: headline?.n_detected ?? null,
  n: headline?.n ?? null,
  p50: headline?.p50_minutes ?? null,
  /** Realised on the fire-free confusion set -- the rate the threshold was solved to. */
  calibrationRate: headline?.alerts_per_camera_night ?? null,
  /** Realised on the development fires themselves, uncapped. Higher, as it should be. */
  devRate: headline?.arm_alerts_per_camera_night ?? null,
};

# The 11 inverted sequences are not spiky nights — they are LOUD cameras, and a per-camera baseline reaches them

**Date:** 2026-09-16 · **Status: a measurement on DEV at a rate-matched proxy threshold; the
calibrated number is a separate `bench.py --baseline` run and is reported in its own file.**
Reproduce: `python3 spike/peak_shape.py` (raw: `results/peak_shape_2026-09-16.json`).
Starts from `spike/why_missed.py`, which must still print 9/29 = 31.0%; it does.

---

## 1. The hypothesis this file was written to test, and its answer: no

`alert_budget_2026-09-06.md` §3: in 11 of 29 DEV sequences the camera's own highest hazard of
the night happens **before** ignition; all 11 are misses. The cheapest possible mechanism was
that those pre-ignition peaks are **events** — a single-frame spike from an exposure step or a
bird — while a plume is a **process** that holds. If so, a persistence transform of the fused
hazard separates them in time without touching S1–S4.

Half of that is true and it does not help. Of the 11 pre-ignition maxima, 7 are single-frame
spikes (longest run at or above the fire's best minute = 1) and the widest is 3 frames. But
the fire's own margin is just as thin: on the 18 detectable sequences the fire holds *above
the quiet night's best minute* for a **median of 3 frames, minimum 1**. Persistence erases
the false spike and the true margin together:

| transform | added latency | inversions (of 29) | DEV-rate-matched detections |
|---|---:|---:|---:|
| shipped (raw) | 0 min | 11 | **9 / 29** |
| min over 2 · 3 · 4 · 5 frames | 1 · 2 · 3 · 4 | 9 · 9 · 10 · 10 | 7 · 9 · 12 · 8 |
| median over 3 · 5 · 7 | 1 · 2 · 3 | 9 · 8 · 10 | 5 · 9 · 6 |
| EMA α = 0.5 · 0.3 · 0.2 | 3 · 7 · 10 | 10 · 8 · 8 | 8 · 6 · 6 |

Nothing in that table leaves the two-sequence noise band `s4_ablation_2026-09-06.md` §3
established, and every row pays minutes for it. **Persistence of the fused hazard is closed.**

## 2. What the same data says instead

Within-camera AUC of post-ignition hazard against pre-ignition hazard, per sequence, next to the
mean hazard of the quiet night:

| | median within-camera AUC | mean pre-ignition hazard |
|---|---:|---|
| 18 not inverted (9 detected among them) | **0.785** | 0.34 – 0.43 on 14 of them |
| 11 inverted (all missed) | 0.551 | **0.47 – 0.61 on 10 of them** |

The inverted sequences are not nights with one bad minute. They are cameras that run **0.15–0.25
hotter all night**, before any fire — and the fusion emits an *absolute* hazard while the
product spends its budget *per camera-night*. A camera whose quiet baseline is 0.59 cannot be
separated from a camera whose fire peaks at 0.58 by any one threshold, and the ranking never
sees the difference because it ranks absolute values across cameras.

That is a baseline problem, and a baseline has a standard fix: score each camera against its
own recent night.

## 3. The per-camera baseline, at the same DEV-rate-matched proxy

`z_t = (h_t − mean(h[t−W : t])) / std(h[t−W : t])`, causal, 5-frame warm-up pinned to zero.
No added latency — the transform reads the past, not the future.

| window W (frames = minutes) | detections | mean TTD | **median TTD** |
|---:|---:|---:|---:|
| 10 | **19–21 / 29** | 16.6–17.8 | 13.0–14.0 |
| 15 | **17–18 / 29** | 16.7–16.9 | 13.0–14.5 |
| 20 | 15–16 / 29 | 16.9–17.7 | 13.0–14.5 |
| 30 | 14–16 / 29 | 16.6–17.4 | — / 14.5 |
| expanding (whole night so far) | 7 / 29 | 13.3 | — |

Ranges are across five values of a floor on the trailing standard deviation (1e-6 … 0.05); the
result is flat in the floor and monotone in the window. The expanding window fails for a
reason that confirms the mechanism: after ignition it absorbs the plume into the camera's own
baseline, so the plume's z falls as it grows. A trailing window gives the plume W minutes to be
seen before it becomes "normal", and mean TTD lands where that predicts.

**The median time-to-detection — the metric the one-liner is written around and which did not
exist under the shipped pipeline at any budget policy's ceiling — exists under every trailing
window from 10 to 20 frames, at 13–14 minutes.**

## 4. What this is not, yet

- **Not a calibrated number.** The threshold above is solved on DEV so that the count of
  frames at or above it equals the shipped run's (70). The product solves its threshold on
  CONFUSE for 2.00 alerts/camera-night. A z-score on a fog night may fire on blips the
  absolute hazard never reached, push the calibrated threshold up, and take detections with
  it. `bench.py --baseline ztrail15` / `ztrail10` answer that; their artefacts carry the cell
  in the filename (`arm_dev_ztrail15_<date>.json`) and are reported separately.
- **Not a change to the shipped pipeline.** `plumerank/` is untouched. Adopting a baseline
  changes what the UI's hazard chips mean (relative to the camera's night, not absolute) and is
  a human decision — the same class as the presence-term gate, and it should be decided with it.
- **Not a verdict on S1–S4.** The per-frame statistics are exactly as informative as they were;
  this changes how their output is read across cameras, not what they measure.

## 5. One more thing the AUC column says

Three sequences have within-camera AUC **below 0.4** (Border58 0.289, Palisades/69bravo 0.295,
Posta 0.372): on those cameras the fire *lowers* the hazard. No baseline reaches them — the
per-frame statistics are pointing the wrong way. They are the honest residual for the S3 gate
on the rebuilt fog class, and they should be named in the limitations section whatever the
calibrated run says.

# Gates 2 and 3, judged where the weights are used: the shipped weights stay

**Date:** 2026-09-16 · **Status: measured; decided by the human the same day.**
Reproduce: `python3 spike/refit_through_baseline.py`.

Gate 0 made the fusion's output an *input* — to `rank.Baseline` — rather than the ranked
quantity. So a candidate weight set is judged on time-to-detection through the baseline at
the alert budget, on the tuning arm, at the same DEV-rate-matched proxy `peak_shape.md` used
(which the calibrated runs then reproduced exactly).

| weights | absolute pipeline | **through baseline** | mean TTD | **p50, all 29** |
|---|---:|---:|---:|---:|
| **shipped (`constants.py`)** | 9 / 29 | **17 / 29** | 16.9 | **32.0 min** |
| refit, no presence term (`fusion_refit_2026-09-06.json`) | 7 / 29 | 15 / 29 | 17.1 | 38.0 min |
| refit, presence explicit (`statistics_confound §6.2`) | 12 / 29 | 14 / 29 | 15.8 | *n/a* |

## What decided it

- **The refit does not fit better.** Its objective — leave-one-sequence-out per-frame AUC —
  moves 0.6387 → 0.6393. The only weight it changes is `FUSION_W_ESCALATION`, from +0.78 to
  −0.38: a sign flip worth 0.0006 AUC is a weight the per-frame fit **cannot identify**.
  Where the weight *is* identified — the headline on the tuning arm — the shipped value wins
  by two sequences and six minutes of p50.
- **Presence explicit is the best per-frame model and the worst through the baseline.** The
  baseline already reads a component-less camera as "nothing happening" (a constant hazard,
  z = 0), so an explicit presence column buys +0.025 LOSO on frames and loses three sequences
  on detections. Gate 2: **no presence term.** It also puts S3 at −2.21 in the shipped
  equation, which the report would then have to explain against the spec §2.
- **HOLD is untouched.** D11 says weights are fit on DEV; choosing among three DEV-fitted
  sets by the DEV headline is tuning on the tuning arm, declared here. The held-out number
  is produced once, later, by `make bench`.

## What the report can say about the learned component

The honest sentence, which survives a judge's question: *the ONNX tile classifier is invoked
on 2.23% of frames; its fusion weight is not identified by the per-frame fit on DEV (a sign
flip moves LOSO by 0.0006); it is kept at the value under which the headline is best on the
tuning arm.* Not: *the learned component helps.*

Decision: **gate 2 no · gate 3 keep shipped weights** (the 2026-09-16 gate record).

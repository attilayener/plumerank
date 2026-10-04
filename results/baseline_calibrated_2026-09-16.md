# Per-camera baseline at the calibrated operating point: 17–18 of 29, and the median exists

**Date:** 2026-09-16 · **Status: measured, NOT adopted.** `plumerank/` is unchanged; the
decision was a human one, taken on 2026-09-16.
Follows `peak_shape_2026-09-16.md`, which could only rate-match on DEV.

Both cells: threshold solved on the 33 CONFUSE windows for 2.00 alerts/camera-night
(realised 1.9885, uncapped), spent on the 29 DEV sequences through the unchanged `rank.emit`.
Same seed, same exclusions, same denominator as `arm_dev_2026-09-06.json`.
Commit `07453ca` — the first containing `--baseline`; two earlier runs of the same cells at
an uncommitted working tree gave the same numbers and were discarded for provenance.

| | shipped (`arm_dev_2026-09-06`) | `--baseline ztrail15` | `--baseline ztrail10` |
|---|---:|---:|---:|
| detected | 9 / 29 · 31.0% | **17 / 29 · 58.6%** | **18 / 29 · 62.1%** |
| p50 TTD, misses = ∞ | *n/a* | **32.0 min** | **30.0 min** |
| p95 | n/a | n/a (12 misses) | n/a (11 misses) |
| mean TTD of detected | 21.3 min | 16.9 min | 17.9 min |
| CONFUSE threshold | 0.740112 | 0.849182 | 0.875122 |
| model calls / camera-night | 2 | 2 | 2 |
| added latency | — | 0 (5-frame warm-up) | 0 (5-frame warm-up) |

`peak_shape.md` §3 predicted 17/29 · 16.9 min for W = 15 at a DEV-rate-matched proxy; the
calibrated run gives 17/29 · 16.9 min. The risk named in its §4 — a z-score firing on fog-night
blips and dragging the calibrated threshold up — did not materialise: CONFUSE placed the
threshold at z ≈ 3.5 against the proxy's 3.4.

10 vs 15 is one sequence apart, inside the two-sequence noise band. The window is a design
choice, not a measurement.

Reproduce: `python3 scripts/bench.py --arm dev --baseline ztrail15` (and `ztrail10`), ~9 min each.

# For judges — the 30-second path

**The claim:** PlumeRank ranks which wildfire-camera tiles a dispatcher should open next,
calibrated to a fixed budget of 2 alerts per camera-night, and measures itself on median
minutes-after-ignition until the right camera reaches the top of that ranking. (A *camera-night*
here is one 81-minute window of footage on one camera — the length of a FIgLib sequence — not a
12-hour night.)

## 1. Click these, in order

1. **The Wall** — `http://44.242.91.201` — no login. The live ranking: 29 development-set fire
   sequences and 33 fire-free confusion-set windows (the confusion set is marked on screen),
   replayed from real HPWREN footage through the real pipeline. Until the held-out run is committed, its threshold is
   the one the development run solved on the fire-free confusion set, and the Wall says so (§4).
2. **`/build-info`** — `http://44.242.91.201/build-info` — `GET /build-info`. Real
   `cv2.getBuildInformation()` output, the EC2 instance id and AMI id, and the git commit this
   box is running, read live off the box — nothing here is typed by hand.
3. **`/results`** — `http://44.242.91.201/results` — the same `/build-info` block rendered
   in-page, and the development-set evidence (baseline race, operating curve, sensitivity strip)
   labelled in-sample. The held-out versions, and the ablation grid, appear once the held-out run
   is committed.

## 2. The receipt

| | |
|---|---|
| Tests | **714 passed, 2 skipped**, 97% coverage on `plumerank/` |
| Property-based testing | **1,500 generated cases** (`tests/test_invariants.py`, `hypothesis`) — the fused hazard never leaves `[0, 1]` across 500 valid-input cases and never silently passes an invalid one through another 500; the alert budget can never be overspent under any of 500 generated call sequences |
| Static analysis | `ruff check .` and `mypy plumerank api --ignore-missing-imports` — both clean |
| Claim/code consistency | `scripts/check_claims.py`, run inside `make test` — 23 defended OpenCV 5 surfaces across 6 modules (18 analysis calls, 5 runtime/introspection), every API route, every dependency, every Makefile target and every file path this document and `README.md` name, checked mechanically against the real source on every test run |
| DEV arm (orientation only — fit data, never the claim) | 17/29 detected, p50 32.0 min; calibrated to 1.99 alerts/camera-night on the confusion set, 2.37 realised on DEV. Like-for-like baselines (B1/B2 through the same per-camera baseline): 14/29 and 10/29 (`results/arm_dev_2026-10-04.json`) |
| HOLD arm (the only number allowed to be called the headline) | **not yet run.** `GET /api/benchmark` 404s (no `results/bench_*.json` yet) rather than inventing one — see §4 |
| Compute | One AWS `t4g.small` (Graviton/ARM), one S3 bucket (the replayed corpus: 5,082 objects, 2.38 GB), CloudWatch metrics, two drift alarms (±15% around the 2.0 budget) and a heartbeat-silence alarm (`deploy/alarms.sh`). No Lambda, no managed vision API. |

## 3. Reproduce it yourself

```bash
git clone https://github.com/edycutjong/plumerank && cd plumerank
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.lock
make test      # ~95s: the 714-test offline suite + coverage + the claim-consistency check above
make verify    # ≤3 min: rebuilds the venv from the lockfile, re-runs the offline suite, then
               # recomputes the micro-corpus metrics and checks them byte-for-byte against the
               # committed results/verify_micro.json. This is the fast sanity check — it does
               # NOT reproduce the published headline, and the command's own output says so.
```

The pair that **does** reproduce the held-out headline, once `results/bench_<date>.json` exists
(the HOLD run — see §4): `make fetch && make bench`, ≈95 minutes (≈35 min corpus fetch + ≈60 min
held-out evaluation). The ablation grid is a separate DEV run,
`scripts/bench.py --arm dev --ablation` (twelve cells, a few hours, run off the box), which the
held-out file quotes verbatim. Nobody is asked to run that to evaluate this submission; it exists so the number is
checkable rather than assertable.

## 4. What is and is not measured yet — read before trusting any number above

**The HOLD run has not happened.** DEV is the arm the fusion weights were fit on; a number from
it is diagnostic, never a claim of generalisation. Until `results/bench_<date>.json` exists:

- `GET /api/benchmark` returns `404` with an explicit message rather than a fabricated number
  (`api/main.py`).
- The live Wall runs on the threshold solved on the fire-free confusion set by the newest
  development run. The threshold depends only on that set, so it is the one the held-out run
  will solve too; `/health` reports which run it came from (`live_pipeline.calibration_kind`).
- `scripts/bench.py`'s `_check_calibration` refuses to publish any threshold it could not bring
  to the budget: *"Publishing this as a calibrated operating point would be a fabrication."*

## 5. Honest limitations

1. **Daylight only** — no IR hardware, no night detection.
2. **Cloud, not fog, is the real adversary, and the single statistics are weak** — the
   statistic (S3) this project was built around scores AUC 0.44 pre- vs post-ignition on DEV
   frames with a component (anti-predictive; S1 0.528 and S2 0.50 are near chance,
   `results/statistics_confound_2026-09-06.md`), yet S3 carries the largest fusion weight and no
   S3-off ablation has run. The per-camera baseline carries the ranking.
   Said here first, not found by you.
3. **One instance, no HA.** Every published number is committed in `results/` so the claim
   survives the box going down.
4. **No dispatch, no suppression, no upload path** — architecturally absent, not merely unused.

## 6. Links

- Live endpoint: `http://44.242.91.201`
- `/build-info` · `/health` · `/results`
- Repository: https://github.com/edycutjong/plumerank (this box's own `/build-info` names the exact commit it runs)
- Site and pitch deck: https://plumerank.edycu.dev · https://plumerank.edycu.dev/pitch/
- `README.md` (architecture + full testing detail) · `DEMO.md` (full reproduce walkthrough)

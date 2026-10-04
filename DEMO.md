# Demo & reproduce

This is the full walkthrough. For the 30-second version, see [`JUDGE.md`](JUDGE.md).

## Live

- Wall: `http://44.242.91.201`
- `GET /build-info` — real `cv2.getBuildInformation()`, instance id, AMI id, commit sha
- `GET /health`
- `/results` — `/build-info` in-page now; operating curve, baseline race, sensitivity strip and
  ablation grid once the held-out run is committed
- `GET /api/watchlist` — the SSE stream the Wall renders
- `GET /api/benchmark` — serves the committed `results/bench_<date>.json` verbatim; **404s
  honestly** until that file exists (the HOLD run — see "What is not measured yet" below)

## Reproduce — real run, measured just now

```bash
git clone <this repo> && cd <this repo>
make test      # offline suite + coverage + claim-consistency check
make verify    # clean-venv sanity check (does NOT reproduce the published headline)
```

Output of `make verify`'s final step (`scripts/bench.py --arm micro`), pasted verbatim; the full
target also rebuilds the `.venv` from `requirements-dev.lock` and runs the offline suite first:

```
========================================================================
REPRODUCED HERE
========================================================================
  what           : the micro corpus, recomputed live by the real OpenCV 5 pipeline
  corpus         : data/micro/ (10 windows x 31 frames)
  PlumeRank      : p50    n/a min   p95    n/a min   (2/5 detected)
  B1 CNN grid    : p50    n/a min   p95    n/a min   (1/5 detected)      (the SAME onnx, every frame)
  B2 frame-diff  : p50    n/a min   p95    n/a min   (0/5 detected)
  realised rate  : 1.57 alerts/camera-night on the 5 non-fire micro windows (target 2.00, uncapped)
  escalation     : 0.65% of frames touched by ONNX (top-2 of a 10-window roster; the published figure in block 2 is the same rule over a larger one)
  checked against: results/verify_micro.json -- MATCH (byte-for-byte)

========================================================================
PUBLISHED, NOT RECOMPUTED HERE
========================================================================
  status         : no committed HOLD run in results/ -- nothing is published yet, and this command does not produce it
  reproduce with: make fetch && make bench  (about 95 min)
```

`make test` alone (the offline suite + coverage + `scripts/check_claims.py`): **714 passed, 2
skipped, 97% coverage on `plumerank/`, ~95s, no network, no credentials.**

### The pair that reproduces the published headline, once it exists

```bash
make fetch   # ~35 min: the corpus, sha256-verified, resumable
make bench   # ~60 min: scripts/bench.py --arm hold --budget 2.0 --sensitivity --seed 20261026
# the ablation grid is its own DEV run, quoted by the held-out file:
.venv/bin/python scripts/bench.py --arm dev --ablation   # twelve cells, a few hours, off the box
```

Nobody needs to run this to evaluate the submission — `results/` already carries every
artefact any document cites, and the live box runs the identical code. It exists so the number
is checkable, not just assertable.

## What is not measured yet, and why that is honest rather than broken

The HOLD arm — the only run whose number this project is willing to call a headline — has not
executed. Concretely, right now:

- `GET /api/benchmark` → `404`, with a message naming the missing `results/bench_*.json`, not a
  number.
- The live Wall runs on the threshold the newest shipped DEV run solved on the fire-free
  confusion set (`results/arm_dev_<date>.json` → `calibration.threshold`). The threshold depends
  only on that confusion set, so it is the one the held-out run will solve too; the Wall labels
  it as development-calibrated, and `/health` reports it under `live_pipeline.calibration_kind`.
- `scripts/bench.py`'s calibration gate (`_check_calibration`) refuses to publish any threshold
  it could not bring to the budget, from either side: *"Publishing this as a calibrated
  operating point would be a fabrication."*

The **DEV** arm exists (the data the fusion weights were fit on — diagnostic only, never the
claim): 17/29 detected, p50 32.0 min; threshold calibrated to 1.99 alerts/camera-night on the
confusion set (2.37 realised on DEV). Like-for-like, B1 and B2 through PlumeRank's own
per-camera baseline detect 14/29 and 10/29; as deployed, 6/29 and 3/29
(`results/arm_dev_2026-10-04.json`).

## What a real-run receipt looks like for this project

There is no per-call provider bill here (no hosted LLM API, no metered generation) — the
"receipt" is the pipeline actually executing against real pixels and being checked against a
committed expectation, which is exactly what the block above shows. The AWS side has its own
receipt: one `t4g.small` (Graviton) instance, one S3 bucket (corpus: the DEV and CONFUSE
frames the box replays, 5,082 objects / 2.38 GB per `aws s3 ls --summarize`),
CloudWatch metrics plus drift (±15%) and silence alarms on the alert budget — all funded by a $150 AWS compute
grant from this competition, not a personal card.

## Honest limitations

See `README.md`'s "Honest limitations" section — identical list, not duplicated here to avoid
the two drifting apart.

# Demo & reproduce

This is the full walkthrough. For the 30-second version, see [`JUDGE.md`](JUDGE.md).

## Live

- Wall: `http://44.242.91.201`
- `GET /build-info` — real `cv2.getBuildInformation()`, instance id, AMI id, commit sha
- `GET /health`
- `/results` — operating curve, baseline race, sensitivity strip, ablation grid
- `GET /api/watchlist` — the SSE stream the Wall renders
- `GET /api/benchmark` — serves the committed `results/bench_<date>.json` verbatim; **404s
  honestly** until that file exists (the HOLD run — see "What is not measured yet" below)

## Reproduce — real run, measured just now

```bash
git clone <this repo> && cd <this repo>
make test      # offline suite + coverage + claim-consistency check
make verify    # clean-venv sanity check (does NOT reproduce the published headline)
```

**`make verify`, executed end-to-end for this README, no shortcuts:**

```
real: 1:53.67 (includes a from-scratch .venv rebuild + pip install from requirements.lock)

Block 1 -- REPRODUCED HERE (the real OpenCV 5 pipeline, run live against real pixels)
  corpus         : data/micro/ (10 windows x 31 frames)
  PlumeRank      : 2/5 detected
  B1 (same ONNX, every frame) : 1/5 detected
  B2 (frame differencing)     : 0/5 detected
  realised alert rate  : 1.57 alerts/camera-night (target 2.00, uncapped on this slice)
  escalation fraction  : 0.65% of frames touched by the ONNX tile classifier
  checked against results/verify_micro.json : MATCH (byte-for-byte)

Block 2 -- PUBLISHED, NOT RECOMPUTED HERE
  status: no committed HOLD run in results/ -- nothing is published yet, and this
          command does not produce it. Reproduce with `make fetch && make bench` (~95 min).
```

`make test` alone (the offline suite + coverage + `scripts/check_claims.py`): **633 passed, 2
skipped, 97% coverage on `plumerank/`, ~95s, no network, no credentials.**

### The pair that reproduces the published headline, once it exists

```bash
make fetch   # ~35 min: the corpus, sha256-verified, resumable
make bench   # ~60 min: scripts/bench.py --arm hold --budget 2.0 --sensitivity --seed 20261026
```

Nobody needs to run this to evaluate the submission — `results/` already carries every
artefact any document cites, and the live box runs the identical code. It exists so the number
is checkable, not just assertable.

## What is not measured yet, and why that is honest rather than broken

The HOLD arm — the only run whose number this project is willing to call a headline — has not
executed. `plumerank/rank.py`'s calibration gate (`_check_calibration`) refuses to publish an
operating point it cannot support: *"Publishing this as a calibrated operating point would be a
fabrication."* Concretely, right now:

- `GET /api/benchmark` → `404`, with a message naming the missing file, not a number.
- The live Wall's `/health` reports `live_pipeline.available: false` with the same reason, and
  the UI shows that degraded state rather than inventing a ranking.
- `GET /build-info` and `GET /health` are unaffected — they do not depend on calibration.

The **DEV** arm exists (the data the fusion weights were fit on — diagnostic only, never the
claim): 17/29 detected, p50 32.0 min, 1.99 alerts/camera-night, baselines B1 1/29 and B2 3/29
(`results/arm_dev_2026-09-16.json`).

## What a real-run receipt looks like for this project

There is no per-call provider bill here (no hosted LLM API, no metered generation) — the
"receipt" is the pipeline actually executing against real pixels and being checked against a
committed expectation, which is exactly what the block above shows. The AWS side has its own
receipt: one `t4g.small` (Graviton) instance, one S3 bucket (corpus: 5,082 frames, 2.38 GB),
CloudWatch metrics plus a drift alarm on the alert budget — all funded by a $150 AWS compute
grant from this competition, not a personal card.

## Honest limitations

See `README.md`'s "Honest limitations" section — identical list, not duplicated here to avoid
the two drifting apart.

<div align="center">
  <h1>PlumeRank 🔥📷</h1>
  <p><em>A wall of wildfire cameras. One dispatcher. Which one gets opened next?</em></p>

  <br/>

  [![Live Endpoint](https://img.shields.io/badge/🟢_Live-44.242.91.201-06b6d4?style=for-the-badge)](http://44.242.91.201)
  [![/judge](https://img.shields.io/badge/👩‍⚖️_For-Judges-f59e0b?style=for-the-badge)](http://44.242.91.201/judge)
  [![Built for OpenCV AI Competition 2026](https://img.shields.io/badge/OpenCV_AI_Competition-2026-8b5cf6?style=for-the-badge)](https://opencv26.devpost.com/)

  <br/>

  ![OpenCV 5](https://img.shields.io/badge/OpenCV-5.0.0-5C3EE8?style=flat&logo=opencv&logoColor=white)
  ![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
  ![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat&logo=next.js)
  ![ONNX](https://img.shields.io/badge/ONNX-runtime%3A_cv2.dnn-005CED?style=flat&logo=onnx&logoColor=white)
  ![AWS Graviton](https://img.shields.io/badge/AWS-Graviton_t4g.small-FF9900?style=flat&logo=amazonaws&logoColor=white)
  ![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat)

</div>

---

## The question

It is 04:50 on a night shift at a CAL FIRE emergency command centre, and the one dispatcher
awake is looking at more than 1,000 camera tiles with no instrument that tells her which four
to open in the next sixty seconds.

Every published smoke detector optimises **per-frame accuracy** — F1, precision, recall. None
of that tells her which tile to open, and none of it tells a fire agency whether the system
will survive contact with a real shift before it gets muted. The number that matters is
different: **how many minutes after ignition does a human actually look at the right camera,
given she will only tolerate a fixed number of interruptions per camera per night?**

**PlumeRank publishes that number.** It watches a camera network and emits, every minute, a
ranked shortlist of which cameras a human should look at now — each entry carrying its own
cropped evidence tile — calibrated to a fixed budget of **2 alerts per camera-night**. It never
suppresses a camera and never dispatches an engine. It reorders human attention.

---

## What it actually measures (today, honestly)

> **This project does not yet have a headline number, and says so on purpose.** The held-out
> (HOLD) evaluation — the only run whose number is allowed to be called the headline — has not
> run yet. What exists today is the **DEV** arm (the data the fusion weights were fit on, never
> the number a judge should be shown as proof): **17 of 29 sequences detected, median
> time-to-detection 32.0 minutes, at 1.99 alerts/camera-night** (`results/arm_dev_2026-10-04.json`).
> That number is for orientation only. Until the HOLD run exists, `/api/benchmark` on the live
> box correctly 404s and the live wall correctly shows itself as degraded rather than inventing
> a ranking — see [`JUDGE.md`](JUDGE.md) and `api/main.py`'s `_calibrated_threshold`.

| Metric | DEV (orientation only — never the claim) |
|---|---|
| Detected | 17 / 29 |
| Median time-to-detection | 32.0 min |
| Realised alert rate | 1.99 alerts/camera-night (target 2.0) |
| Escalation fraction (frames touched by the ONNX classifier) | 2.23% — over the 2% design cap the run itself records (`within_max: false`) |
| Baseline B1 (same CNN, every frame) | 6 / 29 detected — no median |
| Baseline B2 (frame differencing) | 3 / 29 detected — no median |
| B1 through PlumeRank's own per-camera baseline (like-for-like) | 14 / 29 detected — no median |
| B2 through PlumeRank's own per-camera baseline (like-for-like) | 10 / 29 detected — no median |

## Honest limitations

1. **Daylight only.** HPWREN FIgLib (the ground-truth dataset) is daylight-only and no IR
   camera exists in this build. A night-shift dispatcher works hours this system cannot cover.
2. **Cloud is the adversary — not the fog this project was designed against — and the
   statistics are weaker than the design assumed.** Measured on 2026-09-06 under the
   *pre-baseline* pipeline (threshold 0.722, not re-measured under the shipped one), low cloud
   cost 4.25 and cloud shadow 3.86 alerts/camera-night against fog's 1.15
   (`results/class_split_2026-09-06.json`) — and that fog figure is on a contaminated class: of
   54 frames inspected, roughly 27 show no marine layer at all (`results/fog_class_review.md`).
   A relabelled fog census is in progress. Per statistic, on DEV frames with a component,
   pre- vs post-ignition: S3 (diffuse boundary) scores AUC 0.44 — anti-predictive — and S1 0.528
   and S2 0.50 are near chance (`results/statistics_confound_2026-09-06.md`). S3 was never tested
   against fog directly. What carries the ranking is scoring each camera against its own
   trailing 15 minutes, not any single statistic.
7. **One fusion weight is not identified.** Refitting moves `FUSION_W_ESCALATION` from +0.78 to
   −0.38 for a 0.0006 change in leave-one-sequence-out AUC; the shipped value is kept because
   it scores two sequences and six minutes of p50 better on DEV
   (`results/refit_through_baseline_2026-09-16.md`). DEV is the tuning arm, so that choice is
   in-sample by construction.
3. **The novelty is a metric reframe, not a new sensor.** The individual OpenCV operations are
   well-known; what is new is the primitive that composes them and the attention-budget
   objective they are pointed at.
4. **One instance, no HA, no static mirror.** The endpoint is a single free-tier Graviton box.
   Every published number is committed in `results/` so the claim survives the box going down.
5. **No managed cloud vision API baseline.** Both baselines are calibrated to the same alert
   budget as PlumeRank; one of them (B1) shares PlumeRank's own neural network, run over every
   frame instead of the ~2% the classical stage escalates — a controlled experiment, not a
   contest between differently-sized models.
6. **No upload path, no dispatch path, no suppression path — by design, not by omission.** See
   "Responsible use" below.

---

## How it works

Over a rolling, stabilised window, each foreground component on each camera is measured for:

| # | Statistic | Physical claim |
|---|---|---|
| S1 | **Base anchoring** | a fire is tied to the ground; a cloud is not |
| S2 | **Vertical growth anisotropy** | a plume rises faster than it spreads |
| S3 | **Diffuse-boundary contrast** | *measured not to separate on its own* — kept in the fused model anyway, see limitation 2 |
| S4 | **In-box optical-flow drift** | a plume drifts coherently; sensor noise does not — kept on evidence: 17/29 with it vs. 14/29 without, under the shipped baseline |

Those four fold into a per-camera hazard (a fixed logistic, weights fit on DEV, no presence
term). Each camera is then scored against **its own trailing 15 minutes** — a camera that runs
hot all night is not a fire — before a solver calibrates the alert threshold to the budget.

**The learned component never sees a Californian pixel.** The one trained model — a small ONNX
tile classifier reached through `cv2.dnn` — is trained only on Pyro-SDIS (French fire services,
Apache-2.0) and touches only the ~2% of frames the classical stage escalates. That same model
*is* baseline B1, run over every frame instead of 2% of them: PlumeRank cannot win this
comparison by having a bigger model, because the model is identical on both sides.

### Why OpenCV 5, specifically

**Required by the competition**, and structural rather than decorative: 23 defended analysis
surfaces across 6 OpenCV modules, mechanically counted from the source by
`scripts/check_claims.py` on every `make test` run (never hand-typed):

| Surface | Lives in | Without it |
|---|---|---|
| `createBackgroundSubtractorMOG2` + `.apply` | `plumerank/segment.py` | a per-pixel Gaussian-mixture background model, rewritten from the paper |
| `phaseCorrelate` + `createHanningWindow` (+ `phaseCorrelateIterative`, new in 5.x) | `plumerank/stabilise.py` | a padded DFT cross-power-spectrum stabiliser with sub-pixel refinement |
| `estimateAffine2D(..., USAC_MAGSAC)` | `plumerank/stabilise.py` | a correct MAGSAC++ implementation |
| `connectedComponentsWithStats`, `Sobel`, `Laplacian` | `plumerank/segment.py`, `plumerank/statistic.py` | the component boxes S1/S2 read, and the boundary energies S3 reads |
| `DISOpticalFlow_create` | `plumerank/statistic.py` (S4) | a dense flow field at usable speed |
| `dnn.readNetFromONNX` (`dnn.ENGINE_NEW`) + `dnn.blobFromImage` + `dnn.Net.forward` | `plumerank/escalate.py` | a second inference runtime on the arm64 box |
| `getBuildInformation` | `api/main.py` → `GET /build-info` | nothing a judge could `curl` to check the OpenCV 5 claim |

The remaining surfaces (two background-subtractor variants kept for the ablation grid, CLAHE
contrast normalisation, morphology clean-up, build/timing introspection) are listed in full by
`scripts/check_claims.py`'s own output.

### Architecture

```
HPWREN FIgLib replay ──▶ stabilise (phaseCorrelate + MAGSAC) ──▶ segment (MOG2 + connected components)
                                                                        │
                                                            S1·S2·S3·S4 statistics
                                                                        │
                                               escalate (~2% of frames) ──▶ ONNX tile classifier (cv2.dnn)
                                                                        │
                                                      fuse ──▶ per-camera baseline (trailing 15 min)
                                                                        │
                                              solve_threshold (alert budget) ──▶ emit (ranked watch list)
                                                                        │
                                   FastAPI (/api/watchlist SSE, /api/benchmark, /health, /build-info)
                                                                        │
                                                        Next.js — the Wall, /results, /judge
```

One t4g.small (Graviton/ARM) instance, one S3 bucket for the corpus, CloudWatch for the alert
budget and a silence alarm. No Lambda, no managed vision API, no second instance.

---

## Responsible use

1. **Ranking aid for a human, not dispatch.** There is no dispatch path anywhere in this
   codebase (`tests/test_api.py::test_there_is_no_upload_route`, and there is no equivalent
   dispatch route either — `test_exactly_the_four_documented_routes_exist`).
2. **Never hides a camera.** The full roster stays on screen; the system only reorders it.
3. **The alert budget is a published, logged, tunable parameter** — a CloudWatch metric with a
   drift alarm, not a constant buried in a config file.
4. **Every ranked entry ships its evidence crop**, so a human can overrule the ranking in one
   glance.
5. **Derived statistics are persisted, not frames.** No identifiable data exists in the pipeline.

---

## Testing

```
664 passed, 2 skipped · 97% coverage on plumerank/ · ruff clean · mypy clean
```

| Layer | Tool | Notes |
|---|---|---|
| Unit + integration | `pytest` | `make test` — offline, no network, no credentials, ~95s |
| Property-based / exhaustive | `hypothesis` | `tests/test_invariants.py` — **1,500 generated cases** (500 each) verifying the fused hazard never leaves `[0, 1]` for any valid input and never silently passes through an invalid one, and that the alert budget can never be overspent under any call sequence |
| Regression | 630+ example tests, most named after the specific behaviour or defect they pin (e.g. `test_an_unusable_threshold_degrades_the_wall_instead_of_500ing_health`) | `tests/` |
| Claim/code consistency | `scripts/check_claims.py` (runs inside `make test`) | every OpenCV symbol, API route, dependency, Makefile target and file path this document names is mechanically checked against the real source — a document and the code can disagree for at most one test run |
| Static analysis | `ruff check .`, `mypy plumerank api --ignore-missing-imports` | both clean |
| Secret scanning | `gitleaks` (full history), GitHub secret scanning + push protection | CI |
| Dependency audit | `pip-audit`, `npm audit` | CI, advisory |
| SAST | CodeQL (`python`, `javascript-typescript`) | CI |
| E2E | Playwright, against the **real** FastAPI backend (no mocks) | `web/e2e/` |

```bash
make test              # offline suite + coverage, ~95s, no network, no credentials
make verify             # ≤3 min: rebuilds .venv, runs the offline suite, then recomputes
                         # the micro-corpus metrics and checks them against the committed
                         # results/verify_micro.json. Does NOT reproduce the published
                         # headline and never claims to.
make fetch && make bench  # ~95 min total: the pair that reproduces the held-out headline
                           # once results/bench_<date>.json exists (HOLD run, pending)
```

See [`DEMO.md`](DEMO.md) for the full reproduce walkthrough and [`JUDGE.md`](JUDGE.md) for the
30-second path.

---

## Getting started

### Backend (root)
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.lock
.venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn api.main:app --reload   # http://127.0.0.1:8000
```

### Frontend (`web/`)
```bash
cd web && npm install
cp .env.example .env.local   # NEXT_PUBLIC_PLUMERANK_API — see web/.env.example
npm run dev                  # http://127.0.0.1:3000
```

No API keys, no accounts, no login anywhere in this stack.

---

## Project structure

```
plumerank/       # the pipeline: stabilise, segment, statistic, escalate, rank, metrics, types
api/             # FastAPI app — /build-info, /health, /api/watchlist, /api/benchmark
scripts/         # fetch.py, bench.py, check_claims.py, check_submission_readiness.py
web/             # Next.js frontend — the Wall ("/"), "/results", "/judge"
tests/           # 664 tests, 97% coverage on plumerank/
deploy/          # EC2 bootstrap (Caddy, systemd, OpenCV 5 build pin)
results/         # committed run artefacts — every published number lives here
```

---

## Track selection

**Overall only.** Neither special-award box is ticked, on purpose:

- **Best Use of COOL — declined.** This pipeline's hot loop is background modelling and optical
  flow, not resize / adaptive-threshold / contours; a COOL claim would benchmark flat against
  that award's own measured-performance criterion.
- **Agentic Vision — declined.** There is no orchestration loop. The dispatcher is
  human-in-command by design — that *is* the safety argument (see "Responsible use" above).

## License

[Apache License 2.0](LICENSE) © 2026 Edy Cu

## Acknowledgments

Built for the **OpenCV AI Competition 2026, powered by AWS**. Imagery: HPWREN FIgLib
(`http://hpwren.ucsd.edu/`). Training data for the single learned component: Pyro-SDIS
(Apache-2.0, French fire services) — never a Californian pixel.

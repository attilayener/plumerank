# PlumeRank — Technical Report

OpenCV AI Competition 2026, powered by AWS. See [`README.md`](README.md) for the pitch,
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the system diagram, and [`JUDGE.md`](JUDGE.md) for
the 30-second path. This document is the problem/users/evaluation/limitations write-up the
competition's required deliverables list asks for, in one place.

## 1. Problem

Wildfire camera networks (ALERTWildfire, HPWREN, Pyronear) now carry thousands of live tiles,
watched by one or two people per shift. Every published smoke detector optimises **per-frame
accuracy** — F1, precision, recall. None of that number tells a dispatcher which of a thousand
tiles to open next, and none of it tells a procurement officer whether the system will survive
contact with a real shift before it gets muted for crying wolf.

The number that matters is different: **how many minutes after ignition does a human actually
look at the right camera, given she will tolerate only a fixed number of interruptions per
camera per night?** Raise sensitivity and you buy minutes at the cost of alarm fatigue; lower
it and the network goes quiet and late. Every deployed system lives somewhere on that curve.
Nobody publishes it. PlumeRank does.

## 2. Users

- **Primary — the night-shift dispatcher or camera watcher** at a public fire agency command
  centre. Not a data scientist, not a developer. The tool has to be legible at a glance, at
  04:50, on a projector, without training — which is why the Wall's landing band is "cameras,
  rank, evidence crop" and nothing else on the first screen.
- **Secondary — the network operations engineer** who sets the alert budget and answers for
  it. Served by the budget ledger and the CloudWatch drift alarms — the budget is a knob with
  telemetry, not a constant buried in a config file.
- **The population this stands for.** Radeloff et al., *PNAS* 2018: about 43 million US houses —
  roughly one in three — sit in the wildland–urban interface, and the interface is the
  fastest-growing land-use type in the country. Concretely: Paradise, California,
  2018 — 85 deaths, 18,804 structures lost, in a fire whose growth rate made every minute of
  detection latency worth acres of spread.

## 3. Architecture

Full diagram and data flow: [`ARCHITECTURE.md`](ARCHITECTURE.md). In one paragraph: a classical
OpenCV 5 stage (`plumerank/stabilise.py`, `segment.py`, `statistic.py`) measures four statistics
per foreground component on every frame; they are fused into a hazard, the top-2 ranked
cameras whose hazard clears an escalation floor (~2% of frames) get one ONNX tile score through
`cv2.dnn` folded in (`plumerank/escalate.py`), each camera is then scored against its own
trailing 15 minutes, and that hazard is thresholded to a published alert budget
(`plumerank/rank.py`); and the
ranked result is served by a four-route FastAPI app and a three-route Next.js static export on
one AWS `t4g.small` instance.

## 4. OpenCV 5 implementation

**Required by the competition** ("every entry must use OpenCV 5 for substantive image or video
analysis") and structural here, not decorative: **23 defended analysis surfaces across 6
modules** (video, imgproc, geometry, bgsegm, dnn, core), mechanically counted from
the real source on every `make test` run by `scripts/check_claims.py` — never hand-typed, never
quoted from a plan. The full table with per-surface "what you would need without it" is in
`README.md`'s "Why OpenCV 5" section.

The one trained component — a 96×96 ONNX tile classifier, reached through
`cv2.dnn.readNetFromONNX(ENGINE_NEW)` — is trained **only** on Pyro-SDIS (Apache-2.0, French
fire services) and never sees a FIgLib (Californian) pixel; a guard test asserts this at the
manifest level. The identical model is also baseline B1, run over every frame instead of the
~2% the classical stage escalates — the comparison is a controlled experiment rather than a
contest between differently-sized models, because the learned component is the same size on
both sides of it.

## 5. AWS deployment

One `t4g.small` (Graviton/ARM, burstable, `unlimited` credit mode) EC2 instance, one S3 bucket
holding the corpus and a git bundle of the source, CloudWatch carrying the realised alert rate
and escalation fraction as metrics with drift alarms (±15% of the budget) and a heartbeat-silence
alarm (`deploy/alarms.sh`), and an IAM instance profile with read-only S3 access to the corpus
plus `cloudwatch:PutMetricData` scoped to the `PlumeRank` namespace. `deploy/bootstrap.sh` provisions the box from the S3
git bundle; `deploy/Caddyfile` reverse-proxies `/api/*`, `/health` and `/build-info` to Uvicorn
on the **same origin** as the static frontend, so production never makes a cross-origin
request the way a local two-port dev setup does.

`GET /build-info` exists specifically so a judge does not have to trust a sentence: it returns
`cv2.getBuildInformation()`, the running OpenCV version, the EC2 instance id and AMI id, and
the git commit sha, read live off the box.

## 6. Evaluation

**Ground truth is mechanical, not hand-labelled.** HPWREN FIgLib records each sequence's
ignition time, so time-to-detection needs zero labelling, no raters, and no institutional
partner. **Every system is calibrated to the same alert budget on the same confusion set**
before comparison (all land at 1.99 alerts/camera-night there), so the comparison happens at
the operating point a fire agency actually buys at, not at each system's most flattering one.
The like-for-like rows put B1 and B2 through the same per-camera baseline PlumeRank uses; on
DEV the realised rates then differ by system, and are listed.

**What is measured today — the DEV arm (diagnostic, never the headline):**

| Metric | DEV, shipped configuration |
|---|---|
| Detected | 17 / 29 |
| Median time-to-detection | 32.0 min |
| Alert rate | 1.99 alerts/camera-night on the confusion set (target 2.0); 2.37 realised on DEV |
| Escalation fraction | 2.23% of frames — over the 2% design cap (`within_max: false` in the run file) |
| **B1 through the same per-camera baseline as PlumeRank** | **14 / 29 detected — no median** (2.51 realised on DEV) |
| **B2 through the same per-camera baseline as PlumeRank** | **10 / 29 detected — no median** (2.05 realised on DEV) |
| B1 as deployed (same ONNX, every frame) | 6 / 29 detected — no median (0.46 realised on DEV; threshold within 1e-6 of 1.0 — saturated probabilities) |
| B2 as deployed (frame differencing) | 3 / 29 detected — no median (1.84 realised on DEV) |

Source: `results/arm_dev_2026-10-04.json`. DEV is the arm the fusion weights were fit on — a
number from it is diagnostic, never a claim of generalisation.

**Failure cases — every DEV sequence, every system.** Minutes from ignition until the camera is
in the top 4 *and* has fired an alert; `miss` means it never did inside the window. From
`per_sequence` in `results/arm_dev_2026-10-04.json` — the same traces the aggregate is computed
from.

| Sequence | PlumeRank | B1 rebased | B2 rebased | B1 deployed | B2 deployed |
|---|---|---|---|---|---|
| `20240622_Border58Fire_om-w-mobo-c` | 8 | miss | miss | miss | miss |
| `20240625_DelMarFire_tdlln-n-mobo-c` | 38 | 5 | 0 | miss | miss |
| `20240701_McCainfire_ws-e-mobo-c` | miss | 29 | 7 | miss | miss |
| `20240724_GroveFire_tp-w-mobo-c` | 1 | miss | miss | miss | miss |
| `20240905_ValleyFire_cp-s-mobo-c` | 32 | miss | miss | miss | miss |
| `20240907_ChurchFire_ws-w-mobo-c` | miss | 14 | miss | miss | 29 |
| `20240907_FIRE_ws-s-mobo-c` | 19 | miss | miss | miss | miss |
| `20240908_BridgeFire_wilson-e-mobo-c` | 14 | miss | miss | miss | miss |
| `20241108_GardenFire_rm-s-mobo-c` | miss | 9 | miss | miss | miss |
| `20241113_Fire_dwpgm-s-mobo-c` | miss | 4 | 3 | 38 | miss |
| `20250107_PalisadesFire_69bravo-e-mobo-c` | 21 | miss | miss | miss | miss |
| `20250107_PalisadesFire_dwpgm-s-mobo-c` | miss | miss | miss | miss | miss |
| `20250123_Border2Fire_om-w-mobo-c` | miss | 11 | miss | 20 | miss |
| `20250123_GilmanFire_tdlln-s-mobo-c` | 34 | miss | miss | miss | miss |
| `20250514_PrescribedFire_hp-e-mobo-c` | 30 | miss | 3 | 21 | miss |
| `20250520_CrossingFire_hp-n-mobo-c` | 13 | 13 | 18 | miss | miss |
| `20250610_CahuillaFire_hp-n-mobo-c` | miss | 38 | 28 | miss | miss |
| `20250612_HighwayFire_hp-e-mobo-c` | 13 | 1 | 1 | miss | miss |
| `20250616_MissionFire_rm-s-mobo-c` | 12 | miss | 14 | miss | miss |
| `20250716_DaleBradfordFire_hp-n-mobo-c` | 7 | 19 | 1 | miss | miss |
| `20250724_PostaFire_ws-w-mobo-c` | miss | 14 | miss | miss | miss |
| `20251009_RanchoFire_bm-s-mobo-c` | miss | miss | miss | 11 | miss |
| `20251102_ScissorsFire_mp-n-mobo-c` | 8 | miss | miss | miss | 9 |
| `20260428_LiveOakFire_rm-s-mobo-c` | miss | 1 | miss | 3 | miss |
| `20260502_MissionFire_rm-s-mobo-c` | miss | miss | miss | miss | miss |
| `20260618_MissionFire_rm-s-mobo-c` | 17 | 12 | 0 | 24 | miss |
| `20260715_ThornFire_ws-n-mobo-c` | miss | miss | miss | miss | miss |
| `20260722_CreelmanFire_bm-s-mobo-c` | 9 | miss | miss | miss | miss |
| `20260820_ChiaFire_hp-n-mobo-c` | 11 | 23 | miss | miss | 28 |

What the rows say, plainly:

- **PlumeRank misses 12 of 29.** B1 through the same per-camera baseline catches 8 of
  those 12, so the two are partly complementary; PlumeRank's lead (17 vs 14) is a lead in
  coverage on this arm, not a dominance on every sequence, and B1-rebased is sometimes faster where
  both detect.
- **3 sequences are missed by all five systems** (`20250107_PalisadesFire_dwpgm-s-mobo-c`, `20260502_MissionFire_rm-s-mobo-c`, `20260715_ThornFire_ws-n-mobo-c`).
  These are the hardest cases on the arm and a starting point for error analysis.
- **DEV is in-sample.** The fusion weights were fit here; the held-out run is the test of whether
  any of this generalises.

**What is not measured yet, and why that is a decision, not an outage.** The held-out (HOLD)
evaluation — the only run whose number this project is willing to call a headline — has not
executed, so there is no `results/bench_*.json` and `GET /api/benchmark` returns `404` with an
explicit message. The live Wall meanwhile runs on the threshold the DEV run solved on the
fire-free confusion set — the only arm the threshold depends on, so it is the threshold the
held-out run will solve too — and labels it as development-calibrated on screen.
`scripts/bench.py`'s `_check_calibration` refuses to publish any threshold it could not bring
to the budget, from either side. Reproduce path once it exists: `make fetch && make bench` (≈95 minutes;
`make verify`, ≤3 minutes, is a sanity check against the micro-corpus only and does not
reproduce this number — see `DEMO.md`).

**Sensitivity design.** Because the reported ignition time is itself soft, the headline is
designed to be re-scored at every offset from −5 to +5 minutes from the same cached score
matrix in the same run, so the sensitivity curve cannot drift from the headline it describes
(`scripts/bench.py --sensitivity`).

**Testing, as a form of evaluation in its own right.** 664 tests passed, 2 skipped, 97%
coverage on `plumerank/`; 1,500 hypothesis-generated cases (`tests/test_invariants.py`)
verifying the fused hazard never leaves `[0, 1]` and the alert budget can never be overspent
under any input or call sequence; `scripts/check_claims.py` mechanically checks every route,
OpenCV symbol, dependency, Makefile target and file path every document names against the
real source, on every test run.

## 7. Limitations

See `README.md`'s "Honest limitations" section for the full list with evidence citations —
summarised: daylight-only (no IR hardware); **cloud, not fog, turned out to be the real
adversary**, and none of the four statistics is a good detector on its own (AUC 0.528 / 0.50 /
0.44 / — pre- vs post-ignition on DEV; S3, the one the project was built around, is
anti-predictive yet carries the largest fusion weight, and no S3-off ablation has run yet);
the per-camera baseline is what carries the ranking; the novelty is a metric reframe rather than a new sensor;
one instance with no HA; no managed-cloud-vision-API baseline; and no dispatch, suppression,
or upload path, by architecture rather than oversight.

## 8. Responsible-use considerations

**False positive:** rolls an engine to nothing, pulls crews from a real incident, and at
network scale creates the alarm fatigue that gets a whole system muted — that is how these
tools die in the field. **False negative:** a fire grows undetected.

Safeguards, each a build item rather than a paragraph (verified by `tests/test_api.py` and
`tests/test_rank.py`, not merely asserted):

1. A **ranking aid for a human watcher**, with no dispatch path anywhere in the codebase.
2. **Never hides or suppresses** a camera — the full roster stays on screen; the system only
   reorders it.
3. The **alert budget is published, logged and tunable**, with CloudWatch drift alarms.
4. Every ranked entry ships its **evidence crop**, so a human can overrule in one glance.
5. **No new imagery is collected** — the wall replays public HPWREN footage of wildland; the
   only frames written are small evidence crops and thumbnails of it, overwritten per replay.
6. A **persistent, non-dismissible limits strip** on every screen: daylight only; ranking aid,
   not dispatch; never suppresses a camera.

## 9. What's next

Error analysis on the three DEV sequences every system misses, and a test of whether
PlumeRank and B1-through-the-baseline combine (they catch partly different sequences); night/IR detection (needs infrared hardware this build does not have); live RTSP ingest
(replayed from disk today); a second camera network (Pyronear) to test whether the per-camera
baseline and the alert-budget calibration transfer across networks without refitting.

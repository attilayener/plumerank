# Architecture

One diagram, four routes, one box. See [`README.md`](README.md) for what the system does and
why; this page is how it is built.

```mermaid
flowchart TB
    subgraph SRC["Data -- fetched once, by committed manifest"]
        FIG["HPWREN FIgLib<br/>DEV 30 . HOLD 60 . CONFUSE 33 windows<br/>credit: hpwren.ucsd.edu"]
        PYR["Pyro-SDIS (Apache-2.0)<br/>29.5k train / 4.1k val<br/>the ONLY training data"]
    end

    S3B["S3 -- one bucket, one prefix<br/>corpus/"]

    subgraph CV["OpenCV 5 classical path -- plumerank/ -- ~98% of frames stop here"]
        STAB["stabilise.py<br/>phaseCorrelate + createHanningWindow<br/>+ phaseCorrelateIterative (new in 5.x)<br/>estimateAffine2D(USAC_MAGSAC)"]
        SEG["segment.py<br/>createBackgroundSubtractorMOG2<br/>morphologyEx . createCLAHE<br/>connectedComponentsWithStats"]
        STAT["statistic.py -- the custom primitive<br/>S1 base anchoring . S2 growth anisotropy<br/>S3 diffuse-boundary (Laplacian vs Sobel)<br/>S4 DISOpticalFlow drift, in-box"]
        RANK["rank.py<br/>fuse -> per-camera baseline<br/>(trailing 15 min) -> alert-budget<br/>threshold -> ranked watch list + evidence crop"]
    end

    ESC["escalate.py -- ~2% of frames<br/>dnn.readNetFromONNX(ENGINE_NEW)<br/>dnn.blobFromImage -> dnn.Net.forward<br/>one tile score, folded into the hazard"]

    ONNX[("tile_classifier.onnx<br/>96x96, trained on Pyro-SDIS tiles<br/>one model, two uses -- see below")]

    subgraph EVAL["Evaluation -- same box, offline"]
        BENCH["scripts/bench.py<br/>time-to-detection vs the ignition clock<br/>p50/p95, +-5 min sensitivity, ablation grid"]
        BASE["Two baselines, same budget<br/>B1: the SAME onnx over every frame<br/>B2: frame differencing"]
        RES[("results/ -- committed<br/>bench_<date>.json, verify_micro.json")]
    end

    subgraph BOX["One AWS t4g.small (Graviton) -- always on, judge-facing"]
        API["FastAPI api/main.py -- four routes<br/>GET /build-info . GET /health<br/>GET /api/watchlist (SSE)<br/>GET /api/benchmark"]
        WEB["Next.js static export -- three routes<br/>/ the Wall . /results . /judge"]
        CADDY["Caddy -- reverse proxy + static files"]
    end

    CW["CloudWatch -- the alert budget as a metric,<br/>with a drift alarm and a silence alarm"]

    FIG --> S3B --> STAB --> SEG --> STAT --> RANK
    RANK -->|"top-k, above threshold"| ESC --> RANK
    ONNX --> ESC
    ONNX --> BASE
    PYR -->|"training only -- never touches FIgLib"| ONNX
    RANK --> BENCH
    BASE --> BENCH --> RES --> API
    RANK --> API
    API --> CADDY
    WEB --> CADDY
    RANK --> CW
```

## The one model, two uses

`tile_classifier.onnx` is the **only** trained component. It is run through
`dnn.readNetFromONNX(ENGINE_NEW)` in two different places with two different jobs:

1. **`escalate.py`** -- scores the ~2% of frames the classical statistics flag, and that score
   folds into the fused hazard.
2. **Baseline B1** -- the *identical* model, run over a full-frame tile grid on **every** frame.

Because the model is the same on both sides, a difference between PlumeRank and B1 cannot be
explained by model size -- it isolates the value of the classical anchoring statistic and the
budget calibration. That is the controlled-experiment design the comparison rests on.

## Data flow, in order

**Offline, once** -- `scripts/fetch.py` pulls the corpus named by a committed, sha256-pinned
manifest (never the full archive) into `data/` and `s3://<bucket>/corpus/`.
`scripts/build_confusion_set.py` builds a non-fire confusion set from the same cameras at the
same time of day. `scripts/train_tile_classifier.py` trains the one ONNX model on Pyro-SDIS
**only** (a guard test asserts no FIgLib path ever enters the training manifest). `scripts/bench.py`
runs every arm and writes `results/bench_<date>.json`, which `GET /api/benchmark` then serves
byte-for-byte.

**Online, per replay tick** -- stabilise against the previous frame, segment, measure S1-S4,
fuse into a hazard, rebase that hazard against the camera's own trailing 15 minutes, escalate
the top-k candidates through the ONNX tile classifier, solve the alert-budget threshold, emit
the ranked list with one evidence crop per entry (drawn server-side with `cv2.line` /
`cv2.rectangle` / `cv2.putText`, written with `cv2.imwrite`), and stream it over SSE.

**There is no pre-rendered ranking anywhere on the judged path.** The background model warms
from the pre-ignition frames FIgLib supplies, on screen -- exactly what a real camera start
looks like. `GET /api/benchmark` 404s rather than inventing a number when no HOLD run exists
yet (`plumerank/rank.py`'s `_check_calibration`), and the live Wall shows that honestly instead
of a fabricated ranking.

## API endpoints -- four, all GET

| Path | Returns |
|---|---|
| `GET /build-info` | `cv2.getBuildInformation()` verbatim, `cv2.__version__`, a module-availability probe, the instance id and AMI id, the git commit sha |
| `GET /health` | liveness, uptime, replay state, the live-pipeline availability flag and reason |
| `GET /api/watchlist` | Server-Sent Events: a `meta` event (camera roster, budget, threshold), then one tick per minute |
| `GET /api/benchmark` | `results/bench_<date>.json` served verbatim, or `404` with an explicit message if it does not exist yet |

Evidence crops are static files served by Caddy directly -- image serving never touches the
Python process.

## Why OpenCV 5, not a smaller stack

Take OpenCV 5 out and this does not degrade -- it does not exist. Five modules would each need
a replacement: a background-modelling library for `createBackgroundSubtractorMOG2`, a
phase-correlation stabiliser for `phaseCorrelate`, a robust-estimation library for
`estimateAffine2D(USAC_MAGSAC)`, a dense optical-flow implementation for `DISOpticalFlow_create`,
and a second inference runtime for `dnn.readNetFromONNX`. See `README.md`'s "Why OpenCV 5"
table for the full 23-surface count, mechanically verified by `scripts/check_claims.py` on
every `make test` run.

## Deployment

One `t4g.small` (Graviton/ARM) EC2 instance, one S3 bucket for the corpus, CloudWatch for the
alert-budget metric and a drift/silence alarm, and an IAM instance profile carrying read-only
statements only. `deploy/bootstrap.sh` provisions it from a git bundle in S3;
`deploy/Caddyfile` reverse-proxies `/api/*`, `/health` and `/build-info` to Uvicorn on the same
origin as the static frontend, so there is no cross-origin request in production at all.

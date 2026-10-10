"""Four routes, one file: ``/build-info`` ``/health`` ``/api/watchlist`` ``/api/benchmark``.

What each one is for
--------------------
``/build-info`` is the highest-value route in the project and it ships before the
pipeline does. It turns an *asserted* OpenCV 5 integration into one a judge verifies with
a single ``curl``: the verbatim ``cv2.getBuildInformation()``, the version string, a probe
of every module this project actually calls into, the DNN engine the shipped model really
loaded under, the EC2 instance and AMI the box really is, the commit really checked out,
and the hash of the frozen constants. It must therefore answer on a box with no model, no
results file and no corpus -- a route that 500s when the pipeline is absent proves nothing
about the pipeline, and would be down at exactly the moment somebody is trying to find out
why. Every field is filled by a helper that returns a value *or* a null with a stated
reason; none of them can raise into the handler.

``/api/benchmark`` returns the committed ``results/bench_<date>.json`` **as bytes off the
disk**, with no computation and no re-serialisation. That is the whole point: the endpoint
and the repository cannot disagree about the headline, because they are the same bytes.

``/api/watchlist`` is the live judged path. It runs the real pipeline -- stabilise ->
segment -> statistic -> rank, with the top-2 above threshold escalated through the ONNX --
and streams the resulting watch list. When the corpus, the model or the calibrated
threshold is not on the box it serves the *designed* degraded state (an explicit
"live pipeline unavailable" on ``/health`` and an empty roster in ``meta``) and emits no
ticks at all. It never fabricates a row: a wall that invents its own ranking is worse than
a wall that says it cannot run one.

The replay loop is on-demand and process-wide
---------------------------------------------
One loop. Every viewer attaches to the same stream -- no session ids, no per-session
state, no cleanup to get wrong, and it is also the honest behaviour, because every judge
then sees the same wall rather than a private one. It starts on the first SSE connection
and stops ``REPLAY_IDLE_STOP_SECONDS`` after the last one disconnects.

That is a correctness requirement, not a convenience. ``t4g.small`` is burstable: 2 vCPU
at a 20% baseline, 24 credits earned per hour against a 576 ceiling. A wall that ran
continuously would sit above baseline every hour of an unattended 14-day judging window,
exhaust the balance and throttle the endpoint at the moment a judge opens it. So a loop
with no viewers does not compute a tick while it waits out its idle timer -- it sleeps.
Pausing the *work* rather than merely the *stream* is what makes the credit arithmetic
true.

Evidence crops are static files under ``web/evidence/`` and are served by Caddy. Image
serving never touches this process: one request per thumbnail against a single-worker
Uvicorn that is holding 62 warm background models would put file IO in contention with the
pipeline producing the files.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Iterable

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from plumerank.constants import (
    ALERTS_PER_CAMERA_PER_NIGHT,
    BUDGET_ALERTS_PER_CAMERA_NIGHT,
    BUDGET_DRIFT_TOLERANCE,
    HEARTBEAT_SECONDS,
    MINUTES_PER_FRAME,
    REPLAY_IDLE_STOP_SECONDS,
    SECONDS_PER_FRAME,
    TOP_K,
    WARMUP_FRAMES,
    WORK_HEIGHT,
    WORK_WIDTH,
    freeze_hash,
)
from plumerank.escalate import TileClassifier, should_escalate
from plumerank.metrics import alerts_per_camera_night, pin_opencv_threads
from plumerank.rank import Baseline, BudgetState, draw_evidence, emit, fuse
from plumerank.segment import Segmenter
from plumerank.stabilise import stabilise
from plumerank.statistic import Tracker, measure
from plumerank.telemetry import Telemetry, enabled_by_environment
from plumerank.types import CameraScore, Component, Sequence, Stats, Track, WatchEntry

LOG = logging.getLogger("plumerank.api")

# ---------------------------------------------------------------------------------
# Paths. Module-level and read at call time, so a test can point one at a tmp_path
# without a fixture reaching inside a handler.
# ---------------------------------------------------------------------------------

REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR: pathlib.Path = REPO_ROOT / "data"
#: The frame root for a manifest that names none. A manifest's own ``local_root`` wins
#: (CONFUSE lives under ``corpus/mta``) -- the same rule as ``bench.py:_frame_root``.
CORPUS_DIR: pathlib.Path = DATA_DIR / "corpus" / "figlib"
#: Sequences dropped from every arm (night frames, whiteouts...). The wall drops them too:
#: a daylight-only system must not replay the night window the evaluation excluded.
EXCLUSIONS_FILE: str = "excluded.json"
RESULTS_DIR: pathlib.Path = REPO_ROOT / "results"
MODELS_DIR: pathlib.Path = REPO_ROOT / "models"
EVIDENCE_DIR: pathlib.Path = REPO_ROOT / "web" / "evidence"

ONNX_PATH: pathlib.Path = MODELS_DIR / "tile_classifier.onnx"

#: The arms the wall is allowed to replay, and the one it is not. HOLD is read exactly
#: once, by ``bench.py``, after the constants freeze -- ``test_hold_untouched_until_freeze``
#: has no exemption, "including for the demo wall", so the exclusion is enforced here in
#: code rather than trusted to whoever writes the next manifest.
WALL_MANIFESTS: tuple[str, ...] = ("manifest_dev.json", "manifest_confuse.json")
HOLD_MANIFEST: str = "manifest_hold.json"

#: Where working-resolution frames are cached, or None for no disk cache. systemd's
#: ``CacheDirectory=plumerank`` exports ``CACHE_DIRECTORY``; ``PLUMERANK_FRAME_CACHE``
#: overrides it. Read at call time so a test can point it at a tmp dir.
FRAME_CACHE_DIR: pathlib.Path | None = (
    pathlib.Path(os.environ["PLUMERANK_FRAME_CACHE"])
    if os.environ.get("PLUMERANK_FRAME_CACHE")
    else pathlib.Path(os.environ["CACHE_DIRECTORY"].split(":")[0]) / "frames"
    if os.environ.get("CACHE_DIRECTORY")
    else None
)

#: Worker threads for a tick: frame loading, and stabilise/segment/track per camera. Both
#: release the GIL in OpenCV and touch only per-camera state; measure() shares
#: statistic.py's per-shape DIS objects and stays on the tick thread. Two = the t4g's vCPUs.
LOAD_WORKERS: int = 2

#: Evidence directories are named by replay id (``%Y%m%dT%H%M%SZ``); startup prunes every
#: one but the current process's, and touches nothing else under EVIDENCE_DIR.
_REPLAY_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")

# ---------------------------------------------------------------------------------
# Pacing -- deliberately NOT in constants.py
# ---------------------------------------------------------------------------------
# constants.py is what results/constants.sha256 freezes, and the freeze is a claim about
# the numbers that can move a score. Neither number below can: change both and every
# hazard, every rank and every published figure is byte-identical. Only the wall clock
# moves. Putting them in the frozen surface would dilute what "frozen" means, which is the
# same argument rank.py makes for keeping its font size out of it.

#: FIgLib publishes one frame per minute, so one tick is one simulated minute. At one
#: second per tick a whole 81-frame camera-night plays in 81 seconds -- long enough that a
#: judge watches the background model warm up and the ranking reorder, short enough that
#: the whole mechanism fits inside a demo.
REPLAY_TICK_SECONDS: float = 1.0

#: How often a viewer-less loop wakes to check whether anyone came back. Short against
#: REPLAY_IDLE_STOP_SECONDS so a returning judge waits at most this long, and cheap because
#: waking to look at a set is not waking to decode 40 JPEGs.
IDLE_POLL_SECONDS: float = 0.25

#: Ticks a slow client may fall behind before it starts losing them. A wall is a live
#: instrument: showing a judge a stale ranking because their laptop was busy is worse than
#: skipping to the current one, and an unbounded queue would let one stalled reader hold
#: the whole process's memory.
VIEWER_QUEUE_DEPTH: int = 4

#: Concurrent live walls this process will hold open. Each is one SSE connection held for
#: as long as the tab is open, and all of them share the one replay, so a cap costs no
#: judge anything except the 33rd simultaneous one -- who gets a 503 with Retry-After
#: rather than a box whose sockets ran out. Kept under the systemd unit's
#: --limit-concurrency (96) so /health and /build-info always have connections to spare.
#: ``PLUMERANK_MAX_VIEWERS`` overrides it without a redeploy of the code.
MAX_VIEWERS: int = int(os.environ.get("PLUMERANK_MAX_VIEWERS", "32"))

#: Wall tile thumbnails (Replay._write_thumbnails): width in px and JPEG quality. ~7 KB
#: and ~1 ms to encode each from the 480x270 working frame (measured on an M-series laptop).
THUMB_WIDTH: int = 320
THUMB_QUALITY: int = 70

#: Concurrent walls one client address may hold. The global cap alone let one script
#: holding 32 streams lock every judge out; four covers a judge with a few tabs open.
MAX_VIEWERS_PER_CLIENT: int = int(os.environ.get("PLUMERANK_MAX_VIEWERS_PER_CLIENT", "4"))

#: How often an otherwise silent stream sends an SSE comment. Separate from
#: IDLE_POLL_SECONDS, which is how often the generator wakes to notice a disconnect: tying
#: the two sent four comment frames a second and kept phone radios awake.
KEEPALIVE_SECONDS: float = 15.0

#: A tick that raises is skipped, not fatal: the loop retries with exponential backoff
#: (base x 2^(n-1), capped), and only after this many consecutive failures does the wall
#: enter its failed state -- in which the heartbeat stops, so the silence alarm (and its
#: reboot action, deploy/alarms.sh) can recover a box that cannot recover itself.
MAX_TICK_FAILURES: int = 5
TICK_BACKOFF_BASE_SECONDS: float = 1.0
TICK_BACKOFF_CAP_SECONDS: float = 30.0

#: Open walls per client address (see client_address). Module state, like REPLAY.
_CLIENT_VIEWERS: collections.Counter[str] = collections.Counter()


# ---------------------------------------------------------------------------------
# /build-info -- every field is a helper that cannot raise into the handler
# ---------------------------------------------------------------------------------


def _probe(names: Iterable[str], root: Any) -> dict[str, bool]:
    """Presence of each symbol on ``root``. The unit ``/build-info`` reports modules in."""
    return {name: hasattr(root, name) for name in names}


def _module_probes() -> dict[str, dict[str, Any]]:
    """Which OpenCV modules this project's real call sites need, and whether they are here.

    The symbols are the ones actually called, not the module names: an ``import cv2`` that
    succeeds says nothing about whether the contrib build carries ``bgsegm``, and a headless
    wheel that silently lacks ``phaseCorrelateIterative`` would fail four stages downstream
    as a wrong number rather than here as a missing attribute. ``ok`` is the AND, so a judge
    reads one boolean per module and the detail only if it is False.
    """
    bgsegm = getattr(cv2, "bgsegm", None)
    groups: dict[str, tuple[Any, tuple[str, ...]]] = {
        # phaseCorrelate and createHanningWindow are the stabiliser's primary path; the
        # optical-flow pair is its fallback and S4's instrument.
        "video": (
            cv2,
            (
                "phaseCorrelate",
                "phaseCorrelateIterative",
                "createHanningWindow",
                "calcOpticalFlowPyrLK",
                "DISOpticalFlow_create",
                "createBackgroundSubtractorMOG2",
            ),
        ),
        "bgsegm": (
            bgsegm,
            ("createBackgroundSubtractorLSBP", "createBackgroundSubtractorGSOC"),
        ),
        "dnn": (cv2.dnn, ("readNetFromONNX", "blobFromImage", "ENGINE_NEW", "ENGINE_CLASSIC")),
        # 5.x moved the estimator here; USAC_MAGSAC is the method the fallback names.
        "geometry": (cv2, ("estimateAffine2D", "USAC_MAGSAC")),
        "features": (cv2, ("goodFeaturesToTrack",)),
    }

    out: dict[str, dict[str, Any]] = {}
    for module, (root, symbols) in groups.items():
        if root is None:
            out[module] = {"ok": False, "symbols": {s: False for s in symbols}}
            continue
        found = _probe(symbols, root)
        out[module] = {"ok": all(found.values()), "symbols": found}
    return out


def _repo_relative(path: pathlib.Path) -> str:
    """A path as the repository sees it, or absolute when it lies outside the tree.

    The relative form is what a judge compares against a file in their clone. The fallback
    exists because ``pathlib.relative_to`` raises on a path outside the root, and a raise
    inside ``/build-info`` would take down the one route that has to answer on any box.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: pathlib.Path) -> str | None:
    """sha256 of a file, or None when it is not there. Never raises."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        LOG.info("sha256 unavailable for %s: %s", path, exc)
        return None


def _dnn_engine() -> dict[str, Any]:
    """The engine the shipped ONNX **actually loaded under**, not the one we asked for.

    ``engine_selected`` is filled only after a real ``readNetFromONNX`` returned, because
    "we passed ENGINE_NEW" and "ENGINE_NEW ran this model" are different claims and this
    route exists to publish the second one. A missing model is reported as a missing model:
    ENGINE_CLASSIC covers roughly a fifth of the ONNX spec that ENGINE_NEW does, so a quiet
    downgrade would be a materially different system described by the same JSON.
    """
    info: dict[str, Any] = {
        "engine_requested": "ENGINE_NEW",
        "engine_requested_value": int(cv2.dnn.ENGINE_NEW),
        "engine_selected": None,
        "onnx_path": _repo_relative(ONNX_PATH),
        "onnx_present": ONNX_PATH.is_file(),
        "onnx_sha256": _sha256(ONNX_PATH),
        "loaded": False,
        "detail": None,
    }
    if not info["onnx_present"]:
        info["detail"] = f"{info['onnx_path']} is not on this box"
        return info
    try:
        net = TileClassifier(ONNX_PATH)
    except Exception as exc:  # a build-info route that raises is a build-info route that lies
        info["detail"] = f"{type(exc).__name__}: {exc}"
        return info
    info["loaded"] = True
    info["engine_selected"] = "ENGINE_NEW" if net.engine == int(cv2.dnn.ENGINE_NEW) else str(net.engine)
    return info


def _imds(path: str, headers: dict[str, str], method: str = "GET") -> str | None:
    """One IMDSv2 call against the link-local endpoint, or None. Never blocks past 0.2 s.

    The opener carries an empty ProxyHandler on purpose: ``urllib`` honours ``http_proxy``
    from the environment, and a box with a proxy set would send 169.254.169.254 to that
    proxy instead -- which is how a 0.2 s timeout becomes a 30 s hang against a machine
    that is answering, just not about EC2 metadata.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        f"http://169.254.169.254{path}", headers=headers, method=method
    )
    try:
        with opener.open(request, timeout=0.2) as response:
            return response.read(256).decode("ascii", "replace").strip()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        LOG.debug("IMDS %s %s: %s", method, path, exc)
        return None


_IDENTITY_CACHE: dict[str, str | None] | None = None


def _instance_identity() -> dict[str, str | None]:
    """``instance_id`` and ``ami_id`` off IMDSv2, or nulls off-instance.

    Exactly those two spellings, at the top level of the response: ``deploy/launch.md`` section 7
    tells the verifier to curl this route and compare ``ami_id`` against the pinned AMI, and
    ``deploy/bootstrap.sh`` greps for it. ``amiId`` would break a written verification step
    silently.

    Cached for the life of the process because neither value can change under a running
    instance, and ``/health``-adjacent routes get polled daily for a fortnight -- an
    uncached probe would pay three link-local round trips every time, and pay them as a 0.6 s
    stall on a laptop where all three time out.
    """
    global _IDENTITY_CACHE
    if _IDENTITY_CACHE is not None:
        return _IDENTITY_CACHE

    identity: dict[str, str | None] = {"instance_id": None, "ami_id": None}
    token = _imds(
        "/latest/api/token",
        {"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        method="PUT",
    )
    if token:
        headers = {"X-aws-ec2-metadata-token": token}
        identity["instance_id"] = _imds("/latest/meta-data/instance-id", headers)
        identity["ami_id"] = _imds("/latest/meta-data/ami-id", headers)
    _IDENTITY_CACHE = identity
    return identity


def _git_head_from_files() -> str | None:
    """HEAD read straight out of ``.git``, for when the ``git`` binary will not answer.

    This is not belt and braces. The service runs as an unprivileged user against a
    checkout that ``bootstrap.sh`` created as root, and modern git refuses ``rev-parse`` in
    a repository it considers of dubious ownership -- so the subprocess path fails on
    exactly the box whose commit a judge is checking. Packed refs are handled because a
    fresh clone keeps most refs in ``packed-refs`` and nowhere else.
    """
    git_dir = REPO_ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head or None

    ref = head.split(":", 1)[1].strip()
    with contextlib.suppress(OSError):
        return (git_dir / ref).read_text(encoding="ascii").strip() or None
    with contextlib.suppress(OSError):
        for line in (git_dir / "packed-refs").read_text(encoding="ascii").splitlines():
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return None


_COMMIT_CACHE: str | None = None


def _git_commit() -> str | None:
    """The checked-out commit sha, or None. Cached: a running process cannot change it."""
    global _COMMIT_CACHE
    if _COMMIT_CACHE is not None:
        return _COMMIT_CACHE
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if done.returncode == 0 and done.stdout.strip():
            _COMMIT_CACHE = done.stdout.strip()
            return _COMMIT_CACHE
        LOG.info("git rev-parse returned %s: %s", done.returncode, done.stderr.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.info("git rev-parse unavailable: %s", exc)
    _COMMIT_CACHE = _git_head_from_files()
    return _COMMIT_CACHE


def _constants_freeze() -> dict[str, Any]:
    """The live constants hash beside the committed one, and whether they agree.

    Publishing only ``freeze_hash()`` would let a judge check that the file hashes to
    something, not that it hashes to the thing ``results/constants.sha256`` recorded before
    HOLD was ever read. ``matches`` is the claim; the two hashes are the evidence for it.
    """
    live = freeze_hash()
    committed_path = RESULTS_DIR / "constants.sha256"
    committed: str | None = None
    with contextlib.suppress(OSError):
        committed = committed_path.read_text(encoding="ascii").split()[0].strip()
    return {
        "sha256": live,
        "committed_sha256": committed,
        "matches": None if committed is None else committed == live,
    }


# ---------------------------------------------------------------------------------
# The replay corpus
# ---------------------------------------------------------------------------------


def _load_manifest(path: pathlib.Path) -> dict[str, Any] | None:
    """One committed manifest, or None when it is absent. A malformed one raises."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_root(manifest: dict[str, Any]) -> pathlib.Path:
    """Where a manifest's frames live: its ``local_root`` under data/, else CORPUS_DIR.

    Hard-coding ``corpus/figlib`` silently skipped all 33 CONFUSE windows (they live under
    ``corpus/mta``), so the wall replayed fire footage only and its alert rate could not be
    compared with the budget it was calibrated to on fire-free windows.
    """
    local_root = manifest.get("local_root")
    if not local_root:
        return CORPUS_DIR
    return DATA_DIR.joinpath(*str(local_root).split("/"))


def _excluded_ids() -> set[str]:
    """Sequence ids in data/excluded.json (any arm). A missing file excludes nothing."""
    blob = _load_manifest(DATA_DIR / EXCLUSIONS_FILE)
    if not blob:
        return set()
    records = blob.get("excluded", blob) if isinstance(blob, dict) else blob
    if isinstance(records, dict):
        return {str(k) for k in records}
    out: set[str] = set()
    for item in records if isinstance(records, list) else []:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, dict) and "sequence_id" in item:
            out.add(str(item["sequence_id"]))
    return out


def _wall_sequences() -> tuple[list[Sequence], str | None]:
    """Every replayable window whose frames are actually on this disk, and the credit line.

    Windows come from the DEV and CONFUSE manifests only, and any sequence id that also
    appears in the HOLD manifest is refused outright. HOLD is read once, by ``bench.py``,
    after the freeze; ``test_hold_untouched_until_freeze`` grants no exemption "including
    for the demo wall", and a rule with one exemption is a rule a judge has to take on
    trust. Enforcing it here means a mis-edited manifest cannot quietly put a held-out
    sequence on screen.

    Sequences in ``data/excluded.json`` are dropped exactly as the evaluation drops them.

    A window with any frame missing from disk is skipped rather than truncated: a partly
    fetched sequence would replay a shorter camera-night, and every rate in this project is
    per camera-night.
    """
    hold_ids: set[str] = set()
    hold = _load_manifest(DATA_DIR / HOLD_MANIFEST)
    if hold:
        hold_ids = {w["sequence_id"] for w in hold.get("windows", [])}
    excluded = _excluded_ids()

    sequences: list[Sequence] = []
    credit: str | None = None
    for name in WALL_MANIFESTS:
        manifest = _load_manifest(DATA_DIR / name)
        if manifest is None:
            continue
        credit = credit or manifest.get("credit")
        arm = str(manifest.get("arm", ""))
        root = _frame_root(manifest)
        for window in manifest.get("windows", []):
            sequence_id = str(window["sequence_id"])
            if sequence_id in hold_ids:
                LOG.error("refusing HOLD sequence %s on the wall", sequence_id)
                continue
            if sequence_id in excluded:
                continue
            paths = [str(root / sequence_id / frame) for frame in window["frames"]]
            if not paths or not all(pathlib.Path(p).is_file() for p in paths):
                continue
            sequences.append(
                Sequence(
                    sequence_id=sequence_id,
                    camera_id=str(window["camera_id"]),
                    arm=arm,
                    frame_paths=paths,
                    t0_index=window.get("t0_index"),
                )
            )
    return sequences, credit


def _decode_working_frame(path: str) -> np.ndarray:
    """One corpus JPEG at working resolution, BGR -- the definition every cache must match.

    INTER_AREA because the source is 3072x2048 and this is a >6x downscale: bilinear would
    alias a thin plume edge into or out of existence, and S3 is a statistic about that edge.
    """
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"could not decode {path} -- the corpus on this box is damaged")
    return cv2.resize(frame, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA)


def _cache_path(path: str) -> pathlib.Path | None:
    """The cache file for one source frame, keyed on its path, size, mtime and the working
    size -- so a re-fetched or replaced JPEG can never be served from a stale entry."""
    if FRAME_CACHE_DIR is None:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    ident = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{WORK_WIDTH}x{WORK_HEIGHT}"
    key = hashlib.sha256(ident.encode("utf-8")).hexdigest()
    return FRAME_CACHE_DIR / key[:2] / f"{key}.npy"


def _read_working_frame(path: str) -> np.ndarray:
    """``_decode_working_frame``, through a lossless on-disk cache when one is configured.

    Decoding a 3072x2048 JPEG is most of a tick: 60-odd of them a second is more than one
    Graviton core can do, and the wall ran ~2.5x slower than the clock it printed. The
    cache stores the decoded, resized array as ``.npy`` -- byte-identical to what
    ``_decode_working_frame`` returns, so the pipeline sees exactly the pixels bench.py
    sees -- and is filled on first read. Any cache failure falls back to decoding.
    """
    cached = _cache_path(path)
    if cached is not None:
        try:
            arr = np.load(cached, allow_pickle=False)
        except (OSError, ValueError, EOFError):  # EOFError: a zero-length entry (crash mid-write)
            arr = None
        if (
            arr is not None
            and arr.shape == (WORK_HEIGHT, WORK_WIDTH, 3)
            and arr.dtype == np.uint8
        ):
            return arr
    frame = _decode_working_frame(path)
    if cached is not None:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=cached.parent, suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, frame, allow_pickle=False)
            os.replace(tmp, cached)
        except OSError as exc:
            LOG.warning("frame cache write failed for %s: %s", path, exc)
    return frame


def _benchmark_file() -> pathlib.Path | None:
    """The newest committed ``results/bench_<date>.json``, or None.

    Newest by name, which is newest by date: the filenames are ISO dates, so a lexical sort
    is a chronological one and no mtime is involved -- a fresh ``git clone`` rewrites every
    mtime to the checkout time, and picking the headline by mtime would make the served
    number depend on the order git happened to write the files.

    The name is matched exactly (``_HOLD_NAME``), as ``_DEV_SHIPPED`` does for the DEV arm:
    a bare ``bench_*.json`` glob would let ``bench_<date>_s4off.json`` sort after
    ``bench_<date>.json`` and calibrate the wall on an ablation.
    """
    if not RESULTS_DIR.is_dir():
        return None
    candidates = sorted(p for p in RESULTS_DIR.glob("bench_*.json") if _HOLD_NAME.match(p.name))
    return candidates[-1] if candidates else None


#: The held-out run's filename, exactly -- bench.py writes no other name under it.
_HOLD_NAME = re.compile(r"^bench_\d{4}-\d{2}-\d{2}\.json$")


def _heldout_rejection(payload: dict[str, Any]) -> str | None:
    """Why the held-out payload cannot calibrate THIS wall, or None when it can.

    The checks ``_development_rejection`` already applies to the fallback, which the
    held-out path skipped: above all the constants freeze. After any edit to
    ``constants.py`` -- a comment is enough -- the wall would otherwise keep serving the
    held-out threshold, labelled ``held-out``, under constants it was not solved for.
    """
    if payload.get("arm") != "hold":
        return f"arm is {payload.get('arm')!r}, not 'hold'"
    calibration = payload.get("calibration")
    if not (isinstance(calibration, dict) and calibration.get("realised_on") == "confuse"):
        return "its threshold was not solved on the confusion set"
    run = payload.get("run")
    recorded = run.get("constants_sha256") if isinstance(run, dict) else None
    if recorded != freeze_hash():
        return (
            f"solved under constants {str(recorded)[:12]}, running {freeze_hash()[:12]} "
            "-- revert plumerank/constants.py; do NOT re-run HOLD"
        )
    return None


#: The shipped DEV arm's filename -- no cell suffix, so no ablation can occupy it
#: (bench.results_filename). Its calibration block is solved on CONFUSE, the same
#: fire-free confusion set, by the same code, as the held-out run's will be.
_DEV_SHIPPED = re.compile(r"^arm_dev_\d{4}-\d{2}-\d{2}\.json$")


#: Why each shipped DEV file was passed over on the last ``_development_file`` scan,
#: newest first -- the probe's reason string when none is usable.
_DEV_REJECTIONS: list[str] = []


def _development_rejection(path: pathlib.Path) -> str | None:
    """Why this DEV payload cannot calibrate the wall, or None when it can."""
    try:
        dev = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable ({type(exc).__name__})"
    if not isinstance(dev, dict):
        return "not a JSON object"
    calibration = dev.get("calibration")
    if not (isinstance(calibration, dict) and calibration.get("realised_on") == "confuse"):
        return "its threshold was not solved on the confusion set"
    if not (isinstance(dev.get("baseline"), dict) and dev["baseline"].get("shipped")):
        return "not the shipped configuration"
    if dev.get("background") != "mog2" or (
        dev.get("budget_alerts_per_camera_night") != BUDGET_ALERTS_PER_CAMERA_NIGHT
    ):
        return "solved for another background or budget than the wall runs"
    run = dev.get("run")
    recorded = run.get("constants_sha256") if isinstance(run, dict) else None
    if recorded != freeze_hash():
        return (
            "development run predates the running constants "
            f"({str(recorded)[:12]} vs {freeze_hash()[:12]}) -- re-run the DEV bench"
        )
    return None


def _development_file() -> pathlib.Path | None:
    """The newest usable shipped-configuration DEV payload, or None.

    Newest first, and an unusable file falls through to the next older one rather than
    taking the wall down: a same-day ablation or a stale run must not cost the demo a
    threshold an older, valid run already solved. Rejections are kept in
    ``_DEV_REJECTIONS`` so the probe can say why when none is usable.
    """
    _DEV_REJECTIONS.clear()
    if not RESULTS_DIR.is_dir():
        return None
    candidates = sorted(
        (p for p in RESULTS_DIR.glob("arm_dev_*.json") if _DEV_SHIPPED.match(p.name)),
        reverse=True,
    )
    for path in candidates:
        why = _development_rejection(path)
        if why is None:
            return path
        LOG.warning("development file %s not used: %s", path.name, why)
        _DEV_REJECTIONS.append(f"{path.name}: {why}")
    return None


def calibration_kind(source: str | None) -> str | None:
    """``held-out`` or ``development``: which committed run the wall's threshold came from."""
    if source is None:
        return None
    return "held-out" if "bench_" in source else "development"


def _calibrated_threshold() -> tuple[float | None, str | None]:
    """The solved hazard threshold and where it was read from.

    The threshold is *solved*, not chosen: ``rank.solve_threshold`` bisects it on the CONFUSE
    arm and ``bench.py`` commits it. So it is read out of the committed benchmark and nowhere
    else. There is deliberately no default -- a wall running on a made-up threshold would
    show a ranking calibrated to nothing while looking exactly like one calibrated to 2.0
    alerts per camera-night, which is the most expensive plausible wrong answer this file
    could produce. Absent means the live pipeline is unavailable, and says so.
    """
    path = _benchmark_file()
    held_out = path is not None
    if path is None:
        # Before the held-out run exists, the shipped DEV run's threshold is used -- and
        # labelled as such (calibration_kind). It is not a default and not chosen: it is
        # solved on the fire-free CONFUSE arm exactly as the held-out run's will be, which
        # is the only arm the threshold ever depends on. DEV's own detection numbers are
        # in-sample and are never presented as the headline.
        path = _development_file()
        if path is None:
            return None, None
    relative = _repo_relative(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.error("benchmark file %s is unreadable: %s", path, exc)
        # A source, not None: a committed file that cannot be read must reach the probe's
        # "is committed but cannot calibrate" branch, never "run `make bench`" -- that
        # sentence tells an operator to re-read HOLD.
        return None, relative
    # The wall runs MOG2 at the frozen budget, so a threshold solved under anything else
    # would calibrate it to an operating point it does not run at. bench.py refuses to
    # write such a HOLD file; this is the second lock on the same door.
    background = payload.get("background")
    budget = payload.get("budget_alerts_per_camera_night")
    if background not in (None, "mog2") or (
        budget is not None and budget != BUDGET_ALERTS_PER_CAMERA_NIGHT
    ):
        LOG.error(
            "benchmark file %s was solved for bg=%r budget=%r, not the wall's mog2 at %s",
            path, background, budget, BUDGET_ALERTS_PER_CAMERA_NIGHT,
        )
        return None, relative
    # Two spellings and no more. Both are named here rather than discovered by walking the
    # document, because a search that finds *a* number called "threshold" somewhere in a
    # nested ablation grid would calibrate the wall on an ablation cell.
    calibration = payload.get("calibration")
    if isinstance(calibration, dict) and "threshold" in calibration:
        source = f"{relative}:calibration.threshold"
        value = _finite_threshold(calibration["threshold"], source)
    elif "threshold" in payload:
        source = f"{relative}:threshold"
        value = _finite_threshold(payload["threshold"], source)
    else:
        return None, relative
    if value is None:
        return None, source
    # Usable number, then provenance: the held-out file must also be the HOLD arm, solved on
    # CONFUSE, under the constants this process is running.
    if held_out:
        why = _heldout_rejection(payload)
        if why is not None:
            LOG.error("benchmark file %s refused as the wall's calibration: %s", path, why)
            return None, relative
    return value, source


def _finite_threshold(value: Any, source: str) -> float | None:
    """A threshold read off disk, or None when the file does not carry a usable number.

    ``float()`` raises on a JSON ``null``, an object or a list, and the only caller is
    ``probe()`` -- so an unguarded conversion turns a slightly wrong results file into a 500
    on ``/health``, which is the one route whose job is to *report* a box that cannot run.

    Non-finite is refused for a second reason: ``json.loads`` accepts a bare ``NaN``, and a
    NaN threshold compares False against every hazard, so the wall would run all night
    ranking cameras that can never alert while ``meta`` claimed a calibrated operating point.
    Absent is a state the UI has a design for; silently-nonsense is not.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        LOG.error("threshold at %s is not a number: %r", source, value)
        return None
    if not math.isfinite(number):
        LOG.error("threshold at %s is not finite: %r", source, value)
        return None
    return number


# ---------------------------------------------------------------------------------
# One camera's state for the length of one replay
# ---------------------------------------------------------------------------------


class _Camera:
    """Per-camera pipeline state. One background model, one tracker, two frames of memory."""

    __slots__ = (
        "sequence", "frame_offset", "key", "segmenter", "tracker", "prev_grey", "prev_work"
    )

    def __init__(self, sequence: Sequence, frame_offset: int = 0, key: str | None = None) -> None:
        self.sequence = sequence
        #: The roster key. DEV and CONFUSE both carry windows from the same physical camera
        #: on different nights, and the budget and baseline are keyed per roster slot -- so a
        #: repeated camera id gets a ``#n`` suffix rather than sharing (and corrupting) one
        #: slot. bench.py keys by sequence id for the same reason. The suffix is "~n" so the
        #: key stays safe as a filename and URL segment.
        self.key = key or sequence.camera_id
        #: Frames skipped off the head of this window so that its own ignition lands on the
        #: same replay index as every other camera's. See ``Replay._align``.
        self.frame_offset = frame_offset
        self.segmenter = Segmenter()
        self.tracker = Tracker()
        self.prev_grey: np.ndarray | None = None
        self.prev_work: np.ndarray | None = None

    @property
    def camera_id(self) -> str:
        return self.key

    def reset(self) -> None:
        """Start a fresh camera-night. The warm-up is re-run, on screen, on purpose.

        A judge who leaves the tab and comes back sees the background model warming again,
        which is a designed state with its own caption -- and is exactly what a real camera
        start looks like. Carrying a warm model across replays would hide the one part of
        the mechanism that is honest about needing history.
        """
        self.segmenter.reset()
        self.tracker = Tracker()
        self.prev_grey = None
        self.prev_work = None


def _best_track(components: list[Component], tracks: list[Track]) -> tuple[Component, Track] | None:
    """The component this camera is scored on this minute: the largest one.

    ``Segmenter.apply`` returns largest first, so this is index 0 -- written as a named
    choice rather than an index because it *is* a choice. One component per camera per
    minute is what makes the watch list a list of cameras rather than a list of blobs, and
    the largest is the one a dispatcher's eye goes to in the thumbnail.
    """
    if not components:
        return None
    return components[0], tracks[0]


def _align(sequences: list[Sequence]) -> tuple[list[_Camera], int | None, int]:
    """Put every camera's own ignition on the same replay index, and say which index that is.

    The wall is one clock over windows recorded on different nights, and the web Clock
    renders a single "T+n min since ignition" from ``meta.t0_index`` and the tick's frame
    index. FIgLib's convention is 40 frames of lead, but 12 of the 30 DEV windows are one
    frame short of it and carry ``t0_index: 39`` -- so replaying every window from its own
    frame 0 against one published ``t0_index`` puts that clock a whole minute out for 40% of
    the roster, on a wall whose headline is a number of minutes. The offset is real data, it
    is in the manifest, and the fix is to use it.

    The shared index is the **smallest** lead on the roster, because a camera cannot be
    given frames in front of the ones it has: every other window drops that many frames off
    its head instead. A window with no ignition at all (the CONFUSE arm, ``t0_index is
    None``) has nothing to align and starts at its own frame 0.

    Returns:
        The cameras with their offsets, the shared ignition index (``None`` when no window
        on the roster ignites), and the number of replayable frames -- the shortest window
        *after* its offset, so no camera runs off the end of its own list.
    """
    leads = [s.t0_index for s in sequences if s.t0_index is not None]
    shared = min(leads) if leads else None

    totals = collections.Counter(s.camera_id for s in sequences)
    seen: collections.Counter[str] = collections.Counter()
    cameras: list[_Camera] = []
    for sequence in sequences:
        offset = 0 if sequence.t0_index is None or shared is None else sequence.t0_index - shared
        seen[sequence.camera_id] += 1
        key = sequence.camera_id
        if totals[key] > 1:
            # "~", not "#": the key is also a filename and a URL path segment (evidence
            # crops, thumbnails), and "#" would start a URL fragment.
            key = f"{key}~{seen[sequence.camera_id]}"
        cameras.append(_Camera(sequence, frame_offset=offset, key=key))

    # The common length, computed exactly as bench.py's unify_windows computes it: the
    # shortest fire window after its own t0 anchor, and no longer than the shortest window
    # of any kind. Not FRAMES_PER_WINDOW -- the published rate stays in the canonical unit
    # anyway, because ``alerts_per_camera_night`` takes the window length and rescales.
    frame_count = min(len(c.sequence.frame_paths) - c.frame_offset for c in cameras)
    frame_count = min(frame_count, min(len(c.sequence.frame_paths) for c in cameras))
    # Windows with no ignition (CONFUSE) are CENTRE-trimmed to that length, as unify_windows
    # does: the threshold is calibrated on frames 1..79 of an 81-frame CONFUSE window, and
    # replaying frames 0..78 instead warmed every background model on a different minute
    # and moved the wall's fire-free alert rate off the calibration (1.68 vs 1.99).
    for camera in cameras:
        if camera.sequence.t0_index is None or shared is None:
            camera.frame_offset = (len(camera.sequence.frame_paths) - frame_count) // 2
    return cameras, shared, frame_count


# ---------------------------------------------------------------------------------
# The replay loop
# ---------------------------------------------------------------------------------


class Replay:
    """The one process-wide replay loop and the viewers attached to it.

    Public state is read by ``/health`` and by the ``meta`` event; everything else is
    private to the loop task. There is exactly one instance (``REPLAY``), which is why
    ``deploy/plumerank.service`` runs Uvicorn with ``--workers 1``: a second worker would
    give the next judge to connect a second wall, two sets of 62 background models
    competing for 2 GiB, and the burstable CPU budget spent twice.
    """

    def __init__(self) -> None:
        # Read from constants here rather than referenced there, so a test can shorten the
        # idle window without patching the frozen file -- it is the same loop either way.
        self.idle_stop_seconds: float = REPLAY_IDLE_STOP_SECONDS
        self.tick_seconds: float = REPLAY_TICK_SECONDS
        self.replay_id: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        self._viewers: set[asyncio.Queue[str]] = set()
        self._task: asyncio.Task[None] | None = None

        self._cameras: list[_Camera] = []
        self._threshold: float | None = None
        self._threshold_source: str | None = None
        self._classifier: TileClassifier | None = None
        self._credit: str | None = None
        self._frame_count: int = 0
        self._t0_index: int | None = None
        self._unavailable: str | None = "not probed yet"
        #: Set when MAX_TICK_FAILURES consecutive ticks raised. Unlike ``_unavailable`` it
        #: does not latch: the next viewer to attach clears it and re-probes the box.
        self._failed: str | None = None
        self._tick_failures: int = 0
        self._probed: bool = False
        self._probe_task: asyncio.Task[None] | None = None

        self._frame_index: int = 0
        self._ticks: int = 0
        self._alerts_uncapped: int = 0
        self._baseline: Baseline = Baseline()
        self._invocations_at_night_start: int = 0
        self._last_tick_iso: str | None = None
        self._last_payload: str | None = None
        #: Monotonic for the life of the process and deliberately NOT reset by
        #: ``_reset_replay``: it is the ``?t=`` cache-buster on evidence URLs, and a counter
        #: that restarts at 0 when the loop restarts hands a returning judge the browser's
        #: cached crop from the previous session under this session's ranking -- exactly the
        #: stale-image failure the suffix exists to prevent.
        self._evidence_seq: int = 0
        #: The newest thumbnail URL per roster key (see _write_thumbnails), and which
        #: camera the round-robin refreshes next.
        self._frame_urls: dict[str, str] = {}
        self._thumb_cursor: int = 0
        self._budget: BudgetState | None = None
        self._telemetry = Telemetry(enabled=enabled_by_environment())
        #: Roster keys of the fire-free (CONFUSE) windows, and their uncapped threshold
        #: crossings this camera-night: the published RealisedAlertRate is over these alone.
        self._confuse_keys: set[str] = set()
        self._alerts_confuse: int = 0
        #: Measured seconds between tick starts (EWMA), beside the configured tick_seconds.
        self.tick_seconds_measured: float | None = None
        self._last_tick_started: float | None = None
        #: Walls admitted by the route but not yet attached (see watchlist) -- counted
        #: against MAX_VIEWERS so a burst of connects cannot all pass the check.
        self._reserved: int = 0
        self._loader: ThreadPoolExecutor | None = None

    # -----------------------------------------------------------------------------
    # Availability
    # -----------------------------------------------------------------------------

    def probe(self) -> None:
        """Decide once whether this box can run the pipeline at all. Blocking; call in a thread.

        It cannot raise, and that is a route-level requirement rather than defensive habit:
        ``/health`` is polled daily for the length of the judging window and its entire job
        is to *report* a box that cannot replay. A probe that threw would make the route
        explaining the outage part of it -- a 500 where the designed answer is
        ``live_pipeline.available: false`` with the reason attached. An unforeseen failure
        therefore becomes a named reason string and an ERROR in the log, which is the same
        degraded state the three expected ones produce.

        ``_probed`` is raised LAST, once ``_unavailable`` holds the verdict: ``ensure_probed``
        trusts the flag on its fast path, and raising it first let a second viewer through
        while ``_probe_reason`` was still scanning -- onto a roster it had just cleared.
        """
        try:
            self._unavailable = self._probe_reason()
        except Exception as exc:
            LOG.exception("probe failed")
            self._unavailable = f"probe failed: {type(exc).__name__}: {exc}"
        finally:
            self._probed = True

    def _probe_reason(self) -> str | None:
        """Why this box cannot replay, or None when it can -- in which case the inputs are set.

        Three inputs, each a hard requirement and each with its own reason string, because
        "live pipeline unavailable" is only useful to whoever has to fix it if it says which
        of the three is missing.
        """
        # Cleared first, so a failed probe cannot leave a half-set roster behind a False
        # ``available`` -- ``/health`` publishes the camera count beside the flag and the two
        # must never disagree.
        self._cameras = []
        self._threshold = None
        self._threshold_source = None
        self._classifier = None
        self._credit = None
        self._t0_index = None
        self._frame_count = 0
        self._confuse_keys = set()

        try:
            sequences, credit = _wall_sequences()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return f"corpus manifests unreadable: {type(exc).__name__}: {exc}"
        if not sequences:
            return "no replayable window is on this box -- run `make fetch` to pull the corpus"

        threshold, source = _calibrated_threshold()
        if threshold is None:
            # Found-and-unusable gets its own sentence. Telling an operator to run
            # `make bench` when the file is already there and its threshold is null sends
            # them to spend ninety-five minutes on the wrong problem.
            if source is not None and ":" in source:
                return (
                    f"the threshold at {source} is not a usable number -- the wall will not "
                    "invent one; see the ERROR in the service log for the value it found"
                )
            # `make bench` is the held-out (HOLD) run, and it is the only command that
            # writes results/bench_<date>.json. The threshold inside it is solved on
            # CONFUSE, but running anything "on the CONFUSE arm" writes a different file
            # (arm_confuse_<date>.json) that this route never reads.
            if source is not None:
                return (
                    f"{source} is committed but cannot calibrate this wall -- it will not "
                    "invent a threshold; the ERROR in the service log says why"
                )
            if _benchmark_file() is None and _DEV_REJECTIONS:
                return (
                    "no usable calibrated threshold -- the wall will not invent one. "
                    f"Newest development run: {_DEV_REJECTIONS[0]}"
                )
            return (
                "no calibrated threshold in results/ yet -- the wall will not invent one. "
                "It needs the threshold solved by the held-out benchmark run "
                "(`make bench`), which has not been published"
            )

        try:
            classifier = TileClassifier(ONNX_PATH)
        except Exception as exc:
            return f"tile classifier unavailable: {type(exc).__name__}: {exc}"

        self._cameras, self._t0_index, self._frame_count = _align(sequences)
        self._confuse_keys = {c.camera_id for c in self._cameras if c.sequence.arm == "confuse"}
        self._threshold = threshold
        self._threshold_source = source
        self._classifier = classifier
        self._credit = credit
        return None

    async def ensure_probed(self) -> None:
        """Probe exactly once, even when two judges connect inside the same second.

        A plain ``if not self._probed`` lets a second caller through the check while the
        first is still inside ``asyncio.to_thread``: two ONNX loads, two full manifest scans,
        and a second ``_cameras`` list assigned underneath a loop already ticking against the
        first. Callers share the in-flight task instead. ``shield`` because one judge closing
        the tab must not cancel the probe the other is waiting on. A probe already in flight
        is awaited before the flag is consulted, so no caller returns ahead of its verdict.
        """
        task = self._probe_task
        if task is not None and not task.done():
            await asyncio.shield(task)
            return
        if self._probed:
            return
        task = self._probe_task = asyncio.create_task(
            asyncio.to_thread(self.probe), name="plumerank-probe"
        )
        await asyncio.shield(task)

    @property
    def available(self) -> bool:
        return self._unavailable is None and self._failed is None

    @property
    def failed(self) -> str | None:
        """Why the replay loop gave up after repeated tick failures, or None."""
        return self._failed

    def status(self) -> dict[str, Any]:
        """What ``/health`` and the ``meta`` event both report, written once."""
        return {
            "state": (
                "failed"
                if self._failed is not None
                else "unavailable"
                if self._probed and not self.available
                else "running"
                if self._task is not None and not self._task.done()
                else "idle"
            ),
            "viewers": len(self._viewers),
            "replay_id": self.replay_id,
            "ticks": self._ticks,
            "frame_index": self._frame_index,
            "frame_count": self._frame_count,
            "last_tick": self._last_tick_iso,
            "tick_seconds": self.tick_seconds,
            "tick_seconds_measured": self.tick_seconds_measured,
            "idle_stop_seconds": self.idle_stop_seconds,
            "consecutive_tick_failures": self._tick_failures,
            "failure": self._failed,
        }

    def live_pipeline(self) -> dict[str, Any]:
        """The degraded-state flag, in the words the web's degraded state is designed around."""
        return {
            "available": self.available,
            "detail": (
                None
                if self.available
                else f"live pipeline unavailable -- {self._failed or self._unavailable}"
            ),
            "cameras": len(self._cameras),
            "threshold": self._threshold,
            "threshold_source": self._threshold_source,
            "calibration_kind": calibration_kind(self._threshold_source),
        }

    # -----------------------------------------------------------------------------
    # Viewers
    # -----------------------------------------------------------------------------

    async def attach(self) -> asyncio.Queue[str]:
        """Register one viewer and start the loop if it is the first. Returns its queue."""
        if self._failed is not None and (self._task is None or self._task.done()):
            # A failed wall is retried by the next viewer, from a fresh probe: whatever made
            # five ticks in a row raise (a full disk, a damaged frame) may be gone by now.
            LOG.warning("re-probing after replay failure: %s", self._failed)
            self._failed = None
            self._tick_failures = 0
            self._probed = False
        await self.ensure_probed()
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=VIEWER_QUEUE_DEPTH)
        self._viewers.add(queue)
        # No lock guarding the start: nothing awaits between the emptiness check and
        # create_task, so no second connection can interleave with it on a single-threaded
        # event loop. An asyncio.Lock built at import time would also bind itself to
        # whichever loop touched it first, which is a failure that only shows up under a
        # second one.
        if self._task is None or self._task.done():
            self._reset_replay()
            self._task = asyncio.create_task(self._run(), name="plumerank-replay")
        return queue

    @property
    def viewer_count(self) -> int:
        return len(self._viewers)

    def at_capacity(self) -> bool:
        """True when one more viewer would exceed ``MAX_VIEWERS``, counting admitted walls
        that have not attached yet."""
        return len(self._viewers) + self._reserved >= MAX_VIEWERS

    def reserve(self) -> None:
        """Hold one slot for a wall the route has admitted but not yet attached."""
        self._reserved += 1

    def unreserve(self) -> None:
        """Give back a reserved slot (the wall attached, or never got that far)."""
        self._reserved = max(0, self._reserved - 1)

    def detach(self, queue: asyncio.Queue[str]) -> None:
        """Drop one viewer. The loop notices on its next pass and starts its idle timer.

        Deliberately not "stop the loop when this was the last one": a judge reloading the
        page would otherwise tear down 62 warm background models and pay the whole warm-up
        again, every reload. ``REPLAY_IDLE_STOP_SECONDS`` is the grace this buys.
        """
        self._viewers.discard(queue)

    async def stop(self) -> None:
        """Cancel the loop. Called from the app's lifespan on shutdown."""
        task, self._task = self._task, None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # -----------------------------------------------------------------------------
    # The loop
    # -----------------------------------------------------------------------------

    def _reset_replay(self) -> None:
        self._frame_index = 0
        self._ticks = 0
        self._alerts_uncapped = 0
        self._alerts_confuse = 0
        self._last_tick_started = None
        self._last_payload = None
        self._invocations_at_night_start = (
            self._classifier.invocations if self._classifier is not None else 0
        )
        for camera in self._cameras:
            camera.reset()
        self._budget = (
            BudgetState([c.camera_id for c in self._cameras]) if self._cameras else None
        )
        # The per-camera trailing windows start empty with the background models: a camera
        # that has no night yet cannot depart from it, and scores 0.0 through the warm-up.
        self._baseline = Baseline()
        self._frame_urls = {}
        self._thumb_cursor = 0

    async def _run(self) -> None:
        """Tick while anyone is watching; sleep, then exit, when nobody is.

        The idle branch does no work at all -- it does not decode a frame, does not run a
        background model and does not write an evidence crop. That is the whole burstable-CPU
        argument: a loop that kept computing for two minutes after the last judge closed the
        tab would spend credits on frames nobody sees, every time.
        """
        idle_since: float | None = None
        try:
            while True:
                if not self._viewers:
                    self._last_tick_started = None
                    now = time.monotonic()
                    idle_since = now if idle_since is None else idle_since
                    if now - idle_since >= self.idle_stop_seconds:
                        LOG.info("replay stopping: no viewer for %.0f s", self.idle_stop_seconds)
                        return
                    await asyncio.sleep(IDLE_POLL_SECONDS)
                    continue

                idle_since = None
                if not self.available:
                    # Nothing to compute and nothing to fake. The stream stays open with the
                    # meta event's degraded flag on it, which is what the UI renders.
                    await asyncio.sleep(IDLE_POLL_SECONDS)
                    continue

                started = time.monotonic()
                self._observe_tick_start(started)
                try:
                    payload = await asyncio.to_thread(self._tick)
                except Exception as exc:
                    if self._tick_failed(exc):
                        return
                    await asyncio.sleep(self._backoff_seconds())
                    continue
                self._tick_failures = 0
                self._publish(payload)
                # The remainder of the tick, not a flat sleep: at 62 cameras the pipeline
                # itself takes a large slice of the second, and sleeping a whole one on top
                # would make the wall drift slower than the clock it prints.
                await asyncio.sleep(max(0.0, self.tick_seconds - (time.monotonic() - started)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a wall that stops must say why, not just stop
            LOG.exception("replay loop failed")
            self._failed = f"replay loop failed: {type(exc).__name__}: {exc}"
        finally:
            self._task = None

    def _tick_failed(self, exc: Exception) -> bool:
        """Record one failed tick; True when the wall should give up (enter failed state).

        The frame index is advanced past the failing frame, so one undecodable JPEG costs one
        tick rather than stalling the wall on it forever.
        """
        self._tick_failures += 1
        LOG.exception(
            "tick failed (%d consecutive, limit %d)", self._tick_failures, MAX_TICK_FAILURES,
            exc_info=exc,
        )
        if self._frame_count > 0:
            self._frame_index = (self._frame_index + 1) % self._frame_count
        if self._tick_failures >= MAX_TICK_FAILURES:
            self._failed = (
                f"{self._tick_failures} consecutive ticks failed; last: "
                f"{type(exc).__name__}: {exc}"
            )
            return True
        return False

    def _backoff_seconds(self) -> float:
        return min(
            TICK_BACKOFF_CAP_SECONDS,
            TICK_BACKOFF_BASE_SECONDS * 2 ** max(0, self._tick_failures - 1),
        )

    def _observe_tick_start(self, started: float) -> None:
        """Fold one tick-to-tick interval into the published measured period (EWMA).

        The configured ``tick_seconds`` is a floor, not a promise: when the pipeline takes
        longer than it, the wall runs slower than the clock it prints, and /health says so.
        """
        last, self._last_tick_started = self._last_tick_started, started
        if last is None:
            return
        delta = started - last
        previous = self.tick_seconds_measured
        self.tick_seconds_measured = round(
            delta if previous is None else 0.8 * previous + 0.2 * delta, 3
        )

    def _map(self, fn: Callable[[Any], Any], items: list[Any]) -> list[Any]:
        """``fn`` over ``items`` on LOAD_WORKERS threads, results in input order."""
        if LOAD_WORKERS <= 1 or len(items) < 2:
            return [fn(item) for item in items]
        if self._loader is None:
            self._loader = ThreadPoolExecutor(
                max_workers=LOAD_WORKERS, thread_name_prefix="plumerank-work"
            )
        return list(self._loader.map(fn, items))

    def _load_frames(self, paths: list[str]) -> list[np.ndarray]:
        """Every camera's frame for this tick, loaded on LOAD_WORKERS threads, in order."""
        return self._map(_read_working_frame, paths)

    def _tick(self) -> dict[str, Any]:
        """One simulated minute across the whole roster. Blocking; runs in a worker thread.

        stabilise -> segment -> statistic -> rank, then the top-2 above threshold get one
        ONNX forward pass each and are re-fused. The escalation happens between the two
        rankings because ``should_escalate`` takes a rank: the classical hazard decides who
        is worth a model call, and the model call then decides where they finish.
        """
        classifier, threshold, budget = self._classifier, self._threshold, self._budget
        if classifier is None or threshold is None or budget is None:
            raise RuntimeError("_tick ran before probe() found a runnable pipeline")
        index = self._frame_index
        classical: list[CameraScore] = []
        frames: dict[str, np.ndarray] = {}
        picked: dict[str, tuple[Component, Track, Stats]] = {}

        images = self._load_frames(
            [c.sequence.frame_paths[index + c.frame_offset] for c in self._cameras]
        )
        # Two phases. The first -- stabilise, segment, track -- touches only each camera's
        # own state (its MOG2, CLAHE and tracker) and runs on the loader threads, because
        # it is most of a tick's CPU. The second -- measure() -- shares statistic.py's
        # per-shape DIS objects across cameras, so it stays on this thread, in roster order.
        def segment_one(item: tuple[_Camera, np.ndarray]) -> tuple[Any, ...]:
            camera, bgr = item
            grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # The first frame of a replay has no predecessor to register against, so it is
            # its own reference. Warping it against nothing would be a fabricated shift.
            work = grey if camera.prev_grey is None else stabilise(camera.prev_grey, grey).frame
            components = camera.segmenter.apply(work)
            tracks = camera.tracker.update(index, components)
            return camera, bgr, grey, work, _best_track(components, tracks)

        segmented = self._map(segment_one, list(zip(self._cameras, images)))
        # Read here, straight after segmentation: on a window's last frame
        # _reset_camera_night() rebuilds every segmenter before the payload is built, and a
        # flag read then would call a frame scored on warm models cold. Every lap re-warms
        # (the camera-night reset), so the client cannot infer this from the frame index.
        warm, frames_seen = self._warm_state()
        for camera, bgr, grey, work, best in segmented:
            if best is None:
                # No foreground is not a missing measurement, it is a quiet camera: it scores
                # sigmoid(FUSION_BIAS) and stays on the wall. Dropping it would make the
                # published per-camera-night denominator depend on how quiet the night was.
                stats = Stats(s1=0.0, s2=0.0, s3=0.0, s4=0.0, raw_s3=0.0)
                component: Component | None = None
            else:
                component, track = best
                stats = measure(track, work, camera.prev_work)
                picked[camera.camera_id] = (component, track, stats)

            frames[camera.camera_id] = bgr
            classical.append(
                CameraScore(
                    camera_id=camera.camera_id,
                    frame_index=index,
                    hazard=fuse(stats),
                    stats=stats,
                    component=component,
                )
            )
            camera.prev_grey = grey
            camera.prev_work = work

        final = self._escalate(classifier, classical, frames)
        # What is ranked, thresholded and counted is the camera's departure from its own
        # night (rank.Baseline), not the absolute fusion -- the same series bench.py solves
        # the threshold on. The escalation gate above reads the ABSOLUTE rank on purpose:
        # it decides which tiles are worth a model call, and the calibration measured it so.
        final = [
            dataclasses.replace(s, hazard=self._baseline.observe(s.camera_id, s.hazard))
            for s in final
        ]
        entries = self._write_evidence(emit(final, threshold, budget), frames, picked, index)
        self._write_thumbnails(entries, frames, fill_all=self._ticks == 0)

        crossed = [s.camera_id for s in final if s.hazard >= threshold]
        self._alerts_uncapped += len(crossed)
        self._alerts_confuse += sum(1 for key in crossed if key in self._confuse_keys)
        self._frame_index = (index + 1) % self._frame_count
        self._ticks += 1
        self._last_tick_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self._frame_index == 0:
            # One completed cycle is one camera-night, which is the only span in which the
            # published rate has its unit. Publishing mid-cycle would put a partial night on
            # the same axis as a whole one.
            self._publish_telemetry()
            self._reset_camera_night()

        return {
            "replay_id": self.replay_id,
            "frame_index": index,
            # The roster's own aligned ignition index, not the FIgLib constant: after
            # ``_align`` every camera really does ignite here, so the one clock on screen is
            # true of every tile under it. Null when nothing on the roster ignites -- a
            # CONFUSE-only wall has no T+ to count, and the Clock renders that state.
            "t0_index": self._t0_index,
            "minute": (
                None
                if self._t0_index is None
                else round((index - self._t0_index) * MINUTES_PER_FRAME, 3)
            ),
            "threshold": threshold,
            "warm": warm,
            "frames_seen": frames_seen,
            "entries": [
                e.to_json() | {"frame_url": self._frame_urls.get(e.camera_id)} for e in entries
            ],
        }

    def _warm_state(self) -> tuple[bool, int]:
        """(every background model past warm-up?, the fewest frames any of them has seen).

        ``Segmenter.warm`` is the source of truth. It also covers the ranking's own warm-up:
        ``rank.rebase`` returns 0.0 -- a camera_id-ordered list -- until the baseline holds
        BASELINE_WARMUP_FRAMES (5), which ends before WARMUP_FRAMES (8) does.
        """
        warm = bool(self._cameras) and all(c.segmenter.warm for c in self._cameras)
        seen = min((c.segmenter.frames_seen for c in self._cameras), default=0)
        return warm, seen

    def _escalate(
        self,
        classifier: TileClassifier,
        classical: list[CameraScore],
        frames: dict[str, np.ndarray],
    ) -> list[CameraScore]:
        """Fold one ONNX tile score into the hazard of the cameras that earned one.

        The ordering is ``emit``'s key -- hazard descending, ties on camera id -- written a
        second time below: change one, change both. It is applied WITHIN each arm, and the
        arm is ground-truth metadata from the manifest: the wall is the DEV and CONFUSE bench
        replays on one screen, and bench.py escalates the top-2 of each separately. So the
        rank this gate reads is the camera's rank inside its own arm, not the rank a
        dispatcher sees on the merged watch list; a single-roster deployment, with no arm
        labels, would allot the top-2 across every camera.
        """
        # Ranked WITHIN each arm, not across the whole wall. bench.py replays the CONFUSE
        # windows alone (the calibration) and the DEV windows alone, each with its own
        # top-2 escalation; ranking across all 62 let fire footage take the escalation slots
        # from the fire-free cameras, so their alert rate sat at 1.68 against the 1.99 the
        # threshold was calibrated to, and the drift-low alarm fired every camera-night.
        # Per arm, each half of the wall is the bench replay it is calibrated against.
        arm_of = {c.camera_id: c.sequence.arm for c in self._cameras}
        groups: dict[str, list[CameraScore]] = {}
        for score in classical:
            groups.setdefault(arm_of.get(score.camera_id, ""), []).append(score)
        ranked: list[tuple[int, CameraScore]] = []
        for group in groups.values():
            ordered = sorted(group, key=lambda s: (-s.hazard, s.camera_id))
            ranked.extend(enumerate(ordered, start=1))
        out: list[CameraScore] = []
        for rank, score in ranked:
            if score.component is None or not should_escalate(rank, score.hazard):
                out.append(score)
                continue
            tile_score = classifier.score_tile(
                frames[score.camera_id], score.component.bbox()
            )
            out.append(
                dataclasses.replace(
                    score,
                    hazard=fuse(score.stats, tile_score),
                    escalated=True,
                    escalation_score=tile_score,
                )
            )
        return out

    def _write_evidence(
        self,
        entries: list[WatchEntry],
        frames: dict[str, np.ndarray],
        picked: dict[str, tuple[Component, Track, Stats]],
        index: int,
    ) -> list[WatchEntry]:
        """Draw one crop per row that has a component, and hang its URL on the row.

        The path is ``{replay_id}/{frame_index}/{camera}.jpg`` -- the frame index, not the
        simulated minute, because the minute is derived from it and two cameras on the
        roster ignite at different indices. The replay id is fixed for the life of the
        process, so the frame index cycling 0..N-1 overwrites rather than accumulates. That bounds the evidence directory at roster x window files on a box
        nobody is watching for fourteen days -- an id per loop start would have grown it
        without limit, one full set per judge who opened the page.

        The ``?t=`` suffix is cache-busting for that same reuse: same URL, new pixels every
        cycle, and a browser that reused the old JPEG would caption this minute's ranking
        with last cycle's picture.
        """
        self._evidence_seq += 1
        out: list[WatchEntry] = []
        for entry in entries:
            best = picked.get(entry.camera_id)
            if best is None:
                out.append(entry)
                continue
            component, track, stats = best
            directory = EVIDENCE_DIR / self.replay_id / str(index)
            directory.mkdir(parents=True, exist_ok=True)
            final = directory / f"{entry.camera_id}.jpg"
            # Drawn to a temporary name and renamed into place: Caddy may be serving the
            # previous cycle's file at this path, and a reader must never get half a JPEG.
            partial = directory / f".{entry.camera_id}.partial.jpg"
            draw_evidence(frames[entry.camera_id], component, track, stats, partial)
            os.replace(partial, final)
            url = (
                f"/evidence/{self.replay_id}/{index}/{entry.camera_id}.jpg"
                f"?t={self._evidence_seq}"
            )
            out.append(dataclasses.replace(entry, evidence_url=url))
        return out

    def _write_thumbnails(
        self,
        entries: list[WatchEntry],
        frames: dict[str, np.ndarray],
        *,
        fill_all: bool = False,
    ) -> None:
        """Refresh the small live image a wall tile shows, within a fixed per-tick budget.

        Evidence crops exist only for cameras with a tracked component, so without these most
        tiles would read "no frame published". Encoding all ~60 cameras every tick would cost
        a t4g.small tens of milliseconds a second; instead the top-k rows -- the ones a
        dispatcher is reading -- are refreshed every tick, and one more camera per tick in
        round-robin, so every tile has a picture within one roster's worth of ticks.

        One file per camera, overwritten in place (``{replay_id}/frames/{key}.jpg``), so the
        directory is bounded at the roster size; ``?t=`` busts the browser cache on rewrite.

        ``fill_all`` writes every camera once. ``_tick`` sets it on a replay's first tick:
        ``_reset_replay`` has just emptied ``_frame_urls``, and without it a quiet camera
        would read "no frame published" until the round-robin reached it, up to a roster's
        worth of ticks into the judge's first minute (~62 thumbnails, once).
        """
        if fill_all:
            targets = [c.camera_id for c in self._cameras]
        else:
            targets = [e.camera_id for e in entries if e.rank <= TOP_K]
            if self._cameras:
                self._thumb_cursor %= len(self._cameras)
                targets.append(self._cameras[self._thumb_cursor].camera_id)
                self._thumb_cursor += 1
        directory = EVIDENCE_DIR / self.replay_id / "frames"
        for key in dict.fromkeys(targets):
            frame = frames.get(key)
            if frame is None:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            height = round(frame.shape[0] * THUMB_WIDTH / frame.shape[1])
            thumb = cv2.resize(frame, (THUMB_WIDTH, height), interpolation=cv2.INTER_AREA)
            partial = directory / f".{key}.partial.jpg"
            if cv2.imwrite(str(partial), thumb, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY]):
                # Atomic swap, as for evidence crops: never serve a half-written thumbnail.
                os.replace(partial, directory / f"{key}.jpg")
                self._frame_urls[key] = (
                    f"/evidence/{self.replay_id}/frames/{key}.jpg?t={self._evidence_seq}"
                )

    def _reset_camera_night(self) -> None:
        """A new camera-night: new budget, new baseline, and fresh per-camera pipeline state.

        Every camera is reset as well -- background model, tracker, previous frame -- because
        the replay wraps from the window's last frame to its first, a 78-minute jump in the
        footage. Carrying a warm model across it fed the next night a background learned on
        the wrong minutes and gave the tracker negative frame spans; ``bench.py`` replays each
        window exactly once from fresh state, and the wall now does the same each night.

        With this, per-arm escalation (_escalate) and the centre-trimmed CONFUSE windows
        (_align), each camera-night of the wall's fire-free half replays exactly what
        ``bench.py`` calibrated on: its ``RealisedAlertRate`` reproduces the calibration rate
        (1.9885 on the committed DEV run, every night, measured locally on 2026-10-04), so the
        drift alarm fires only when something actually moves.
        """
        for camera in self._cameras:
            camera.reset()
        self._alerts_uncapped = 0
        self._alerts_confuse = 0
        self._invocations_at_night_start = (
            self._classifier.invocations if self._classifier is not None else 0
        )
        self._budget = BudgetState([c.camera_id for c in self._cameras])
        self._baseline.reset()

    def night_metrics(self) -> dict[str, float]:
        """The metrics one completed camera-night publishes, computed (not sent).

        ``RealisedAlertRate`` is the **uncapped** count of threshold crossings -- the quantity
        ``solve_threshold`` was calibrated on -- and it is taken over the fire-free CONFUSE
        windows ONLY. The threshold is solved on fire-free windows to 2.0 alerts per
        camera-window; the DEV windows on the same wall are fire footage that is *supposed*
        to alert after ignition, so a rate over the whole roster sits above budget by design
        and the drift alarm fired every time anyone watched. With no CONFUSE window on the
        roster the rate is not published at all. ``EscalationFraction`` stays over the whole
        roster: it is the model-call cost of the wall as it runs.
        """
        out: dict[str, float] = {}
        if not self._cameras or self._frame_count <= 0:
            return out
        if self._confuse_keys:
            # Every camera on the wall is replayed for the same _frame_count (_align trims
            # to a common length), so the lengths really are uniform.
            out["RealisedAlertRate"] = alerts_per_camera_night(
                self._alerts_confuse, [self._frame_count] * len(self._confuse_keys)
            )
        if self._classifier is not None:
            # Differenced against the count at the start of this night, not the lifetime
            # total: the published fraction is per camera-night.
            calls = self._classifier.invocations - self._invocations_at_night_start
            out["EscalationFraction"] = calls / (len(self._cameras) * self._frame_count)
        return out

    def _publish_telemetry(self) -> None:
        """Send ``night_metrics`` on a background thread, once per completed camera-night.

        Off the tick thread: a slow or unreachable CloudWatch must cost the wall nothing,
        and put_metric_data is a network call (telemetry.py also bounds it to one short
        attempt).
        """
        metrics = self.night_metrics()
        if not metrics:
            return
        units = {"RealisedAlertRate": "Count", "EscalationFraction": "None"}

        def send() -> None:
            for name, value in metrics.items():
                self._telemetry.put(name, value, unit=units[name])

        threading.Thread(target=send, name="plumerank-telemetry", daemon=True).start()

    # -----------------------------------------------------------------------------
    # Broadcast
    # -----------------------------------------------------------------------------

    def _publish(self, payload: dict[str, Any]) -> None:
        frame = _sse("tick", payload)
        self._last_payload = frame
        for queue in list(self._viewers):
            if queue.full():
                # Drop this viewer's oldest tick rather than block the wall on their
                # connection. A stalled reader must cost itself latency, never the roster.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(frame)

    def last_tick_frame(self) -> str | None:
        """The most recent tick, already SSE-encoded, or None before the first one."""
        return self._last_payload

    def heartbeat(self) -> bool:
        """One proof-of-life datapoint. Blocking; call in a thread.

        Withheld while the wall is in its failed state, so the silence alarm (and its reboot
        action) fires on a box whose replay loop gave up. A wall that is unavailable BY
        DESIGN -- no calibrated threshold yet -- still heartbeats: that is a healthy box,
        and a reboot would change nothing, so it must not loop.
        """
        if self._failed is not None:
            return False
        return self._telemetry.heartbeat()

    def meta(self) -> dict[str, Any]:
        """The first event on every stream: everything static for this replay.

        This is what three deleted routes turned into. ``/api/cameras``, ``/api/replay`` and
        ``/api/budget`` were all static-per-replay metadata, so they became the first event
        on the one stream that already existed -- three fetch hooks and three loading states
        deleted, and no window in which the wall is drawn against a roster it has not read.
        """
        return {
            "replay_id": self.replay_id,
            # Flat keys, and the flatness is the contract: web/lib/types.ts reads
            # budget_target / budget_per_camera / threshold / threshold_hash / top_k by
            # those names, and a nested object here would leave the frontend rendering its
            # designed "the stream did not say" state over a stream that did say.
            "cameras": [c.camera_id for c in self._cameras],
            "sequences": [
                {
                    "camera_id": c.camera_id,
                    "sequence_id": c.sequence.sequence_id,
                    "arm": c.sequence.arm,
                    # Ground truth, and the UI labels it as such. The pipeline never sees
                    # it: it reaches the browser from the manifest, not from a stage.
                    # ``t0_index`` is this window's own index in the manifest and
                    # ``frame_offset`` is what ``_align`` skips off its head, so
                    # t0_index - frame_offset is the shared index published above for
                    # every camera and the arithmetic is checkable from the payload.
                    "t0_index": c.sequence.t0_index,
                    "frame_offset": c.frame_offset,
                }
                for c in self._cameras
            ],
            "credit": self._credit,
            "frame_count": self._frame_count,
            "t0_index": self._t0_index,
            # The corpus cadence, not the wall-clock tick: the clock on screen reads
            # "T+7 min since ignition", which is a statement about the imagery. The rate
            # the replay is played back at is tick_seconds and is a different number.
            "seconds_per_frame": SECONDS_PER_FRAME,
            "minutes_per_frame": MINUTES_PER_FRAME,
            "tick_seconds": self.tick_seconds,
            "tick_seconds_measured": self.tick_seconds_measured,
            "warmup_frames": WARMUP_FRAMES,
            "top_k": TOP_K,
            "budget_target": BUDGET_ALERTS_PER_CAMERA_NIGHT,
            "budget_per_camera": ALERTS_PER_CAMERA_PER_NIGHT,
            "budget_capacity": len(self._cameras) * ALERTS_PER_CAMERA_PER_NIGHT,
            "drift_tolerance": BUDGET_DRIFT_TOLERANCE,
            # Solved on the CONFUSE arm and read off disk, never defaulted -- so it is
            # null exactly when live_pipeline says the wall cannot run.
            "threshold": self._threshold,
            "threshold_source": self._threshold_source,
            "calibration_kind": calibration_kind(self._threshold_source),
            "threshold_hash": freeze_hash(),
            "live_pipeline": self.live_pipeline(),
            "evidence_base": "/evidence",
        }


REPLAY = Replay()


def _sse(event: str, payload: dict[str, Any]) -> str:
    """One server-sent event. ``allow_nan=False`` so a NaN can never reach the browser.

    ``json.dumps`` writes a bare ``NaN`` by default, which is not JSON: the browser's parse
    throws, the stream handler dies and the wall stops updating with no error anywhere. A
    raise here is the loud version of the same failure, one stage earlier and with a name.
    """
    return f"event: {event}\ndata: {json.dumps(payload, allow_nan=False)}\n\n"


# ---------------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------------


async def _heartbeat() -> None:
    """One proof-of-life datapoint per minute, whether or not anyone is watching.

    The silence alarm fires on the absence of this metric, so it cannot be tied to the
    replay loop: an idle box is the normal overnight state and a dead box must not look
    like one. ``Telemetry`` is off unless PLUMERANK_TELEMETRY=1, so a laptop publishes nothing.
    """
    while True:
        await asyncio.to_thread(REPLAY.heartbeat)
        await asyncio.sleep(HEARTBEAT_SECONDS)


def warm_frame_cache(stop: threading.Event) -> int:
    """Fill the frame cache for every wall frame, once, at the lowest CPU priority.

    The first replay after a deploy would otherwise decode every full-size JPEG on the tick
    path (~2 s a tick on a laptop, more on Graviton). Run on a daemon thread at nice 19, so
    a judge's wall always wins the CPU; a frame already cached costs one stat(). Returns how
    many frames it decoded.
    """
    if FRAME_CACHE_DIR is None:
        return 0
    if hasattr(os, "setpriority") and hasattr(threading, "get_native_id"):
        with contextlib.suppress(OSError):  # per-thread nice on Linux; harmless elsewhere
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    try:
        sequences, _ = _wall_sequences()
    except (OSError, KeyError, TypeError, ValueError) as exc:
        LOG.warning("frame cache warm-up skipped: %s", exc)
        return 0
    decoded = 0
    for sequence in sequences:
        for path in sequence.frame_paths:
            if stop.is_set():
                return decoded
            cached = _cache_path(path)
            if cached is None or cached.is_file():
                continue
            try:
                _read_working_frame(path)
                decoded += 1
            except ValueError as exc:
                LOG.warning("frame cache warm-up: %s", exc)
    LOG.info("frame cache warm: %d frames decoded", decoded)
    return decoded


def prune_evidence(keep: str) -> list[str]:
    """Delete every replay-id evidence directory under EVIDENCE_DIR except ``keep``.

    The replay id is fixed per process, so a running wall overwrites in place; but every
    process start (deploy, crash restart, reboot) used to leave its predecessor's directory
    -- up to ~24 MB -- behind forever. Only names shaped like a replay id are touched.
    """
    removed: list[str] = []
    if not EVIDENCE_DIR.is_dir():
        return removed
    for child in EVIDENCE_DIR.iterdir():
        if child.name == keep or not _REPLAY_ID_RE.match(child.name):
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child.name)
    return removed


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # One OpenCV thread for the whole process, set before anything can do CV work.
    # t4g.small has 2 vCPUs and this process is both the replay loop and the web server:
    # letting OpenCV size its pool from the core count makes every morphologyEx and
    # Laplacian contend with the SSE writer, which shows up as tick-latency variance that
    # has nothing to do with the pipeline -- and on a burstable instance, contention is
    # spend. Parallelism belongs across cameras, not inside one frame.
    app.state.opencv_threads = pin_opencv_threads()
    with contextlib.suppress(OSError):
        pruned = prune_evidence(REPLAY.replay_id)
        if pruned:
            LOG.info("pruned %d stale evidence directories", len(pruned))
    app.state.started_at = time.time()
    app.state.started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    beat = asyncio.create_task(_heartbeat(), name="plumerank-heartbeat")
    warm_stop = threading.Event()
    threading.Thread(
        target=warm_frame_cache, args=(warm_stop,), name="plumerank-cache-warm", daemon=True
    ).start()
    try:
        yield
    finally:
        warm_stop.set()
        beat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beat
        await REPLAY.stop()


# docs_url / redoc_url / openapi_url are off: the documented surface is four routes, and
# FastAPI's three generated ones would make that claim false the moment check_claims.py
# introspects app.routes. They are also three unnecessary endpoints on a public box.
app = FastAPI(
    title="PlumeRank",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# All four routes are public, unauthenticated GET, with no cookies and nothing to
# mutate -- there is no credentialed cross-origin state for a CORS policy to protect
# here. Without this, README.md's own "point your local dev frontend at the live box"
# instructions (web/.env.example) fail in-browser with a CORS error the moment the
# frontend's origin differs from the API's, which production (Caddy, same origin)
# never exercises but a judge or contributor running `npm run dev` locally does.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _file_key(path: pathlib.Path) -> tuple[str, int, int] | None:
    """Identity of a file for cache invalidation: path, size, mtime -- or None if absent."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_size, st.st_mtime_ns)


_BUILD_INFO_CACHE: tuple[tuple[Any, ...], dict[str, Any]] | None = None


def _build_info_body() -> dict[str, Any]:
    """The /build-info body, computed once per model file and constants file.

    Uncached, every request loaded the ONNX model through ``readNetFromONNX``, re-hashed it
    twice and re-ran the module probes -- a few hundred milliseconds of CPU on a burstable
    box for a route anyone can curl in a loop. Nothing in the body can change under a
    running process except by those two files changing, so their size and mtime are the
    cache key: a model or a freeze committed and redeployed is picked up, and nothing else
    pays twice.
    """
    global _BUILD_INFO_CACHE
    key = (
        str(ONNX_PATH),
        _file_key(ONNX_PATH),
        str(RESULTS_DIR),
        _file_key(RESULTS_DIR / "constants.sha256"),
    )
    if _BUILD_INFO_CACHE is not None and _BUILD_INFO_CACHE[0] == key:
        return _BUILD_INFO_CACHE[1]
    identity = _instance_identity()
    body: dict[str, Any] = {
        "service": "plumerank",
        "opencv": {
            "version": cv2.__version__,
            "modules": _module_probes(),
            # Verbatim, in full, and last in the object so the interesting fields are
            # readable before 8 KB of build log. This is the field that cannot be faked:
            # a judge greps it for the module list and the compiler that produced it.
            "build_information": cv2.getBuildInformation(),
        },
        "dnn": _dnn_engine(),
        "constants": _constants_freeze(),
        "instance_id": identity["instance_id"],
        "ami_id": identity["ami_id"],
        "commit": _git_commit(),
        "onnx_sha256": _sha256(ONNX_PATH),
    }
    _BUILD_INFO_CACHE = (key, body)
    return body


@app.get("/build-info")
async def build_info() -> dict[str, Any]:
    """Everything a judge needs to check the OpenCV 5 claim without trusting a sentence."""
    return await asyncio.to_thread(_build_info_body)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health(request: Request) -> dict[str, Any]:
    """Liveness, uptime, the last tick, the replay state and the viewer count."""
    await REPLAY.ensure_probed()
    started = float(getattr(request.app.state, "started_at", time.time()))
    return {
        # "degraded" only for a wall that failed while running; unavailable-by-design (no
        # calibrated threshold yet) is a healthy box and stays "ok".
        "status": "degraded" if REPLAY.failed is not None else "ok",
        "started_at": getattr(request.app.state, "started_iso", None),
        "uptime_seconds": round(time.time() - started, 3),
        "replay": REPLAY.status(),
        "live_pipeline": REPLAY.live_pipeline(),
    }


def client_address(request: Request) -> str:
    """The viewer's address. Behind Caddy the peer is loopback, so the address is the last
    ``X-Forwarded-For`` hop -- the one Caddy itself appended, which a client cannot forge.
    A non-loopback peer is taken as-is and its header ignored."""
    peer = request.client.host if request.client else "unknown"
    if peer in ("127.0.0.1", "::1"):
        forwarded = request.headers.get("x-forwarded-for", "")
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        if hops:
            return hops[-1]
    return peer


class _Slot:
    """One admitted wall's claim on the caps, released exactly once.

    Taken in the handler, before the response exists, so a burst of simultaneous connects
    cannot all pass the cap check and then all attach. Released by ``_WallResponse`` when
    the response finishes for any reason -- including a client that vanished before the
    body was ever iterated, where the generator's own ``finally`` would never run.
    """

    __slots__ = ("client", "_reserved", "_released")

    def __init__(self, client: str) -> None:
        self.client = client
        _CLIENT_VIEWERS[client] += 1
        REPLAY.reserve()
        self._reserved = True
        self._released = False

    def attached(self) -> None:
        """The wall is now a counted viewer; its reservation is no longer needed."""
        if self._reserved:
            self._reserved = False
            REPLAY.unreserve()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.attached()
        _CLIENT_VIEWERS[self.client] -= 1
        if _CLIENT_VIEWERS[self.client] <= 0:
            del _CLIENT_VIEWERS[self.client]


class _WallResponse(StreamingResponse):
    """A StreamingResponse that releases its ``_Slot`` however the response ends."""

    def __init__(self, content: AsyncIterator[str], release: Callable[[], None], **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()


@app.get("/api/watchlist")
async def watchlist(request: Request) -> StreamingResponse:
    """The live wall, as server-sent events: one ``meta``, then one ``tick`` per minute."""
    client = client_address(request)
    if _CLIENT_VIEWERS[client] >= MAX_VIEWERS_PER_CLIENT:
        raise HTTPException(
            status_code=429,
            detail=f"at most {MAX_VIEWERS_PER_CLIENT} live walls per client; close a tab",
            headers={"Retry-After": "30"},
        )
    if REPLAY.at_capacity():
        # Refused before the response starts, so the browser's EventSource sees a plain
        # 503 and backs off on Retry-After instead of holding a socket that never ticks.
        raise HTTPException(
            status_code=503,
            detail=f"the live wall is at its {MAX_VIEWERS}-viewer limit; retry shortly",
            headers={"Retry-After": "30"},
        )

    slot = _Slot(client)

    async def stream() -> AsyncIterator[str]:
        queue = await REPLAY.attach()
        slot.attached()
        loop = asyncio.get_running_loop()
        quiet_since = loop.time()
        try:
            yield _sse("meta", REPLAY.meta())
            # A judge who connects mid-replay sees the current ranking immediately instead
            # of a blank wall until the next tick. There is one loop, so "the current
            # ranking" is unambiguous -- this is the same frame every other viewer has.
            last = REPLAY.last_tick_frame()
            if last is not None:
                yield last
            while True:
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=IDLE_POLL_SECONDS)
                    quiet_since = loop.time()
                except asyncio.TimeoutError:
                    # A comment frame, not a tick: it keeps proxies from closing an idle
                    # stream, and an SSE client ignores it by definition -- so a degraded
                    # wall never has to invent a tick to stay connected. Every
                    # KEEPALIVE_SECONDS of silence, not every poll.
                    if loop.time() - quiet_since >= KEEPALIVE_SECONDS:
                        quiet_since = loop.time()
                        yield ": keep-alive\n\n"
                if await request.is_disconnected():
                    return
        finally:
            # The finally is the disconnect detector that actually works: a client that
            # vanishes without a close frame is noticed when the generator is closed, which
            # happens whether it ended, raised or was cancelled. Polling is_disconnected
            # alone would leave a phantom viewer holding the loop open for the whole
            # judging window.
            REPLAY.detach(queue)

    return _WallResponse(
        stream(),
        slot.release,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Caddy is configured with flush_interval -1 for this route; the header is here
            # for any other proxy a judge puts in front, where buffering turns a live wall
            # into a stream that arrives in clumps or not at all.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/benchmark")
async def benchmark() -> Response:
    """``results/bench_<date>.json``, verbatim off the disk. No computation, ever.

    Bytes rather than a parsed-and-re-serialised object: the endpoint, the committed file
    and the printed block have to agree, and the only way three things agree forever is if
    two of them are the same bytes. The parse below is a validity check whose result is
    thrown away.
    """
    path = _benchmark_file()
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "no results/bench_<date>.json on this box -- the published headline is "
                "produced by `make fetch && make bench` and committed"
            ),
        )
    raw = path.read_bytes()
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        # Loud, and a 500 rather than a 404: the file exists and is wrong, which is a
        # different fact from the file being absent, and serving it as application/json
        # would break the page with a parse error three layers away from the cause.
        raise HTTPException(
            status_code=500, detail=f"{path.name} is not valid JSON: {exc}"
        ) from exc
    return Response(
        content=raw,
        media_type="application/json",
        headers={"X-PlumeRank-Results-File": _repo_relative(path)},
    )

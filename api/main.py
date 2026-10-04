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
Uvicorn that is holding 40 warm background models would put file IO in contention with the
pipeline producing the files.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterable

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

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
from plumerank.telemetry import Telemetry
from plumerank.types import CameraScore, Component, Sequence, Stats, Track, WatchEntry

LOG = logging.getLogger("plumerank.api")

# ---------------------------------------------------------------------------------
# Paths. Module-level and read at call time, so a test can point one at a tmp_path
# without a fixture reaching inside a handler.
# ---------------------------------------------------------------------------------

REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR: pathlib.Path = REPO_ROOT / "data"
CORPUS_DIR: pathlib.Path = DATA_DIR / "corpus" / "figlib"
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


def _wall_sequences() -> tuple[list[Sequence], str | None]:
    """Every replayable window whose frames are actually on this disk, and the credit line.

    Windows come from the DEV and CONFUSE manifests only, and any sequence id that also
    appears in the HOLD manifest is refused outright. HOLD is read once, by ``bench.py``,
    after the freeze; ``test_hold_untouched_until_freeze`` grants no exemption "including
    for the demo wall", and a rule with one exemption is a rule a judge has to take on
    trust. Enforcing it here means a mis-edited manifest cannot quietly put a held-out
    sequence on screen.

    A window with any frame missing from disk is skipped rather than truncated: a partly
    fetched sequence would replay a shorter camera-night, and every rate in this project is
    per camera-night.
    """
    hold_ids: set[str] = set()
    hold = _load_manifest(DATA_DIR / HOLD_MANIFEST)
    if hold:
        hold_ids = {w["sequence_id"] for w in hold.get("windows", [])}

    sequences: list[Sequence] = []
    credit: str | None = None
    for name in WALL_MANIFESTS:
        manifest = _load_manifest(DATA_DIR / name)
        if manifest is None:
            continue
        credit = credit or manifest.get("credit")
        arm = str(manifest.get("arm", ""))
        for window in manifest.get("windows", []):
            sequence_id = str(window["sequence_id"])
            if sequence_id in hold_ids:
                LOG.error("refusing HOLD sequence %s on the wall", sequence_id)
                continue
            paths = [str(CORPUS_DIR / sequence_id / frame) for frame in window["frames"]]
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


def _read_working_frame(path: str) -> np.ndarray:
    """One corpus JPEG at working resolution, BGR.

    INTER_AREA because the source is 3072x2048 and this is a >6x downscale: bilinear would
    alias a thin plume edge into or out of existence, and S3 is a statistic about that edge.
    """
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"could not decode {path} -- the corpus on this box is damaged")
    return cv2.resize(frame, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA)


def _benchmark_file() -> pathlib.Path | None:
    """The newest committed ``results/bench_<date>.json``, or None.

    Newest by name, which is newest by date: the filenames are ISO dates, so a lexical sort
    is a chronological one and no mtime is involved -- a fresh ``git clone`` rewrites every
    mtime to the checkout time, and picking the headline by mtime would make the served
    number depend on the order git happened to write the files.
    """
    if not RESULTS_DIR.is_dir():
        return None
    candidates = sorted(RESULTS_DIR.glob("bench_*.json"))
    return candidates[-1] if candidates else None


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
    if path is None:
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.error("benchmark file %s is unreadable: %s", path, exc)
        return None, None

    relative = _repo_relative(path)
    # Two spellings and no more. Both are named here rather than discovered by walking the
    # document, because a search that finds *a* number called "threshold" somewhere in a
    # nested ablation grid would calibrate the wall on an ablation cell.
    calibration = payload.get("calibration")
    if isinstance(calibration, dict) and "threshold" in calibration:
        source = f"{relative}:calibration.threshold"
        return _finite_threshold(calibration["threshold"], source), source
    if "threshold" in payload:
        source = f"{relative}:threshold"
        return _finite_threshold(payload["threshold"], source), source
    return None, relative


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

    __slots__ = ("sequence", "frame_offset", "segmenter", "tracker", "prev_grey", "prev_work")

    def __init__(self, sequence: Sequence, frame_offset: int = 0) -> None:
        self.sequence = sequence
        #: Frames skipped off the head of this window so that its own ignition lands on the
        #: same replay index as every other camera's. See ``Replay._align``.
        self.frame_offset = frame_offset
        self.segmenter = Segmenter()
        self.tracker = Tracker()
        self.prev_grey: np.ndarray | None = None
        self.prev_work: np.ndarray | None = None

    @property
    def camera_id(self) -> str:
        return self.sequence.camera_id

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

    The wall is one clock over windows recorded on different nights, and ``ui.md``'s Clock
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

    cameras: list[_Camera] = []
    for sequence in sequences:
        offset = 0 if sequence.t0_index is None or shared is None else sequence.t0_index - shared
        cameras.append(_Camera(sequence, frame_offset=offset))

    # The shortest window after its own offset, not FRAMES_PER_WINDOW. The published rate
    # stays in the canonical unit anyway -- ``alerts_per_camera_night`` takes the window
    # length and rescales, which is exactly why it takes that argument.
    frame_count = min(len(c.sequence.frame_paths) - c.frame_offset for c in cameras)
    return cameras, shared, frame_count


# ---------------------------------------------------------------------------------
# The replay loop
# ---------------------------------------------------------------------------------


class Replay:
    """The one process-wide replay loop and the viewers attached to it.

    Public state is read by ``/health`` and by the ``meta`` event; everything else is
    private to the loop task. There is exactly one instance (``REPLAY``), which is why
    ``deploy/plumerank.service`` runs Uvicorn with ``--workers 1``: a second worker would
    give the next judge to connect a second wall, two sets of 40 background models
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
        self._budget: BudgetState | None = None
        self._telemetry = Telemetry()

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
        """
        self._probed = True
        try:
            self._unavailable = self._probe_reason()
        except Exception as exc:
            LOG.exception("probe failed")
            self._unavailable = f"probe failed: {type(exc).__name__}: {exc}"

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
        the tab must not cancel the probe the other is waiting on.
        """
        if self._probed:
            return
        task = self._probe_task
        if task is None or task.done():
            task = self._probe_task = asyncio.create_task(
                asyncio.to_thread(self.probe), name="plumerank-probe"
            )
        await asyncio.shield(task)

    @property
    def available(self) -> bool:
        return self._unavailable is None

    def status(self) -> dict[str, Any]:
        """What ``/health`` and the ``meta`` event both report, written once."""
        return {
            "state": "running" if self._task is not None and not self._task.done() else "idle",
            "viewers": len(self._viewers),
            "replay_id": self.replay_id,
            "ticks": self._ticks,
            "frame_index": self._frame_index,
            "frame_count": self._frame_count,
            "last_tick": self._last_tick_iso,
            "tick_seconds": self.tick_seconds,
            "idle_stop_seconds": self.idle_stop_seconds,
        }

    def live_pipeline(self) -> dict[str, Any]:
        """The degraded-state flag, in the words ``ui.md`` section 7 designed for it."""
        return {
            "available": self.available,
            "detail": None if self.available else f"live pipeline unavailable -- {self._unavailable}",
            "cameras": len(self._cameras),
            "threshold": self._threshold,
            "threshold_source": self._threshold_source,
        }

    # -----------------------------------------------------------------------------
    # Viewers
    # -----------------------------------------------------------------------------

    async def attach(self) -> asyncio.Queue[str]:
        """Register one viewer and start the loop if it is the first. Returns its queue."""
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
        """True when one more viewer would exceed ``MAX_VIEWERS``."""
        return len(self._viewers) >= MAX_VIEWERS

    def detach(self, queue: asyncio.Queue[str]) -> None:
        """Drop one viewer. The loop notices on its next pass and starts its idle timer.

        Deliberately not "stop the loop when this was the last one": a judge reloading the
        page would otherwise tear down 40 warm background models and pay the whole warm-up
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
                payload = await asyncio.to_thread(self._tick)
                self._publish(payload)
                # The remainder of the tick, not a flat sleep: at 40 cameras the pipeline
                # itself takes a large slice of the second, and sleeping a whole one on top
                # would make the wall drift slower than the clock it prints.
                await asyncio.sleep(max(0.0, self.tick_seconds - (time.monotonic() - started)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a wall that stops must say why, not just stop
            LOG.exception("replay loop failed")
            self._unavailable = f"replay loop failed: {type(exc).__name__}: {exc}"
        finally:
            self._task = None

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

        for camera in self._cameras:
            bgr = _read_working_frame(camera.sequence.frame_paths[index + camera.frame_offset])
            grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # The first frame of a replay has no predecessor to register against, so it is
            # its own reference. Warping it against nothing would be a fabricated shift.
            work = grey if camera.prev_grey is None else stabilise(camera.prev_grey, grey).frame

            components = camera.segmenter.apply(work)
            tracks = camera.tracker.update(index, components)
            best = _best_track(components, tracks)

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

        self._alerts_uncapped += sum(1 for s in final if s.hazard >= threshold)
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
            "entries": [e.to_json() for e in entries],
        }

    def _escalate(
        self,
        classifier: TileClassifier,
        classical: list[CameraScore],
        frames: dict[str, np.ndarray],
    ) -> list[CameraScore]:
        """Fold one ONNX tile score into the hazard of the cameras that earned one.

        The ordering is ``emit``'s ordering -- hazard descending, ties on camera id -- and
        not a second sort written here, because ``should_escalate`` is a statement about the
        rank a dispatcher would read, and two ranking rules that are meant to agree and are
        written twice do not stay agreed.
        """
        ordered = sorted(classical, key=lambda s: (-s.hazard, s.camera_id))
        out: list[CameraScore] = []
        for rank, score in enumerate(ordered, start=1):
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
            draw_evidence(
                frames[entry.camera_id],
                component,
                track,
                stats,
                directory / f"{entry.camera_id}.jpg",
            )
            url = (
                f"/evidence/{self.replay_id}/{index}/{entry.camera_id}.jpg"
                f"?t={self._evidence_seq}"
            )
            out.append(dataclasses.replace(entry, evidence_url=url))
        return out

    def _reset_camera_night(self) -> None:
        """A new night gets a new budget. The 80 slots are per camera-night, not per process."""
        self._alerts_uncapped = 0
        self._invocations_at_night_start = (
            self._classifier.invocations if self._classifier is not None else 0
        )
        self._budget = BudgetState([c.camera_id for c in self._cameras])
        self._baseline.reset()

    def _publish_telemetry(self) -> None:
        """The two measured rates, once per completed camera-night.

        ``RealisedAlertRate`` is the **uncapped** count of threshold crossings, which is the
        quantity ``solve_threshold`` was calibrated on. ``emit``'s ``alerted`` flag is the
        budget-capped one, and publishing that instead would make every threshold look
        compliant -- the drift alarm would then be watching a number that cannot drift.
        """
        if not self._cameras or self._frame_count <= 0:
            return
        # Every camera on the wall is replayed for the same _frame_count -- _align()
        # trims each window to a common length so one ignition clock is honest across a
        # roster whose FIgLib windows are 80 and 81 frames -- so the lengths here really
        # are uniform and the list comprehension is not hiding a mixed-length corpus.
        rate = alerts_per_camera_night(
            self._alerts_uncapped, [self._frame_count] * len(self._cameras)
        )
        self._telemetry.put("RealisedAlertRate", rate, unit="Count")
        if self._classifier is not None:
            # Differenced against the count at the start of this night, not the lifetime
            # total: the published fraction is per camera-night, and a lifetime numerator
            # over one night's denominator would climb with uptime rather than measure it.
            calls = self._classifier.invocations - self._invocations_at_night_start
            examined = len(self._cameras) * self._frame_count
            self._telemetry.put("EscalationFraction", calls / examined, unit="None")

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
        """One proof-of-life datapoint. Blocking; call in a thread."""
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
    like one. ``Telemetry`` is a no-op without credentials, so this costs a laptop nothing.
    """
    while True:
        await asyncio.to_thread(REPLAY.heartbeat)
        await asyncio.sleep(HEARTBEAT_SECONDS)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # One OpenCV thread for the whole process, set before anything can do CV work.
    # t4g.small has 2 vCPUs and this process is both the replay loop and the web server:
    # letting OpenCV size its pool from the core count makes every morphologyEx and
    # Laplacian contend with the SSE writer, which shows up as tick-latency variance that
    # has nothing to do with the pipeline -- and on a burstable instance, contention is
    # spend. Parallelism belongs across cameras, not inside one frame.
    app.state.opencv_threads = pin_opencv_threads()
    app.state.started_at = time.time()
    app.state.started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    beat = asyncio.create_task(_heartbeat(), name="plumerank-heartbeat")
    try:
        yield
    finally:
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
        "status": "ok",
        "started_at": getattr(request.app.state, "started_iso", None),
        "uptime_seconds": round(time.time() - started, 3),
        "replay": REPLAY.status(),
        "live_pipeline": REPLAY.live_pipeline(),
    }


@app.get("/api/watchlist")
async def watchlist(request: Request) -> StreamingResponse:
    """The live wall, as server-sent events: one ``meta``, then one ``tick`` per minute."""
    if REPLAY.at_capacity():
        # Refused before the response starts, so the browser's EventSource sees a plain
        # 503 and backs off on Retry-After instead of holding a socket that never ticks.
        raise HTTPException(
            status_code=503,
            detail=f"the live wall is at its {MAX_VIEWERS}-viewer limit; retry shortly",
            headers={"Retry-After": "30"},
        )

    async def stream() -> AsyncIterator[str]:
        queue = await REPLAY.attach()
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
                except asyncio.TimeoutError:
                    # A comment frame, not a tick: it keeps proxies and phone radios from
                    # closing an idle stream, and an SSE client ignores it by definition --
                    # so a degraded wall never has to invent a tick to stay connected.
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

    return StreamingResponse(
        stream(),
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

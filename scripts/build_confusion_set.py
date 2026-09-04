"""Build the 45-window CONFUSE arm: the false-alarm denominator the threshold is solved on.

A confusion set is not a bag of pretty non-fires. It is the *denominator* of the operating
point -- ``rank.solve_threshold`` bisects on this arm until the realised rate hits 2.0
alerts per camera-night, and every headline in the project is quoted at that point. So the
only interesting question about this arm is whether it is hard enough to be worth
calibrating on, and that question has three parts:

**1. Same cameras.** Every CONFUSE window sits on a camera that appears in
``manifest_hold.json``. A confusion set drawn from other cameras would calibrate the
threshold against a different lens, a different horizon and a different exposure schedule
from the one it is then applied to -- a straw man, and ``test_confuse_shares_cameras``
fails on it.

**2. Same hour.** Each window is centred within +/-45 minutes of the local time of day of
its camera's paired HOLD ignition. Marine layer at 06:00 and marine layer at 16:00 are
different physical objects with different illumination; matching the clock is what makes
"one camera-night" the same unit in both arms.

**3. Hard, and hard by a stated measurement.** Picking which fog is the *difficult* fog is
the actual intellectual work here, and it is done by computation rather than by eye: every
candidate is ranked by ``statistic.s3_diffuse_boundary``'s raw value on its own largest
moving region -- the exact statistic that is supposed to reject fog, applied to fog. The
CONFUSE arm is therefore adversarially selected against this project's own discriminator,
which makes the solved threshold conservative and the published rate honest. The measured
value is written into the manifest beside every window, so the selection can be disputed
without re-reading the archive.

How the archive is walked, and why it is not walked
---------------------------------------------------
The MTA is ``https://cdn.hpwren.ucsd.edu/MTA/<camera>/large/<YYYYMMDD>/Q<n>/<epoch>.jpg``,
where the **file name** is a UTC epoch but the **directory** is San Diego local: the date
is the local date and ``Q<n>`` is the local three-hour block numbered from 1 at local
midnight, so n runs 1..8. See ``quarter_directory`` -- the 60 URLs in
``spike/urls_fog.txt`` all sit in the one band where the older UTC six-hour reading
happened to give the same answer, so they could not have caught this. The bucket has no
directory index: it answers ``AccessDenied`` to a bare directory, to ``?list-type=2`` and
to ``index.html`` alike, and the origin host 404s. So enumeration cannot list; it probes.
Frames land on a 60 s grid at a per-camera phase -- one camera writes 31 s past the
minute, another 55 s -- so the first slot of a window is searched across a whole minute to
discover that phase and every later slot is searched only within ``JITTER_SECONDS`` of it,
carrying the phase forward as it drifts. Some cameras in the HOLD roster are absent from
the MTA altogether, which is why a candidate window is cheaply screened on ``PROBE_SLOTS``
frames before anything commits to enumerating all 81. On a camera that IS present the
minute grid is complete: an 81-frame window on hp-s-mobo-c enumerated 81 of 81.

Degrading honestly
------------------
This script writes ``data/manifest_confuse.json`` once, at the end, or not at all. There is
no partial manifest: a half-built CONFUSE arm would still load, still solve a threshold and
still print a number, and that number would be wrong in a way nothing downstream could see.
An unreachable archive, a class that cannot be filled, or a window that cannot produce
exactly 81 frames all exit non-zero with the reason.

Usage::

    python scripts/build_confusion_set.py --end-date 2026-09-01
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import cv2
import numpy as np

# `python scripts/foo.py` puts scripts/ on sys.path, not the repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPTS = pathlib.Path(__file__).resolve().parent
for _path in (str(_REPO_ROOT), str(_SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from plumerank.constants import (  # noqa: E402
    CC_CONNECTIVITY,
    EPS,
    FRAMES_PER_WINDOW,
    IGNITION_FRAME_INDEX,
    MIN_COMPONENT_AREA_PX,
    MIN_COMPONENT_EXTENT_PX,
    MORPH_CLOSE_KERNEL,
    MORPH_OPEN_KERNEL,
    SECONDS_PER_FRAME,
    WORK_HEIGHT,
    WORK_WIDTH,
)
from plumerank.statistic import s3_diffuse_boundary  # noqa: E402
from plumerank.types import Component  # noqa: E402

# fetch.py owns the HTTP client and the polite-client settings. Importing it rather than
# copying them keeps the User-Agent and the retry policy in one place; the build contract
# fixes scripts/ at six files, so there is no seventh module for them to live in.
from fetch import (  # noqa: E402
    RETRY_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
    USER_AGENT,
    FetchError,
    fetch_with_retry,
)

__all__ = [
    "CLASS_QUOTAS",
    "Candidate",
    "Measurement",
    "classify",
    "frame_path",
    "enumerate_window",
    "hardness",
    "local_minutes_of_day",
    "measure",
    "minutes_apart",
    "paired_hold_windows",
    "probe_exists",
    "build",
    "main",
]

# ---------------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------------

MTA_BASE_URL: str = "https://cdn.hpwren.ucsd.edu/MTA/"

#: HPWREN's cameras are in San Diego County. Every "time of day" in this file is local
#: there, because the thing being matched is the sun angle, not a UTC offset.
PACIFIC = ZoneInfo("America/Los_Angeles")

CREDIT: str = "Imagery: HPWREN, http://hpwren.ucsd.edu/"

# ---------------------------------------------------------------------------------
# Search budget
# ---------------------------------------------------------------------------------
#
# These govern how much of somebody else's archive this one-off script reads. They are not
# pipeline numbers -- nothing at inference time consults them -- but they do decide which
# windows exist, so every one of them is written into the manifest under
# ``search_budget``. results/constants.sha256 freezes what the detector computes;
# manifest_confuse.json freezes what the corpus is, and the two are different artefacts.

#: A camera writes on a 60 s grid at ITS OWN phase -- one camera lands 31 s past the
#: minute, another 55 s -- and a window's predicted epochs are built from a local wall
#: clock, which knows nothing about that. So the FIRST slot of every window is searched
#: across a whole minute to discover the phase; searching only the jitter would find
#: nothing on most cameras and this script would report an empty archive.
PHASE_SEARCH_SECONDS: int = 30

#: Seconds either side of a predicted epoch, once the phase is known. Measured on the live
#: archive: a missing slot is usually a frame one or two seconds off the grid rather than
#: an absent frame, and the phase drifts slowly across a day.
JITTER_SECONDS: int = 3

#: Slots probed to screen a candidate before committing to the full 81. Spread across the
#: window so a hole at either end is seen.
PROBE_SLOTS: int = 7
PROBE_MIN_PRESENT: int = 5

#: Sockets held open against the CDN while probing. Measured on the live archive: a HEAD
#: costs ~0.6 s of round trip whether or not the connection is reused -- keep-alive changes
#: nothing, so the walk is latency-bound and not bandwidth-bound. One request at a time
#: makes the sweep arithmetically impossible: enumerating one 81-frame window alone is 203
#: HEADs and 45 of them is the smallest thing this script does. Six is fetch.py's
#: politeness budget minus a margin, and it is a cap on a public academic service rather
#: than a tuning knob: raising it is a decision about somebody else's bandwidth, so it is
#: written into the manifest with the rest of the search budget.
MAX_PROBE_CONCURRENCY: int = 6

#: Days back from ``--end-date``, and the stride between them. Consecutive days are highly
#: correlated -- the same marine layer sits over the same camera for a week -- so the sweep
#: strides rather than taking the most recent N days.
CANDIDATE_DAYS: int = 60
CANDIDATE_DAY_STRIDE: int = 3

#: Offsets from the paired HOLD time of day, in minutes. All inside the +/-45 min the
#: contract allows, so a camera whose exact matching minute is missing from the archive
#: still has somewhere to go.
TOD_OFFSETS_MINUTES: tuple[int, ...] = (0, -30, 30)

#: Candidates measured per camera before giving up on it. Each one costs PROBE_SLOTS
#: probes plus up to PROBE_SLOTS image GETs.
MAX_CANDIDATES_PER_CAMERA: int = 6

# ---------------------------------------------------------------------------------
# The six classes, and the measurement that assigns each one
# ---------------------------------------------------------------------------------

CLASS_QUOTAS: dict[str, int] = {
    "fog_marine": 18,
    "dust": 8,
    "cloud_shadow": 7,
    "exposure_hunting": 5,
    "low_cloud": 4,
    "lens_artefact": 3,
}

#: All thresholds below are on the 0-255 luma scale at WORK_WIDTH x WORK_HEIGHT, chosen a
#: priori from the physics of each process rather than fitted -- there is nothing to fit
#: them on, because the labels they produce ARE the corpus. Every feature they are compared
#: against is recorded per window, so a reader who disagrees with a cutline can re-derive
#: the whole assignment from the manifest.

#: A fixed camera's sky is the one region that should carry no persistent structure. A
#: ridgeline test would fire on every frame of every camera; structure in the SKY that does
#: not move over 81 minutes is on the glass. Clouds fail it because clouds move.
LENS_SKY_GRADIENT_MIN: float = 12.0
LENS_STATIC_TEMPORAL_STD_MAX: float = 2.0
LENS_SKY_STRUCTURE_MIN: float = 0.02

#: Auto-exposure hunting is a whole-frame brightness excursion that REVERSES: the shutter
#: overshoots, then corrects. The magnitude alone would not do -- probe frames are ~13 min
#: apart, and over thirteen minutes a rising sun moves whole-frame luma by more than this
#: while a marine layer sits under it, so a plain max-step test labels dawn fog as a broken
#: camera and spends the fog quota on it. Every atmospheric and solar process is monotone
#: over the span of one window; only the instrument goes up and then back down.
EXPOSURE_REVERSAL_MIN: float = 8.0

#: Dust is an obscuring layer that is brown. Fog is an obscuring layer that is grey. The
#: red-minus-blue difference and the saturation separate them; nothing else does.
DUST_WARMTH_MIN: float = 8.0
DUST_SATURATION_MIN: float = 40.0

#: Below this mean gradient energy the scene's structure has been eaten by something
#: airborne. Above it the air is clear and any dark patches are shadows cast onto it.
HAZE_EDGE_ENERGY_MAX: float = 18.0

#: Sky-over-ground brightness. A marine layer sits BELOW the camera and lifts the ground
#: to the sky's brightness or past it; a low cloud deck sits above and leaves the ground
#: dark. One number, and it is the only thing that distinguishes the two classes.
FOG_SKY_GROUND_RATIO_MAX: float = 1.05

#: Sky and ground are the top and bottom third of the frame. The middle third is the
#: horizon band, where a camera sees both, and averaging it into either would blunt the one
#: number that separates a marine layer from a low cloud deck.
SKY_GROUND_BAND_DIVISOR: int = 3

#: Blur applied to both endpoint frames before differencing them. Wide enough that JPEG
#: blocking and sensor grain do not survive into the Otsu threshold, narrow enough that a
#: fog bank's leading edge does -- the edge is the only thing S3 is then asked about.
DIFF_BLUR_KERNEL_PX: int = 5

#: Cloud shadow: the air is clear, but a large soft luminance patch has moved across the
#: scene since the first probe. Measured on a heavily blurred frame so that terrain
#: texture, which does not move, contributes nothing.
SHADOW_LOWPASS_SIGMA_PX: float = 24.0
SHADOW_MOTION_MIN: float = 6.0

#: The lowpass is computed on a decimated image and not at working resolution. A sigma-24
#: Gaussian is a ~145-tap kernel per axis, which OpenCV runs in 13 ms per frame -- 5.5 s
#: across one selection run, and the slowest thing in this file by two orders of magnitude.
#: INTER_AREA decimation IS a box lowpass, so blurring the 1/8 image with sigma/8 carries
#: the same low-frequency field at 1/60th the cost. The factor divides both WORK dimensions
#: exactly, so no rounding enters the measurement.
SHADOW_LOWPASS_DECIMATION: int = 8

#: Order matters and is not alphabetical. The two instrument faults are checked first
#: because a camera with a droplet on the lens or a hunting shutter is that, whatever the
#: weather is doing behind it; then colour separates dust from the neutral layers; then
#: height separates fog from low cloud; cloud shadow is last because it is the only class
#: defined on a CLEAR scene and therefore cannot be confused with the others.
_CLASS_ORDER: tuple[str, ...] = (
    "lens_artefact",
    "exposure_hunting",
    "dust",
    "fog_marine",
    "low_cloud",
    "cloud_shadow",
)


# ---------------------------------------------------------------------------------
# Clock arithmetic
# ---------------------------------------------------------------------------------


def local_minutes_of_day(epoch: int) -> int:
    """Minutes past local midnight in San Diego, for a unix epoch."""
    moment = datetime.fromtimestamp(int(epoch), PACIFIC)
    return moment.hour * 60 + moment.minute


def minutes_apart(a: int, b: int) -> int:
    """Circular distance between two times of day, in minutes.

    Circular and not ``abs(a - b)``: 23:50 and 00:10 are twenty minutes apart, and the
    linear answer of 1420 would silently reject a legal pre-dawn pairing.
    """
    delta = abs(a - b) % (24 * 60)
    return min(delta, 24 * 60 - delta)


def frame_epoch(frame_name: str) -> int:
    """The unix epoch a frame name carries, for either archive.

    FIgLib names are ``<epoch>_<signed offset>.jpg`` and MTA names are ``<epoch>.jpg``, so
    the leading integer is the epoch in both and the tests can re-derive every time-of-day
    claim in the manifest from the frame lists alone.
    """
    stem = pathlib.PurePosixPath(frame_name).stem
    return int(stem.split("_")[0])


#: The archive bins a camera's day into EIGHT three-hour directories, not four six-hour
#: ones, and it bins them by San Diego local time rather than by UTC. Probed against the
#: live bucket rather than assumed: on hp-s-mobo-c, 2026-08-26, one frame per local hour
#: was found at every hour of the day under ``local_date/Q(local_hour // 3 + 1)``, and the
#: Q3/Q4 edge was walked minute by minute to 08:59 -> Q3, 09:00 -> Q4.
QUARTER_HOURS: int = 3


def quarter_directory(epoch: int) -> str:
    """``<YYYYMMDD>/Q<n>`` for one epoch, in San Diego local time.

    **Corrected against the live archive.** The 60 URLs in ``spike/urls_fog.txt`` were the
    only ground truth this file originally had, and they are not enough to pin the rule:
    every one of them falls in local hours 06-08, where ``UTC hour // 6 + 1`` and
    ``local hour // 3 + 1`` happen to agree on Q3. They disagree everywhere else, and the
    disagreement is silent -- a wrong quarter answers 403, which this file's own
    ``probe_exists`` reads as "the archive does not have that frame". The first run against
    the real bucket therefore reported a sparse archive with hour-long holes and no data
    at all after local noon, all of which was this function.

    The rule that fits every hour of a probed day is local: the date directory is the
    LOCAL date, and Q is the local three-hour block numbered from 1 at local midnight, so
    Q runs 1..8 and not 1..4. Local also makes the DST transition correct for free -- the
    quarter follows the camera's own clock, which is the clock the archive names its
    directories with.
    """
    moment = datetime.fromtimestamp(int(epoch), PACIFIC)
    return f"{moment:%Y%m%d}/Q{moment.hour // QUARTER_HOURS + 1}"


def frame_path(epoch: int) -> str:
    """The MTA key under ``large/``, which is what a manifest window stores as a frame."""
    return f"large/{quarter_directory(epoch)}/{int(epoch)}.jpg"


def frame_url(camera_id: str, epoch: int) -> str:
    return MTA_BASE_URL + camera_id + "/" + frame_path(epoch)


# ---------------------------------------------------------------------------------
# Walking an archive that will not list itself
# ---------------------------------------------------------------------------------


def probe_exists(url: str, timeout: float = 15.0) -> bool:
    """Does this key exist? A HEAD, because the answer is one bit and the file is 500 KB.

    A missing key answers 403 rather than 404 on this bucket (listing is disabled, so S3
    will not confirm absence), and both are treated as "no" on the first answer -- there
    is nothing transient about them and asking again is neither useful nor polite. A 5xx
    or a transport error is NOT treated as "no": it is retried on the same exponential
    backoff ``fetch_with_retry`` uses and then raised, because reading a flaky network as
    an empty archive is precisely how this script would quietly build a smaller corpus
    than it claims.

    The retry is not decoration. A full sweep is tens of thousands of HEADs over a couple
    of hours across six sockets, and one transient DNS failure -- observed, ``[Errno 8]
    nodename nor servname provided`` -- would otherwise discard the whole run at whatever
    minute it happened to land on.
    """
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    last: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=timeout
            ) as response:
                return 200 <= response.status < 300
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                return False
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt + 1 < RETRY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
    raise FetchError(f"HEAD {url} failed after {RETRY_ATTEMPTS} attempts: {last}")


def _jitter_order(limit: int) -> list[int]:
    """0, -1, +1, -2, +2, ... -- the predicted epoch first, then outward."""
    order = [0]
    for step in range(1, limit + 1):
        order.extend((-step, step))
    return order


def _slot_epochs(centre_epoch: int, n_frames: int) -> list[int]:
    """The predicted epoch of every slot in a window centred on ``centre_epoch``."""
    step = int(SECONDS_PER_FRAME)
    return [
        centre_epoch + (i - IGNITION_FRAME_INDEX) * step for i in range(n_frames)
    ]


def _chunks(items: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    """``items`` in fixed-size groups, order preserved."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _search_slot(camera_id: str, base_epoch: int, width: int) -> int | None:
    """The epoch nearest ``base_epoch`` that exists within ``width`` seconds, or None.

    Sequential inside one worker and concurrent across workers, rather than the reverse:
    "nearest wins" is an ordering constraint, and a slot that hits on its first probe --
    the common case once the phase is known -- must not have paid for six more.
    """
    for delta in _jitter_order(width):
        epoch = base_epoch + delta
        if probe_exists(frame_url(camera_id, epoch)):
            return epoch
    return None


def _first_existing(camera_id: str, epochs: Sequence[int]) -> int | None:
    """The first epoch of ``epochs`` that exists, probing ahead under a bounded pool.

    Used only for the full-minute phase discovery, where ``epochs`` is 61 long and the
    answer is uniformly distributed inside it, so probing strictly one at a time spends
    thirty seconds of wall clock to save a handful of HEADs. Chunked and not fully
    speculative, so a hit in the first chunk still costs six requests and not sixty-one:
    the result is identical to the sequential scan, only the wasted tail differs.
    """
    for chunk in _chunks(epochs, MAX_PROBE_CONCURRENCY):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            hits = list(pool.map(lambda e: probe_exists(frame_url(camera_id, e)), chunk))
        for epoch, hit in zip(chunk, hits):
            if hit:
                return epoch
    return None


def enumerate_window(
    camera_id: str,
    centre_epoch: int,
    slots: Sequence[int],
    n_frames: int = FRAMES_PER_WINDOW,
    known: dict[int, int] | None = None,
    phase: int | None = None,
    min_present: int | None = None,
) -> dict[int, int]:
    """Which of ``slots`` exist, as ``{slot index: actual epoch}``.

    Two search widths, and the difference is the whole trick. Until a frame has been found
    the camera's phase is unknown, so the slot is searched across the full minute
    (``PHASE_SEARCH_SECONDS`` either side). After that the phase is carried forward and
    each further slot costs one HEAD in the common case, with the narrow jitter search
    reserved for the frames that really are missing -- 81 full-minute searches per window
    would be 5,000 requests at somebody else's expense for information already in hand.

    ``known`` seeds that phase from a candidate's screening probes, so the full
    enumeration does not re-discover what the screen already paid for. ``phase`` seeds it
    from a DIFFERENT window on the same camera, which is legitimate because the phase is a
    property of the camera and not of the day: a predicted epoch is always a whole local
    minute and the archive's offsets are whole hours, so the phase is just the second of
    the minute each camera writes at. Measured across two months of the live archive it
    moves by one or two seconds and no more -- hp-s writes at :31 in July and at :31 in
    September -- which is well inside ``JITTER_SECONDS``. Without it every candidate day
    on the sweep re-pays the full-minute discovery, and the sweep does not finish.

    ``min_present`` abandons a candidate as soon as it cannot reach that many slots. It is
    lossless -- the abandoned candidate would have been discarded by the same number -- and
    it caps an empty camera-day at three slots instead of seven. Two of the 45 HOLD cameras
    (ml-n-mobo-c, ml-w-mobo-c) are absent from the MTA entirely, so every one of their
    sixty candidate days runs this path.
    """
    predicted = _slot_epochs(centre_epoch, n_frames)
    found: dict[int, int] = dict(known or {})
    if found:
        first = min(found)
        phase = found[first] - predicted[first]

    pending = [index for index in slots if index not in found]

    # Phase discovery. The original walked slot by slot at full width until something was
    # found, and so does this: a camera-day can be missing its first slot and hold the
    # rest.
    while phase is None and pending:
        index = pending.pop(0)
        epoch = _first_existing(
            camera_id,
            [predicted[index] + d for d in _jitter_order(PHASE_SEARCH_SECONDS)],
        )
        if epoch is not None:
            found[index] = epoch
            phase = epoch - predicted[index]
        elif min_present is not None and len(found) + len(pending) < min_present:
            return found

    while pending:
        if min_present is not None and len(found) + len(pending) < min_present:
            break
        # Never probe further into the window than the point at which the candidate could
        # already be abandoned. A screening pass allows ``len(slots) - min_present``
        # misses, so once that many have accumulated the next miss ends the candidate and
        # a full chunk of six would have bought five answers nobody reads.
        width = MAX_PROBE_CONCURRENCY
        if min_present is not None:
            width = min(width, max(1, len(found) + len(pending) - min_present + 1))
        chunk = pending[:width]
        pending = pending[width:]
        assert phase is not None  # the loop above either set it or emptied ``pending``
        carried = phase
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            hits = list(
                pool.map(
                    lambda i: _search_slot(
                        camera_id, predicted[i] + carried, JITTER_SECONDS
                    ),
                    chunk,
                )
            )
        for index, epoch in zip(chunk, hits):
            if epoch is not None:
                found[index] = epoch
                # Carry the drift forward from the latest slot resolved, as the sequential
                # walk did. Slots are handed out in increasing order, so this is the same
                # carry to within one chunk -- and the drift over one chunk is under a
                # second on every camera measured.
                phase = epoch - predicted[index]
    return found


# ---------------------------------------------------------------------------------
# Measuring a candidate
# ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Measurement:
    """What was measured on a candidate's probe frames, and what it implies."""

    features: dict[str, float]
    label: str | None
    """The class, or ``None`` for a window that is not any recognised confuser -- an
    ordinary clear afternoon has no business in a false-alarm denominator."""
    hardness: float
    """Raw S3 on the largest moving region. Higher means this window looks more like smoke
    to the statistic that is supposed to say it is not."""


def _decode(data: bytes) -> np.ndarray:
    """Archive bytes to a working-resolution BGR frame.

    Decoded and downscaled here rather than at full 3072x2048 because the classification
    below has to see what the pipeline sees: an edge-energy threshold measured at source
    resolution would be a different number by a factor of six.
    """
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FetchError("cv2.imdecode refused a frame the archive served as an image")
    return cv2.resize(image, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA)


def _largest_moving_component(first: np.ndarray, last: np.ndarray) -> Component | None:
    """The biggest thing that changed between two greyscale frames.

    Frame differencing and not ``segment.Segmenter``: the background subtractor needs
    WARMUP_FRAMES of history and this stage has seven frames spread over 81 minutes on
    purpose. The component is only ever used to place S3's boundary band, and Otsu on a
    blurred difference puts that band on the same soft edge the subtractor would.
    """
    kernel = (DIFF_BLUR_KERNEL_PX, DIFF_BLUR_KERNEL_PX)
    delta = cv2.absdiff(
        cv2.GaussianBlur(first, kernel, 0), cv2.GaussianBlur(last, kernel, 0)
    )
    _, mask = cv2.threshold(delta, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=CC_CONNECTIVITY
    )
    best: Component | None = None
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < MIN_COMPONENT_AREA_PX:
            continue
        if w < MIN_COMPONENT_EXTENT_PX or h < MIN_COMPONENT_EXTENT_PX:
            continue
        if best is None or area > best.area:
            best = Component(
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                cx=float(centroids[label][0]),
                cy=float(centroids[label][1]),
            )
    return best


def hardness(frames: Sequence[np.ndarray]) -> float:
    """Raw S3 on the largest moving region -- this project's own discriminator, on a non-fire.

    Deliberately the raw statistic and not ``rank.fuse``: the fusion weights are fitted on
    DEV, so ranking the confusion set with them would make the corpus a function of a fit
    and the fit a function of the corpus. ``S3_BAND_FRACTION`` is a priori and was frozen
    before the Sep 1 gate, so selecting on raw S3 borrows nothing that was learned.

    A window with nothing moving scores 0.0, which is the truth: it cannot fool anything.
    A window whose S3 is ``nan`` is a different statement -- the band geometry did not fit
    the component -- and it is raised rather than folded into that 0.0, because ranking the
    unmeasurable alongside the harmless is exactly how the softest windows in the archive
    would end up presented as the hardest. ``MIN_COMPONENT_EXTENT_PX`` should make it
    unreachable; if it fires, the filter and the statistic have drifted apart.
    """
    if len(frames) < 2:
        return 0.0
    first = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    last = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    component = _largest_moving_component(first, last)
    if component is None:
        return 0.0
    _, raw = s3_diffuse_boundary(last, component)
    if math.isnan(raw):
        raise FetchError(
            f"s3_diffuse_boundary returned nan on a {component.w}x{component.h} component "
            f"that cleared MIN_COMPONENT_EXTENT_PX={MIN_COMPONENT_EXTENT_PX}. No manifest "
            f"was written."
        )
    return float(raw)


def measure(frames: Sequence[np.ndarray]) -> Measurement:
    """Eight numbers per candidate window, then the class they imply."""
    if len(frames) < 2:
        raise FetchError("a candidate needs at least two probe frames to be measured")

    greys = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    stack = np.stack([g.astype(np.float32) for g in greys])

    means = [float(g.mean()) for g in greys]
    edge = float(
        np.mean(
            [
                np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
                + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
                for g in greys
            ]
        )
    )
    third = WORK_HEIGHT // SKY_GROUND_BAND_DIVISOR
    sky = float(np.mean([g[:third].mean() for g in greys]))
    ground = float(np.mean([g[-third:].mean() for g in greys]))

    warmth = float(
        np.mean([f[:, :, 2].astype(np.float32).mean() - f[:, :, 0].astype(np.float32).mean() for f in frames])
    )
    saturation = float(
        np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 1].mean() for f in frames])
    )

    # The smaller leg of each direction change, so a 30-up-then-2-down wobble scores 2 and
    # not 30: a reversal is only as big as the part of it that came back.
    diffs = [means[i] - means[i - 1] for i in range(1, len(means))]
    reversal = max(
        (
            min(abs(a), abs(b))
            for a, b in zip(diffs, diffs[1:])
            if a * b < 0.0
        ),
        default=0.0,
    )

    small = (
        WORK_WIDTH // SHADOW_LOWPASS_DECIMATION,
        WORK_HEIGHT // SHADOW_LOWPASS_DECIMATION,
    )
    lowpass = [
        cv2.GaussianBlur(
            cv2.resize(g, small, interpolation=cv2.INTER_AREA),
            (0, 0),
            SHADOW_LOWPASS_SIGMA_PX / SHADOW_LOWPASS_DECIMATION,
        ).astype(np.float32)
        for g in greys
    ]
    shadow_motion = float(
        np.mean([np.abs(lp - lowpass[0]).mean() for lp in lowpass[1:]])
    )

    temporal_std = stack.std(axis=0)
    mean_frame = stack.mean(axis=0)
    sky_gradient = np.abs(cv2.Sobel(mean_frame[:third], cv2.CV_32F, 1, 0, ksize=3)) + np.abs(
        cv2.Sobel(mean_frame[:third], cv2.CV_32F, 0, 1, ksize=3)
    )
    structured = sky_gradient >= LENS_SKY_GRADIENT_MIN
    persistent = structured & (temporal_std[:third] <= LENS_STATIC_TEMPORAL_STD_MAX)
    sky_structure = float(persistent.sum()) / float(persistent.size)

    features = {
        "luma_mean": round(float(np.mean(means)), 3),
        "edge_energy": round(edge, 3),
        "sky_ground_ratio": round(sky / (ground + EPS), 4),
        "warmth_r_minus_b": round(warmth, 3),
        "saturation_mean": round(saturation, 3),
        "luma_reversal_max": round(reversal, 3),
        "shadow_motion": round(shadow_motion, 3),
        "sky_persistent_structure": round(sky_structure, 5),
    }
    return Measurement(
        features=features, label=classify(features), hardness=hardness(frames)
    )


def classify(features: dict[str, float]) -> str | None:
    """Which confuser class these features describe, or ``None``.

    Every branch is one comparison against one named threshold, evaluated in ``_CLASS_ORDER``.
    A decision tree with tuned depths would assign these labels better and would be
    unreviewable; a judge has to be able to read this function and disagree with it.
    """
    for label in _CLASS_ORDER:
        if label == "lens_artefact":
            if features["sky_persistent_structure"] >= LENS_SKY_STRUCTURE_MIN:
                return label
        elif label == "exposure_hunting":
            if features["luma_reversal_max"] >= EXPOSURE_REVERSAL_MIN:
                return label
        elif label == "dust":
            if (
                features["warmth_r_minus_b"] >= DUST_WARMTH_MIN
                and features["saturation_mean"] >= DUST_SATURATION_MIN
            ):
                return label
        elif label == "fog_marine":
            if (
                features["edge_energy"] <= HAZE_EDGE_ENERGY_MAX
                and features["sky_ground_ratio"] <= FOG_SKY_GROUND_RATIO_MAX
            ):
                return label
        elif label == "low_cloud":
            if features["edge_energy"] <= HAZE_EDGE_ENERGY_MAX:
                return label
        elif label == "cloud_shadow":
            if features["shadow_motion"] >= SHADOW_MOTION_MIN:
                return label
    return None


def criterion_rule(label: str) -> str:
    """The predicate that put a window in its class, as text, for the manifest."""
    return {
        "lens_artefact": (
            f"sky_persistent_structure >= {LENS_SKY_STRUCTURE_MIN} (fraction of sky "
            f"pixels with gradient >= {LENS_SKY_GRADIENT_MIN} and temporal std <= "
            f"{LENS_STATIC_TEMPORAL_STD_MAX})"
        ),
        "exposure_hunting": f"luma_reversal_max >= {EXPOSURE_REVERSAL_MIN}",
        "dust": (
            f"warmth_r_minus_b >= {DUST_WARMTH_MIN} and saturation_mean >= "
            f"{DUST_SATURATION_MIN}"
        ),
        "fog_marine": (
            f"edge_energy <= {HAZE_EDGE_ENERGY_MAX} and sky_ground_ratio <= "
            f"{FOG_SKY_GROUND_RATIO_MAX}"
        ),
        "low_cloud": (
            f"edge_energy <= {HAZE_EDGE_ENERGY_MAX} and sky_ground_ratio > "
            f"{FOG_SKY_GROUND_RATIO_MAX}"
        ),
        "cloud_shadow": (
            f"edge_energy > {HAZE_EDGE_ENERGY_MAX} and shadow_motion >= "
            f"{SHADOW_MOTION_MIN}"
        ),
    }[label]


# ---------------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    camera_id: str
    centre_epoch: int
    offset_minutes: int
    paired_hold_sequence_id: str
    paired_hold_t0_epoch: int

    @property
    def date(self) -> str:
        return f"{datetime.fromtimestamp(self.centre_epoch, PACIFIC):%Y%m%d}"

    @property
    def sequence_id(self) -> str:
        return f"confuse_{self.camera_id}_{self.date}"


@dataclass(slots=True)
class _Scored:
    candidate: Candidate
    measurement: Measurement
    probes: dict[int, int] = field(default_factory=dict)


def paired_hold_windows(hold_manifest: dict) -> dict[str, dict]:
    """One HOLD window per camera -- the one a CONFUSE window on that camera is paired to.

    A camera with several HOLD windows gets its earliest by sequence id, deterministically.
    Which one hardly matters for the sun angle; that the choice is a rule rather than a
    coin flip is what lets ``test_confuse_shares_cameras`` recompute the pairing.
    """
    paired: dict[str, dict] = {}
    for window in sorted(hold_manifest["windows"], key=lambda w: w["sequence_id"]):
        paired.setdefault(window["camera_id"], window)
    return paired


def hold_t0_epoch(window: dict) -> int:
    """The unix epoch of a HOLD window's ignition frame."""
    return frame_epoch(window["frames"][window["t0_index"]])


def candidate_days(end: date) -> list[date]:
    """The days swept, newest first. Reads the budget globals at call time on purpose --
    a default argument would freeze them at import and make the sweep untestable."""
    return [
        end - timedelta(days=CANDIDATE_DAY_STRIDE * i)
        for i in range(CANDIDATE_DAYS // CANDIDATE_DAY_STRIDE)
    ]


def candidate_centre_epoch(day: date, minutes_of_day: int, offset_minutes: int) -> int:
    """The epoch at which ``day`` reaches a given local time of day, plus an offset.

    Built from a localised ``datetime`` rather than by adding seconds to midnight UTC, so
    that a window straddling a DST boundary lands at the right sun angle instead of an hour
    off -- which for a marine-layer camera is the difference between fog and no fog.
    """
    total = minutes_of_day + offset_minutes
    midnight = datetime(day.year, day.month, day.day, tzinfo=PACIFIC)
    return int((midnight + timedelta(minutes=total)).timestamp())


def _figlib_fire_dates(manifests: Iterable[dict]) -> set[tuple[str, str]]:
    """``(camera, YYYYMMDD)`` pairs that FIgLib records a fire on.

    Excluded from the candidate sweep. It is a weak filter -- FIgLib catalogues 521 fires,
    not every fire -- and the manifest says so rather than claiming the arm is certified
    fire-free. The class predicates are the second filter: a plume does not look like
    marine layer to any of them.
    """
    dates: set[tuple[str, str]] = set()
    for manifest in manifests:
        for window in manifest.get("windows", []):
            dates.add((window["camera_id"], window["date"]))
    return dates


# ---------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------


def _probe_slot_indices(n_frames: int = FRAMES_PER_WINDOW, n: int = PROBE_SLOTS) -> list[int]:
    return [round(i * (n_frames - 1) / (n - 1)) for i in range(n)]


def _score_camera(
    camera_id: str,
    paired: dict,
    days: Sequence[date],
    excluded: set[tuple[str, str]],
) -> list[_Scored]:
    """Screen up to MAX_CANDIDATES_PER_CAMERA windows on one camera and measure each."""
    t0_epoch = hold_t0_epoch(paired)
    minutes = local_minutes_of_day(t0_epoch)
    slots = _probe_slot_indices()
    scored: list[_Scored] = []
    #: Learned once from whichever candidate day first yields a frame, then reused for
    #: every later day on this camera. See ``enumerate_window``: the phase is the camera's
    #: second-of-the-minute, not the day's, and re-discovering it 60 times per camera is
    #: 60 full-minute searches spent on a number that does not change.
    phase: int | None = None

    for day in days:
        if len(scored) >= MAX_CANDIDATES_PER_CAMERA:
            break
        for offset in TOD_OFFSETS_MINUTES:
            if len(scored) >= MAX_CANDIDATES_PER_CAMERA:
                break
            centre = candidate_centre_epoch(day, minutes, offset)
            candidate = Candidate(
                camera_id=camera_id,
                centre_epoch=centre,
                offset_minutes=offset,
                paired_hold_sequence_id=paired["sequence_id"],
                paired_hold_t0_epoch=t0_epoch,
            )
            if (camera_id, candidate.date) in excluded:
                continue
            probes = enumerate_window(
                camera_id, centre, slots, phase=phase, min_present=PROBE_MIN_PRESENT
            )
            if probes and phase is None:
                first = min(probes)
                phase = probes[first] - _slot_epochs(centre, FRAMES_PER_WINDOW)[first]
            if len(probes) < PROBE_MIN_PRESENT:
                continue
            frames = [
                _decode(fetch_with_retry(frame_url(camera_id, probes[i])))
                for i in sorted(probes)
            ]
            measurement = measure(frames)
            if measurement.label is None:
                continue
            scored.append(_Scored(candidate, measurement, probes))
            break  # one candidate per day per camera; the next offset is the same weather
    return scored


def _select(
    pool: dict[str, list[_Scored]],
    quotas: dict[str, int],
) -> dict[str, _Scored]:
    """Assign one camera to each slot of each class, scarcest class first.

    Scarcest first and not alphabetical, because the classes are wildly unequal in supply:
    three cameras in the roster may ever show a lens artefact, and filling the 18 fog slots
    first would consume the only cameras that could have supplied them. Within a class the
    order is hardness descending -- the whole point of the arm.

    Returns ``{sequence_id: scored}``. Raises when a class cannot be filled, because a
    CONFUSE arm with 14 fog windows instead of 18 is a different denominator from the one
    every published rate is quoted against.
    """
    by_class: dict[str, list[_Scored]] = {label: [] for label in quotas}
    for entries in pool.values():
        for scored in entries:
            label = scored.measurement.label
            if label in by_class:
                by_class[label].append(scored)
    for label in by_class:
        by_class[label].sort(
            key=lambda s: (-s.measurement.hardness, s.candidate.sequence_id)
        )

    order = sorted(quotas, key=lambda label: (len(by_class[label]), label))
    chosen: dict[str, _Scored] = {}
    taken: set[str] = set()
    for label in order:
        need = quotas[label]
        for scored in by_class[label]:
            if need == 0:
                break
            if scored.candidate.camera_id in taken:
                continue
            taken.add(scored.candidate.camera_id)
            chosen[scored.candidate.sequence_id] = scored
            need -= 1
        if need:
            raise FetchError(
                f"class {label!r} needs {quotas[label]} windows and the archive offered "
                f"{len(by_class[label])} candidates on unused cameras -- {need} short. "
                f"No manifest was written."
            )
    return chosen


# ---------------------------------------------------------------------------------
# Building the manifest
# ---------------------------------------------------------------------------------


def _window_json(scored: _Scored, epochs: dict[int, int]) -> dict:
    candidate = scored.candidate
    centre = epochs[IGNITION_FRAME_INDEX]
    paired_minutes = local_minutes_of_day(candidate.paired_hold_t0_epoch)
    local_minutes = local_minutes_of_day(centre)
    label = scored.measurement.label
    assert label is not None  # _score_camera drops unlabelled candidates
    return {
        "sequence_id": candidate.sequence_id,
        "camera_id": candidate.camera_id,
        "date": candidate.date,
        # No ignition, ever: that is what makes this arm a false-alarm denominator.
        # types.Sequence reads a null t0_index as "nothing ignites here".
        "t0_index": None,
        "t0_offset_seconds": None,
        "centre_index": IGNITION_FRAME_INDEX,
        "n_frames": len(epochs),
        "url_prefix": MTA_BASE_URL + candidate.camera_id + "/",
        "class": label,
        "criterion": {
            "rule": criterion_rule(label),
            "hardness_raw_s3": round(scored.measurement.hardness, 4),
            "hardness_note": (
                "Raw S3 (statistic.s3_diffuse_boundary) on the largest region that moved "
                "between the first and last probe frame. Candidates were ranked by it "
                "descending, so this arm is adversarial to PlumeRank's own discriminator."
            ),
            "features": scored.measurement.features,
        },
        "paired_hold_sequence_id": candidate.paired_hold_sequence_id,
        "paired_hold_t0_epoch": candidate.paired_hold_t0_epoch,
        "paired_hold_local_minutes": paired_minutes,
        "local_minutes": local_minutes,
        "local_minutes_delta": minutes_apart(local_minutes, paired_minutes),
        "frames": [frame_path(epochs[i]) for i in sorted(epochs)],
    }


def build(
    hold_manifest: dict,
    excluded: set[tuple[str, str]],
    end: date,
    quotas: dict[str, int] = CLASS_QUOTAS,
) -> dict:
    """Screen, select, enumerate and assemble. Raises rather than returning something partial."""
    paired = paired_hold_windows(hold_manifest)
    days = candidate_days(end)

    pool: dict[str, list[_Scored]] = {}
    for camera_id in sorted(paired):
        pool[camera_id] = _score_camera(camera_id, paired[camera_id], days, excluded)

    chosen = _select(pool, quotas)

    windows: list[dict] = []
    for sequence_id in sorted(chosen):
        scored = chosen[sequence_id]
        epochs = enumerate_window(
            scored.candidate.camera_id,
            scored.candidate.centre_epoch,
            list(range(FRAMES_PER_WINDOW)),
            known=scored.probes,
        )
        if len(epochs) != FRAMES_PER_WINDOW:
            raise FetchError(
                f"{sequence_id} enumerated {len(epochs)} of {FRAMES_PER_WINDOW} frames. "
                f"A short window would be a shorter camera-night than every other window "
                f"in the evaluation and would silently change the alert rate's "
                f"denominator. No manifest was written."
            )
        windows.append(_window_json(scored, epochs))

    total = sum(quotas.values())
    if len(windows) != total:
        raise FetchError(f"assembled {len(windows)} windows, expected {total}")

    return {
        "arm": "confuse",
        "n_windows": len(windows),
        "generated": date.today().isoformat(),
        "source": "HPWREN medium-term archive (MTA)",
        "credit": CREDIT,
        "licence_note": (
            "HPWREN imagery carries no SPDX licence and requires a credit reference. "
            "These windows contain no fire and are used ONLY as the false-alarm "
            "denominator that rank.solve_threshold is calibrated on."
        ),
        "base_url": MTA_BASE_URL,
        "url_note": (
            "Frame URL = window url_prefix + frame, with '+' percent-encoded as %2B. "
            "A frame name carries its own quarter directory because an 81-minute window "
            "can straddle two of them. The file name is a UTC epoch but the directory is "
            "San Diego local: the date is the local date and Q<n> is the local three-hour "
            "block numbered from 1 at local midnight, so n runs 1..8. Probed hour by hour "
            "against the live bucket; spike/urls_fog.txt alone cannot pin this, because "
            "all 60 of its URLs fall in the one band where a UTC six-hour reading gives "
            "the same answer."
        ),
        "local_root": "corpus/mta",
        "catalogue": (
            "The MTA publishes no directory index (the bucket answers AccessDenied to a "
            "bare directory, to ?list-type=2 and to index.html), so frames were "
            "enumerated by probing the 60 s grid with a +/-"
            f"{JITTER_SECONDS} s jitter search."
        ),
        "selection": [
            "Every window is on a camera that appears in manifest_hold.json and is "
            "centred within +/-45 min of the local time of day of that camera's paired "
            "HOLD ignition. A confusion set on other cameras or at another hour would be "
            "a straw man; tests/test_confusion_set.py and test_confuse_shares_cameras "
            "assert both from the frame epochs rather than from these fields.",
            "One window per camera, so the denominator spans the roster instead of "
            "concentrating on whichever camera happened to be foggiest.",
            "Within each class, candidates were ranked by raw S3 -- PlumeRank's own fog "
            "discriminator -- and the hardest were taken. The arm is adversarial to this "
            "project by construction, which makes the solved threshold conservative.",
            "Classes were filled scarcest-supply first, because the six classes differ by "
            "an order of magnitude in how many cameras can supply them.",
            "Non-fire basis: candidate days exclude every (camera, date) FIgLib records a "
            "fire on, and every window additionally had to satisfy one of the six "
            "confuser predicates. FIgLib catalogues 521 fires and not every fire, so this "
            "is a filter and not a certification.",
        ],
        "class_quotas": dict(quotas),
        "criterion_thresholds": {
            "LENS_SKY_GRADIENT_MIN": LENS_SKY_GRADIENT_MIN,
            "LENS_STATIC_TEMPORAL_STD_MAX": LENS_STATIC_TEMPORAL_STD_MAX,
            "LENS_SKY_STRUCTURE_MIN": LENS_SKY_STRUCTURE_MIN,
            "EXPOSURE_REVERSAL_MIN": EXPOSURE_REVERSAL_MIN,
            "DUST_WARMTH_MIN": DUST_WARMTH_MIN,
            "DUST_SATURATION_MIN": DUST_SATURATION_MIN,
            "HAZE_EDGE_ENERGY_MAX": HAZE_EDGE_ENERGY_MAX,
            "FOG_SKY_GROUND_RATIO_MAX": FOG_SKY_GROUND_RATIO_MAX,
            "SHADOW_LOWPASS_SIGMA_PX": SHADOW_LOWPASS_SIGMA_PX,
            "SHADOW_LOWPASS_DECIMATION": SHADOW_LOWPASS_DECIMATION,
            "SHADOW_MOTION_MIN": SHADOW_MOTION_MIN,
            "SKY_GROUND_BAND_DIVISOR": SKY_GROUND_BAND_DIVISOR,
            "DIFF_BLUR_KERNEL_PX": DIFF_BLUR_KERNEL_PX,
            "class_order": list(_CLASS_ORDER),
        },
        "search_budget": {
            "PHASE_SEARCH_SECONDS": PHASE_SEARCH_SECONDS,
            "JITTER_SECONDS": JITTER_SECONDS,
            "PROBE_SLOTS": PROBE_SLOTS,
            "PROBE_MIN_PRESENT": PROBE_MIN_PRESENT,
            "MAX_PROBE_CONCURRENCY": MAX_PROBE_CONCURRENCY,
            "CANDIDATE_DAYS": CANDIDATE_DAYS,
            "CANDIDATE_DAY_STRIDE": CANDIDATE_DAY_STRIDE,
            "TOD_OFFSETS_MINUTES": list(TOD_OFFSETS_MINUTES),
            "MAX_CANDIDATES_PER_CAMERA": MAX_CANDIDATES_PER_CAMERA,
            "end_date": end.isoformat(),
        },
        "sha256_note": (
            "Per-file sha256 is not in this manifest and that is deliberate: hashing the "
            "frames requires downloading them, which is what scripts/fetch.py does. It "
            "writes data/fetch_receipt.json on the first fetch and verifies against it on "
            "every later run. The manifest pins WHICH frames; the receipt pins WHAT they "
            "contained."
        ),
        "windows": windows,
    }


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select 45 hard non-fire windows on the HOLD cameras at the HOLD hour and "
            "write data/manifest_confuse.json."
        )
    )
    parser.add_argument(
        "--data-root", type=pathlib.Path, default=_REPO_ROOT / "data"
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date.today(),
        help="most recent day of the candidate sweep (YYYY-MM-DD)",
    )
    args = parser.parse_args(argv)

    data_root: pathlib.Path = args.data_root
    out_path = data_root / "manifest_confuse.json"

    manifests: list[dict] = []
    for name in ("manifest_hold.json", "manifest_dev.json"):
        path = data_root / name
        if not path.is_file():
            print(f"FAILED  {path} is missing; it is what pins the cameras and the hour.", file=sys.stderr)
            return 2
        manifests.append(json.loads(path.read_text()))
    hold_manifest, dev_manifest = manifests

    try:
        manifest = build(
            hold_manifest,
            _figlib_fire_dates((hold_manifest, dev_manifest)),
            args.end_date,
        )
    except FetchError as exc:
        print(f"FAILED  {exc}", file=sys.stderr)
        return 1

    # Written last, in one call, and only on the success path. A partial manifest is worse
    # than no manifest: it loads, it solves a threshold, and nothing downstream can tell.
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    histogram = {label: 0 for label in CLASS_QUOTAS}
    for window in manifest["windows"]:
        histogram[window["class"]] += 1
    print(f"wrote {out_path}: {manifest['n_windows']} windows")
    for label in _CLASS_ORDER:
        print(f"  {label:<18} {histogram[label]:3d}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    sys.exit(main())

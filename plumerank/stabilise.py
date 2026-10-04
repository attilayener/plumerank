"""Camera motion removal -- the stage that has to run before the temporal model exists.

Why this module is first, and why it is not optional
----------------------------------------------------
Every background subtractor OpenCV 5 ships -- MOG2, LSBP, GSOC -- models a *per-pixel*
history and therefore assumes the camera does not move. HPWREN's cameras are PTZ: they
pan on a schedule and they twitch in wind. Hand an unregistered pair to MOG2 and the
entire ridgeline comes back as foreground, which is precisely where S1's base-anchoring
statistic lives. So the library's design and this project's primitive are in structural
tension, and the only honest resolution is to bolt registration in *front* of the model
and then publish how well it worked (``residual_error``), rather than to average the
failure away (the OpenCV 5 design notes section 8.1).

The two paths, and the single line between them
-----------------------------------------------
**Primary** is phase correlation: a Hanning-windowed ``cv2.phaseCorrelate`` for the
coarse shift and its peak response, then ``cv2.phaseCorrelateIterative`` -- new in
OpenCV 5.x, refining the shift inside phase-correlation space rather than by
interpolating a 5x5 centroid -- for the sub-pixel number we actually use. On a real
HPWREN frame given a synthetic 3.5 px shift in seven directions the coarse peak is
0.04-0.45 px out and the refinement 0.001-0.011 px out, so the 5.x call is worth about
two orders of magnitude on exactly the input this project runs on -- and without it the
worst case sits ON the contract's 0.5 px bar instead of clearing it. The refinement is
accepted only when it agrees with the peak it refines -- see ``SUBPIXEL_AGREEMENT_PX``,
which is there because on a featureless sky it does not.

**Fallback** runs on exactly one condition, ``response < PHASE_RESPONSE_MIN``: a dawn sky
with nothing to correlate. Then ``goodFeaturesToTrack``, ``calcOpticalFlowPyrLK``, and a
MAGSAC++ fit. One branch, one threshold, no adaptive scheme -- an estimator that picks its
own path per frame is an estimator whose error distribution nobody can publish.

``cv2.goodFeaturesToTrack`` is genuine analysis but it lives in OpenCV 5's ``features``
module (the renamed ``features2d``), **not** in ``video`` where a careless reading would
file it, and this project counts it as **used but not counted** toward its defended
analysis depth. Saying so here is cheaper than being caught rounding in our own favour.

Failure is a value, not a guess
-------------------------------
An estimate larger than ``MAX_PLAUSIBLE_SHIFT_PX`` is not shake, it is a pan or a corrupt
frame; a robust fit with fewer than ``FALLBACK_MIN_INLIERS`` inliers is noise agreeing
with itself. Both return ``method="passthrough"`` with ``dx = dy = 0.0`` and the frame
untouched. A wrong warp applied quietly costs more than no warp: it manufactures
foreground along every high-contrast edge, and nothing downstream can tell that apart
from smoke.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .constants import (
    FALLBACK_MAX_CORNERS,
    FALLBACK_MIN_DISTANCE,
    FALLBACK_MIN_INLIERS,
    FALLBACK_QUALITY_LEVEL,
    MAX_PLAUSIBLE_SHIFT_PX,
    PHASE_RESPONSE_MIN,
    SUBPIXEL_AGREEMENT_PX,
    USAC_CONFIDENCE,
    USAC_MAX_ITERS,
    USAC_REPROJ_THRESHOLD,
)
from .types import Stabilised

__all__ = ["stabilise", "warp", "residual_error"]

# ---------------------------------------------------------------------------------
# The Hanning window, cached
# ---------------------------------------------------------------------------------

#: Keyed by (height, width). Every camera in the wall runs at WORK_HEIGHT x WORK_WIDTH,
#: so this dict holds exactly one entry in production and the cache is free.
_HANNING_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _hanning_window(shape: tuple[int, int]) -> np.ndarray:
    """The 2-D Hann window for ``shape``, built once per shape and then handed out.

    OpenCV's own ``phaseCorrelate`` documentation says the window "may be cached until
    the array size changes", and the wall runs 40 cameras at one resolution, so the cache
    holds a single entry and never grows. The saving is small next to the two DFTs (0.05
    ms against 4.1 ms for the whole stage, measured at 480x270 on this machine) -- it is
    kept because it is free and because the per-frame classical budget is 30 ms total,
    not because it is where the time goes.

    The array is returned, not copied -- callers must treat it as read-only.
    """
    h, w = shape
    win = _HANNING_CACHE.get((h, w))
    if win is None:
        # createHanningWindow takes (width, height); getting this backwards on a
        # non-square frame fails loudly at the phaseCorrelate call, not silently here.
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        _HANNING_CACHE[(h, w)] = win
    return win


# ---------------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------------


def _check_pair(prev_grey: np.ndarray, cur_grey: np.ndarray) -> None:
    """Raise on any pair this module cannot register. Silence means the pair is usable."""
    for name, arr in (("prev_grey", prev_grey), ("cur_grey", cur_grey)):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be single-channel greyscale, got shape {arr.shape}"
            )
    if prev_grey.shape != cur_grey.shape:
        raise ValueError(
            f"frame pair must match: {prev_grey.shape} != {cur_grey.shape}"
        )
    if min(prev_grey.shape) < 2:
        raise ValueError(f"frame too small to correlate: {prev_grey.shape}")


def _windowed(grey: np.ndarray, window: np.ndarray) -> np.ndarray:
    """``grey`` as CV_32F with the Hann coefficients already in the pixels.

    Windowing here instead of passing ``window`` to ``cv2.phaseCorrelate`` is a
    correctness requirement, not a preference. That call applies the window by
    multiplying its *inputs in place* whenever the frame needs no DFT padding -- and
    480x270, the working resolution, needs none -- so handing it a caller's CV_32F frame
    replaces that frame, in the caller, with a vignetted copy of itself, and the frame
    this module then warps and returns is the vignette. It also leaves
    ``phaseCorrelateIterative``, which takes no window argument, with nothing consistent
    to be given: windowing its inputs separately applies the coefficients a second time,
    so the coarse peak and the refinement would be measuring two different images and the
    agreement check between them would be comparing the wrong pair.

    One multiply, one fresh array, both estimators. ``np.multiply`` with an explicit
    output dtype does the 8-bit-to-float conversion in the same pass.
    """
    return np.multiply(grey, window, dtype=np.float32)


def _as_u8(grey: np.ndarray) -> np.ndarray:
    """PyrLK and goodFeaturesToTrack want 8-bit; the phase path wants float.

    Converting here rather than asking the caller to pass both keeps the seam in
    ``types.Stabilised`` a single array.
    """
    if grey.dtype == np.uint8:
        return grey
    return np.clip(grey, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------------
# The fallback: features -> PyrLK -> robust fit
# ---------------------------------------------------------------------------------


def _affine_fallback(
    prev_grey: np.ndarray, cur_grey: np.ndarray
) -> tuple[float, float] | None:
    """Scene shift from a robust fit over PyrLK correspondences, or ``None`` to reject.

    ``None`` means "this pair cannot be registered" and is the caller's cue to pass the
    frame through. It is returned rather than a best-effort number at every one of the
    three places the chain can thin out -- too few corners, too few surviving tracks, too
    few inliers -- because on a featureless sky all three failures look identical from
    the outside and only the last one is detectable after the fact.

    ``cv2.goodFeaturesToTrack`` is an OpenCV 5 ``features`` call (the renamed
    ``features2d``). It is real analysis, it is **used but not counted** as this
    project's analysis depth, and it is listed that way in the defence rather than filed
    under ``video`` where it would inflate the count.
    """
    prev_u8 = _as_u8(prev_grey)
    cur_u8 = _as_u8(cur_grey)

    corners = cv2.goodFeaturesToTrack(
        prev_u8,
        maxCorners=FALLBACK_MAX_CORNERS,
        qualityLevel=FALLBACK_QUALITY_LEVEL,
        minDistance=FALLBACK_MIN_DISTANCE,
    )
    if corners is None or len(corners) < FALLBACK_MIN_INLIERS:
        return None

    nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev_u8, cur_u8, corners, None)  # type: ignore[call-overload]
    if nxt is None or status is None:
        return None

    kept = status.ravel() == 1
    src = corners[kept].reshape(-1, 1, 2)
    dst = nxt[kept].reshape(-1, 1, 2)
    if len(src) < FALLBACK_MIN_INLIERS:
        return None

    # estimateAffine2D, not estimateAffinePartial2D: in 5.0.0 the 4-DoF variant rejects
    # every USAC method at runtime ("Unknown or unsupported robust estimation method",
    # geometry/src/ptsetreg.cpp:1155) and only the 6-DoF one accepts USAC_MAGSAC. Keeping
    # MAGSAC++ and paying two extra degrees of freedom is the better trade here, because
    # MAGSAC++ marginalises over the inlier threshold instead of fixing it -- and on a
    # low-texture sky the correspondence quality is exactly the thing that varies.
    model, inliers = cv2.estimateAffine2D(
        src,
        dst,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=USAC_REPROJ_THRESHOLD,
        maxIters=USAC_MAX_ITERS,
        confidence=USAC_CONFIDENCE,
    )
    if model is None or inliers is None:
        return None
    if int(np.count_nonzero(inliers)) < FALLBACK_MIN_INLIERS:
        return None

    # The shift is read off the fitted transform AT THE IMAGE CENTRE, not off its
    # translation column. The rotation/scale the 6-DoF fit also recovers is discarded on
    # purpose -- warping by an estimated zoom resamples the plume itself, and S2 measures
    # a growth rate a wrong scale corrupts directly -- but column 2 is the displacement at
    # the pixel ORIGIN, so any recovered roll or zoom leaks into it multiplied by the
    # lever arm to the corner: one degree of roll about the centre of a 480x270 frame is
    # zero shift in the middle of the frame and (2.4, -4.2) px at column 2. The middle is
    # where the plume is, and a 4 px phantom pan is well inside MAX_PLAUSIBLE_SHIFT_PX,
    # so it would have been applied rather than rejected.
    h, w = prev_grey.shape
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    return (
        float(model[0, 0] * cx + model[0, 1] * cy + model[0, 2] - cx),
        float(model[1, 0] * cx + model[1, 1] * cy + model[1, 2] - cy),
    )


# ---------------------------------------------------------------------------------
# The public surface
# ---------------------------------------------------------------------------------


def _passthrough(cur_grey: np.ndarray, response: float) -> Stabilised:
    """The documented refusal: frame untouched, shift exactly zero, reason recorded.

    Zero is chosen over ``nan`` so that a downstream caller which forgets to check
    ``method`` still warps by nothing rather than crashing three stages later on an
    arithmetic it cannot attribute to this one.
    """
    return Stabilised(
        frame=cur_grey.copy(),
        dx=0.0,
        dy=0.0,
        response=response,
        method="passthrough",
    )


def warp(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Register ``frame`` back onto its predecessor, given an estimated scene shift.

    ``dx, dy`` are the shift the scene *underwent*, exactly as ``Stabilised`` carries it,
    so the translation actually applied is its negation. The argument order is that way
    round so a caller with a ``Stabilised`` can register any other view of the same
    camera -- the colour frame, an evidence crop -- with ``warp(other, s.dx, s.dy)`` and
    no sign to get wrong at the call site.

    Border pixels are replicated rather than filled with zeros. A black band along one
    edge is a step change of 128 grey levels that MOG2 reports as foreground for the full
    length of the frame, which is a fabricated detection in the literal sense.
    """
    if not (math.isfinite(dx) and math.isfinite(dy)):
        raise ValueError(f"non-finite warp requested: dx={dx}, dy={dy}")
    if dx == 0.0 and dy == 0.0:
        # The identity warp still costs a full resample, and the passthrough path takes
        # it on every rejected frame.
        return frame.copy()

    h, w = frame.shape[:2]
    matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def stabilise(prev_grey: np.ndarray, cur_grey: np.ndarray) -> Stabilised:
    """Register ``cur_grey`` onto ``prev_grey`` and report how it was done.

    Args:
        prev_grey: the previous frame, single-channel, working resolution.
        cur_grey: this frame, same shape and dtype.

    Returns:
        A ``Stabilised`` whose ``frame`` is ``cur_grey`` registered onto ``prev_grey``,
        whose ``dx, dy`` are the estimated scene shift in working pixels (positive ``dx``
        means the scene moved right), and whose ``method`` is one of:

        * ``"phase+iterative"`` -- the normal path.
        * ``"phase"`` -- the coarse peak was usable but the sub-pixel refinement was not
          (non-finite, or further than ``SUBPIXEL_AGREEMENT_PX`` from the peak it
          claims to refine), so the unrefined shift is used and labelled as such.
        * ``"affine"`` -- the phase peak was untrustworthy and the feature fallback ran.
        * ``"passthrough"`` -- no estimate survived the plausibility check; the frame is
          returned unwarped and ``dx = dy = 0.0``.

    Raises:
        TypeError, ValueError: on a pair that is not two matching 2-D arrays. A caller
        who hands this stage a colour frame has a bug, and discovering it four stages
        later as a strange hazard score is worse than discovering it here.

    ``response`` is always the measured phase-peak response, including on the fallback
    and passthrough paths -- it is the covariate ``DEMO.md`` plots time-to-detection
    against, so the frames where the estimator refused are exactly the ones it must not
    be missing for.
    """
    _check_pair(prev_grey, cur_grey)

    window = _hanning_window(prev_grey.shape)
    prev_w = _windowed(prev_grey, window)
    cur_w = _windowed(cur_grey, window)

    # No window argument: the coefficients are already in prev_w and cur_w, and passing
    # them again would window twice (see _windowed for why they are applied there).
    (dx, dy), response = cv2.phaseCorrelate(prev_w, cur_w)
    response = float(response)

    if response < PHASE_RESPONSE_MIN:
        estimate = _affine_fallback(prev_grey, cur_grey)
        if estimate is None:
            return _passthrough(cur_grey, response)
        dx, dy = estimate
        method = "affine"
    else:
        # The same two arrays the coarse peak was measured on, which is the only reason
        # the agreement check below is a comparison and not a coincidence.
        rdx, rdy = cv2.phaseCorrelateIterative(prev_w, cur_w)
        refined_ok = (
            math.isfinite(rdx)
            and math.isfinite(rdy)
            and math.hypot(rdx - dx, rdy - dy) <= SUBPIXEL_AGREEMENT_PX
        )
        if refined_ok:
            dx, dy = float(rdx), float(rdy)
            method = "phase+iterative"
        else:
            method = "phase"

    if not (math.isfinite(dx) and math.isfinite(dy)):
        return _passthrough(cur_grey, response)
    if math.hypot(dx, dy) > MAX_PLAUSIBLE_SHIFT_PX:
        return _passthrough(cur_grey, response)

    return Stabilised(
        frame=warp(cur_grey, dx, dy),
        dx=float(dx),
        dy=float(dy),
        response=response,
        method=method,
    )


def residual_error(
    prev_grey: np.ndarray, cur_grey: np.ndarray, result: Stabilised
) -> float:
    """Misalignment still present after ``result`` was applied, in working pixels.

    Measured the only way that is comparable across the two estimator paths: correlate
    the registered frame against ``prev_grey`` again and report the magnitude of whatever
    shift is left. A photometric score -- mean absolute difference, say -- would move with
    the smoke as well as with the camera, and this number exists to be a *covariate* of
    time-to-detection, so it has to be free of the signal it is being plotted against.

    Returns:
        The residual shift magnitude in pixels, or ``float("inf")`` when the second
        correlation is itself untrustworthy (``response < PHASE_RESPONSE_MIN``, the same
        question asked of the same quantity). ``inf`` propagates through
        ``metrics.percentiles`` without becoming a plausible small number, which is the
        whole reason that function refuses to interpolate.

    On a ``passthrough`` result this returns the full unregistered motion, because that
    is what the residual honestly is when nothing was corrected.

    **What this number is not.** The instrument is ``phaseCorrelate``, so on the
    ``"phase"`` path it is grading an estimate it produced itself and reads near zero by
    construction (median 0.007 px over the real consecutive pairs in ``spike/frames``);
    on ``"phase+iterative"`` it is measuring the refinement's disagreement with that same
    coarse peak (median 0.068 px, and the two medians are the same quantity seen from
    either side). It is an independent check only on the ``"affine"`` and
    ``"passthrough"`` paths. So the published distribution is a per-method
    self-consistency figure, and quoting the phase-path median as stabilisation
    *accuracy* would be a claim this function cannot support.
    """
    _check_pair(prev_grey, cur_grey)

    window = _hanning_window(prev_grey.shape)
    registered = warp(cur_grey, result.dx, result.dy)
    (rdx, rdy), response = cv2.phaseCorrelate(
        _windowed(prev_grey, window), _windowed(registered, window)
    )
    if float(response) < PHASE_RESPONSE_MIN:
        return math.inf
    if not (math.isfinite(rdx) and math.isfinite(rdy)):
        return math.inf
    return float(math.hypot(rdx, rdy))

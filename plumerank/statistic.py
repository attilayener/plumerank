"""THE PRIMITIVE -- the anchored plume-growth statistic OpenCV does not ship.

Every other module in this package is assembly. This one is the reason the project
exists: four cheap, interpretable, per-component numbers that say *this foreground blob
behaves like a fire plume and that one behaves like weather*, computed from geometry and
two derivative filters rather than from a model that has to be trained, hosted and
defended.

The four claims, each of which is a sentence about physics before it is a line of code
-----------------------------------------------------------------------------------
* **S1, base anchoring.** A fire is tied to the ground. Its lowest extent is pinned at the
  ignition point while its top climbs. A cloud, a fog bank and a lens flare all translate
  as rigid bodies -- foot and top move together. S1 is the ratio between how far the top
  climbed and how far the foot failed to stay put.
* **S2, vertical growth anisotropy.** A plume rises faster than it spreads. Convection is
  directional; diffusion is not. S2 is d(height)/dt over d(width)/dt.
* **S3, diffuse-boundary contrast.** Smoke's edge is soft and its interior is turbulent;
  a cloud shadow's edge is hard and its interior is flat. S3 is interior Laplacian energy
  over boundary-band Sobel energy. **This is the statistic the project's GO/NO-GO gate was
  decided on** -- see the hard requirement below.
* **S4, DIS drift.** A plume rises *coherently*; sensor noise does not. S4 is the mean
  upward flow inside the component's box against the incoherent residual around it.

Why a single squash helper
--------------------------
Each statistic first forms a **dimensionless ratio** in the units its own physics fixes,
and then -- and only then -- passes through ``squash``. One map, ``x / (1 + x)``, used
four times. The alternative, a bespoke normalisation per statistic, is four unfalsifiable
choices instead of one: nobody could later say which of them the fusion weights in
``constants.py`` were compensating for. With one map, the fusion weights are the only
place a per-statistic scale can hide, and they are fitted on DEV in the open.

Why S3's raw value survives into ``Stats``
------------------------------------------
``s3_diffuse_boundary`` returns ``(squashed, raw)`` and ``Stats.raw_s3`` carries the raw
one. The Sep 1 gate published an AUC in raw units; keeping those units in production is
what makes the published number evidence about the running system rather than about a
script that no longer exists.

**Hard requirement, and the one thing in this file that must never be "improved":** the
raw S3 below is numerically identical to ``spike/s3_separation.py``'s ``s3(roi, "v0")``,
down to the ``+ EPS`` in the denominator and the geometric ring construction.
``tests/test_statistic.py::test_s3_matches_the_gate`` asserts that agreement against the
spike script itself, imported by file path, on random ROIs. ``S3_BAND_FRACTION`` is fixed
a priori and is never refitted: refitting it after the gate would retro-actively
invalidate the gate.

Failure is flagged, never smoothed
----------------------------------
A component too thin to hold a boundary band has **no** S3, and that is reported as
``nan`` in both the squashed and the raw slot -- exactly what the spike returns. It is
*not* reported as 0.0, because 0.0 already means something specific and defensible ("hard
edge, flat interior; this is a shadow"), and a caller cannot tell a confident 0.0 from a
missing measurement. Downstream code must treat a non-finite statistic as an unrankable
component rather than fuse it.

OpenCV surfaces used here (the OpenCV 5 design notes sections 1 and 2):
``cv2.Laplacian`` and ``cv2.Sobel`` (S3, imgproc) and ``cv2.DISOpticalFlow_create`` with
``cv2.DISOPTICAL_FLOW_PRESET_FAST`` (S4, video).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .constants import (
    EPS,
    MAX_PLAUSIBLE_SHIFT_PX,
    S1_ANCHOR_TOLERANCE_PX,
    S2_ISOTROPIC_RATIO,
    S2_WIDTH_RATE_FLOOR_PX_PER_FRAME,
    S3_BAND_FRACTION,
    S3_BAND_MIN_PX,
    S4_ENABLED,
    S4_MIN_BBOX_PX,
    STAT_WINDOW_FRAMES,
)
from .types import Component, Stats, Track

__all__ = [
    "Tracker",
    "squash",
    "measure",
    "s1_anchoring",
    "s2_anisotropy",
    "s3_diffuse_boundary",
    "s4_drift",
]


# ---------------------------------------------------------------------------------
# The one squash
# ---------------------------------------------------------------------------------


def squash(ratio: float) -> float:
    """The single monotone ``[0, inf) -> [0, 1)`` map every statistic passes through.

    ``x / (1 + x)``. Chosen over a logistic because it needs no centre and no slope --
    two more numbers that would have had to be fitted, frozen and defended -- and over a
    hard ``min(x, 1)`` clip because a clip destroys ordering above the clip point, and
    the ranking is *only* an ordering. Saturating rather than clipping means a component
    with a ratio of 40 still outranks one with a ratio of 10.

    A negative ratio means "the evidence points the other way", which is the same
    decision as "no evidence", so it floors at 0.0 rather than producing a negative
    score that the fusion weights would then have to interpret. ``nan`` propagates: an
    unmeasurable statistic must stay unmeasurable.
    """
    if math.isnan(ratio):
        return math.nan
    if ratio == math.inf:
        return 1.0
    r = max(0.0, float(ratio))
    return r / (1.0 + r)


def _slope(frames: list[int], values: list[float]) -> float:
    """Least-squares d(value)/d(frame), closed form.

    ``np.polyfit`` for a degree-1 fit on at most ``STAT_WINDOW_FRAMES`` points builds a
    Vandermonde matrix and calls ``lstsq`` -- a lot of machinery per component per frame,
    at 40 cameras a minute, and it emits a RankWarning on the degenerate input this code
    hits routinely (a track with every sample on one frame). The closed form is three
    reductions and returns a documented 0.0 on that degeneracy instead of warning.
    """
    t = np.asarray(frames, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    t_centred = t - t.mean()
    denom = float((t_centred * t_centred).sum())
    if denom <= 0.0:
        return 0.0
    return float((t_centred * (v - v.mean())).sum() / denom)


# ---------------------------------------------------------------------------------
# S1 -- base anchoring
# ---------------------------------------------------------------------------------


def s1_anchoring(track: Track) -> float:
    """How much of the track's vertical growth happened *without* the foot moving.

    Returns 1.0 for a perfectly pinned foot under a climbing top, 0.0 for a foot that
    drifts as much as the top climbs (a translating cloud), and 0.0 for a top that is not
    climbing at all.

    Three deliberate choices in how the two numbers are measured:

    **The foot is measured peak-to-peak, the top by fitted slope.** They are not
    symmetric because the claims about them are not symmetric. S1's claim about the foot
    is *it never went anywhere*, and a foot that swings 12 px down and 12 px back has a
    fitted slope of zero while plainly having gone somewhere; only the excursion catches
    that. S1's claim about the top is *it trended upward over ten minutes*, which is a
    trend, and a single bad segmentation frame must not be allowed to set it.

    **``S1_ANCHOR_TOLERANCE_PX`` is a dead band, not a threshold.** Drift inside it costs
    nothing at all rather than costing a little; that is what "stays within" means, and it
    keeps the statistic from being a segmentation-jitter meter at 480x270, where the
    bottom row of a porous plume moves a pixel or two per frame for free.

    **The dead band is subtracted from the climb as well as from the drift, and that is
    what makes S1 speed-invariant.** Charging the tolerance to the foot alone but not to
    the top leaves a leak that grows as the motion gets slower: a rigid body -- a fog bank,
    a cloud, a lens flare -- translating upward at 3 px/frame then scores 0.15, but the
    same body at 0.5 px/frame scores **0.89**, because the dead band has eaten its entire
    drift while leaving its entire climb standing. Slow rigid drift is exactly the fog
    behaviour S1 exists to reject, and 0.5 px/frame at 480x270 is a real fog bank, so the
    leak sat on the wrong side of the statistic. Subtracting the same band from both makes
    a rigid translation score 0.0 at *every* speed, which is what "0.0 for a foot that
    drifts as much as the top" was always supposed to mean. The cost is that a climb
    smaller than the tolerance now scores 0.0 rather than 1.0 -- correctly: a 3 px rise
    over ten minutes is segmentation jitter, and calling it perfectly anchored growth is
    the more dangerous of the two errors.

    A track with fewer than two samples returns 0.0: there is genuinely no growth to
    measure yet, and 0.0 -- "no evidence of anchored growth" -- is the honest reading of
    that, not a guess.
    """
    if len(track) < 2:
        return 0.0

    span = float(track.frames[-1] - track.frames[0])
    if span <= 0.0:
        return 0.0

    tops = [float(c.top) for c in track.history]
    bottoms = [float(c.bottom) for c in track.history]

    # y grows downward, so a climbing top has a negative slope.
    climb_px = -_slope(track.frames, tops) * span
    if climb_px <= 0.0:
        return 0.0

    drift_px = max(bottoms) - min(bottoms)
    excess_drift_px = max(0.0, drift_px - S1_ANCHOR_TOLERANCE_PX)
    excess_climb_px = max(0.0, climb_px - S1_ANCHOR_TOLERANCE_PX)

    # Both excursions are measured in excess of the same dead band, so a rigid translation
    # cancels exactly and scores 0.0 at any speed. Dividing by the excess drift rather
    # than by the climb is what makes a pinned foot score 1.0 exactly -- with EPS in the
    # denominator the ratio diverges, and squash saturates.
    return squash(max(0.0, excess_climb_px - excess_drift_px) / (excess_drift_px + EPS))


# ---------------------------------------------------------------------------------
# S2 -- vertical growth anisotropy
# ---------------------------------------------------------------------------------


def s2_anisotropy(track: Track) -> float:
    """d(height)/dt over d(width)/dt, squashed. A plume rises faster than it spreads.

    The rate ratio is formed exactly as ``build-contract.md`` specifies -- least-squares
    slopes over the window, denominator floored at
    ``S2_WIDTH_RATE_FLOOR_PX_PER_FRAME`` so a component that does not widen at all yields
    a large ratio rather than an infinity.

    **The squash is fed the ratio's excess over ``S2_ISOTROPIC_RATIO``, not the ratio.**
    That constant is 1.0 and is not a knob: it is the *definition* of isotropy -- growing
    as fast sideways as upward -- and a statistic named "anisotropy" that scored 0.5 on a
    perfectly isotropic disc would be misnamed. Subtracting it puts the squash's zero on
    the physical null hypothesis instead of on an arbitrary point, so ``s2 == 0`` reads
    "this thing expands like a diffusing cloud" and nothing has to interpret 0.5. It lives
    in ``constants.py`` and not in this line because it decides an output, and the freeze
    that ``results/constants.sha256`` records only covers that file.

    **Consequence, stated rather than left to be discovered:** because the denominator is
    floored, subtracting 1.0 turns that floor into an absolute gate -- a component whose
    height rate is below ``S2_WIDTH_RATE_FLOOR_PX_PER_FRAME`` scores exactly 0.0 no matter
    how little it widens. That is intended. Two pixels of vertical growth across a ten
    minute window at 480x270 is segmentation jitter, and "no evidence of anisotropic
    growth" is the honest reading of it; ``test_s2_has_an_absolute_growth_floor`` pins it.

    Shrinking counts as not growing, not as negative growth: a component whose height is
    falling is not a plume that is un-rising, it is a segmentation that is losing the
    object, and the fusion weights should not be handed a signed value to disentangle.
    """
    if len(track) < 2:
        return 0.0

    heights = [float(c.h) for c in track.history]
    widths = [float(c.w) for c in track.history]

    d_height = _slope(track.frames, heights)
    d_width = _slope(track.frames, widths)

    rate_ratio = max(0.0, d_height) / max(d_width, S2_WIDTH_RATE_FLOOR_PX_PER_FRAME)
    return squash(rate_ratio - S2_ISOTROPIC_RATIO)


# ---------------------------------------------------------------------------------
# S3 -- diffuse-boundary contrast. THE ONE THE GATE WAS DECIDED ON.
# ---------------------------------------------------------------------------------


def _s3_raw(roi: np.ndarray) -> float:
    """Line-for-line the gate's ``s3(roi, "v0")``. Do not refactor this into elegance.

    Every operation, its order and its dtype are load-bearing, because the only thing
    that makes the Sep 1 AUC evidence about *this* code is that this code returns the
    same float. ``tests/test_statistic.py::test_s3_matches_the_gate`` imports the spike
    script by path and diffs the two on random ROIs, so a well-meant simplification here
    fails the suite rather than quietly detaching production from its own gate.
    """
    roi = roi.astype(np.float32)
    h, w = roi.shape
    b = max(S3_BAND_MIN_PX, int(S3_BAND_FRACTION * min(w, h)))
    if h - 2 * b < 2 or w - 2 * b < 2:
        return math.nan  # box too small to hold a boundary band

    # Interior and ring are built GEOMETRICALLY, never by eroding a mask: erosion's
    # default borderValue is +inf, so eroding a solid mask leaves it solid, the ring comes
    # back empty, and every score is NaN. That bug is the reason the spike builds it this
    # way and the reason this comment outlives the code it explains. The symbol is spelled
    # without its module prefix on purpose: check_claims.py greps source for OpenCV
    # symbols, and a surface named only to say it is NOT used must not inflate the count.
    interior = np.zeros((h, w), bool)
    interior[b : h - b, b : w - b] = True
    ring = ~interior

    lap = np.abs(cv2.Laplacian(roi, cv2.CV_32F))
    gx = np.abs(cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3))

    num = float(lap[interior].mean())
    den = float((gx + gy)[ring].mean()) + EPS
    return num / den


def s3_diffuse_boundary(grey: np.ndarray, component: Component) -> tuple[float, float]:
    """``(squashed, raw)`` interior-blur-over-boundary-edge energy for one component.

    Smoke is soft-edged and turbulent inside: high interior Laplacian energy against low
    Sobel energy in the band around it. A cloud shadow, a fog bank's leading edge or a
    lens artefact inverts that ratio. This is the only one of the four statistics that
    separates smoke from fog, which is why it carries the largest fusion weight.

    Returns ``(nan, nan)`` when the component is too thin to hold a boundary band --
    the band is ``max(S3_BAND_MIN_PX, int(S3_BAND_FRACTION * min(w, h)))`` px on every
    side, and ``MIN_COMPONENT_AREA_PX`` admits shapes that cannot spare it. ``nan``
    rather than 0.0 because 0.0 is a real, confident answer here and must not be
    confused with a missing one.

    The raw value is returned alongside so ``Stats.raw_s3`` can carry the gate's own
    units into production. Do not squash twice.
    """
    x, y, w, h = component.bbox()
    if w <= 0 or h <= 0:
        return math.nan, math.nan

    roi = grey[y : y + h, x : x + w]
    if roi.shape[0] != h or roi.shape[1] != w:
        # The bbox ran off the frame. Scoring the truncated crop would silently change
        # the band geometry and hand back a number in different units from every other
        # component -- worse than admitting the measurement is missing.
        return math.nan, math.nan

    raw = _s3_raw(roi)
    if math.isnan(raw):
        return math.nan, math.nan
    return squash(raw), raw


# ---------------------------------------------------------------------------------
# S4 -- DIS drift, inside the bounding box only
# ---------------------------------------------------------------------------------

#: ONE DIS INSTANCE PER ROI SHAPE, not one per process, and the reason is a measured
#: OpenCV 5.0.0 behaviour rather than caution.
#:
#: ``cv2.DISOpticalFlow.calc`` returns an **all-NaN flow field, with no exception**, for
#: short wide inputs: the boundary swept on 5.0.0.93 is sharp and reproducible at
#: ``height < 32 and width >= 100``. It is asymmetric -- 30x195 fails, 195x30 does not --
#: so it is a property of the row count against the coarsest pyramid level, not of the
#: aspect ratio. A wide short component is not exotic here: it is an early plume spreading
#: along a ridgeline, and it is exactly the shape the first real end-to-end run hit.
#:
#: A single reused instance made this *worse rather than better*. After a call on a larger
#: ROI the instance's buffers are already allocated, so the next 30x195 call returns
#: FINITE NUMBERS computed against stale buffers -- a plausible wrong answer, arriving
#: silently, and dependent on what the previous component happened to be. That is
#: order-dependence in a pipeline whose entire value rests on ``make verify`` printing the
#: same number twice. Keying the cache by shape removes it: every ROI size gets an
#: instance that has only ever seen that size.
#:
#: The cache is bounded because the pipeline emits at most MAX_COMPONENTS_PER_FRAME
#: components per frame at a working resolution of 480x270; if it ever grows unbounded,
#: that is a symptom worth seeing rather than a leak worth hiding.
#:
#: Filed upstream (the build plan D52) with this minimal repro:
#:     a = np.random.randint(0, 255, (30, 195), np.uint8); b = np.roll(a, -1, 0)
#:     cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(a, b, None)
#:     -> field of nan, no exception raised
_DIS_BY_SHAPE: dict[tuple[int, int], cv2.DISOpticalFlow] = {}


def _dis(shape: tuple[int, int]) -> cv2.DISOpticalFlow:
    dis = _DIS_BY_SHAPE.get(shape)
    if dis is None:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)  # type: ignore[attr-defined]
        _DIS_BY_SHAPE[shape] = dis
    return dis


def s4_drift(
    grey: np.ndarray,
    prev_grey: np.ndarray | None,
    component: Component,
) -> float:
    """Coherent upward drift inside the component's box, against the incoherent residual.

    Dense flow is computed **inside the bounding box only**, not over the frame. That is
    both faster -- a 40x60 box instead of 480x270, on every component of every camera --
    and more correct: DIS propagates its coarse-to-fine initialisation across the whole
    input, so a frame-wide field lets the sky's aperture-problem garbage leak into the
    one region we are actually asking about.

    The ratio is *mean upward flow* over *mean residual magnitude after the mean vector
    is removed*. A signal-to-noise form rather than raw speed, because raw speed cannot
    tell a rising plume from a fast-panning camera, and because it is dimensionless: px
    per frame cancels, so the same squash applies here as everywhere else.

    Returns 0.0 -- meaning "no drift evidence", which is the truth -- when S4 is disabled
    by ablation, when there is no previous frame, when the box is under ``S4_MIN_BBOX_PX``
    on either side (where DIS's patch grid has nothing to fit), and when the box runs off
    the frame so the two crops are not the same rectangle. The last case is 0.0 rather
    than ``nan`` because S4 is an additive piece of evidence under the ablation gate, not
    a measurement the ranking depends on the way it depends on S3.
    """
    if not S4_ENABLED or prev_grey is None:
        return 0.0

    x, y, w, h = component.bbox()
    if min(w, h) < S4_MIN_BBOX_PX:
        return 0.0

    cur_roi = grey[y : y + h, x : x + w]
    prev_roi = prev_grey[y : y + h, x : x + w]
    if cur_roi.shape != prev_roi.shape or cur_roi.shape[0] != h or cur_roi.shape[1] != w:
        return 0.0

    # DIS wants contiguous 8-bit single-channel input; a bbox crop is a strided view.
    prev_roi = np.ascontiguousarray(prev_roi, dtype=np.uint8)
    cur_roi = np.ascontiguousarray(cur_roi, dtype=np.uint8)

    flow = _dis((h, w)).calc(prev_roi, cur_roi, None)  # type: ignore[call-overload]

    # DIS reports an unmeasurable geometry by returning nan rather than by raising (see
    # _DIS_BY_SHAPE). Checking the field instead of the shape is deliberate: the exact
    # boundary is a library detail that can move between OpenCV versions, and a hardcoded
    # "height < 32" guard would go quietly wrong the day it does. 0.0 here means the same
    # thing as the four exits above -- no drift evidence -- and NOT nan, because S4 is
    # additive evidence under the D10 ablation gate rather than something the ranking
    # depends on the way it depends on S3.
    if not np.isfinite(flow).all():
        return 0.0

    mean_vec = flow.reshape(-1, 2).mean(axis=0)
    # calc(I0, I1) returns the displacement from I0 to I1 and y grows downward, so
    # content that climbed between frames carries a negative vy.
    upward = max(0.0, -float(mean_vec[1]))

    residual = flow - mean_vec
    incoherence = float(np.hypot(residual[..., 0], residual[..., 1]).mean())

    return squash(upward / (incoherence + EPS))


# ---------------------------------------------------------------------------------
# Tracking -- because a growth statistic cannot be computed from one frame
# ---------------------------------------------------------------------------------


class Tracker:
    """Follows components across frames so that S1 and S2 have a window to fit over.

    Association is greedy nearest-centroid in ascending distance, gated by
    ``MAX_PLAUSIBLE_SHIFT_PX``. Two things about that:

    **Greedy, not Hungarian.** ``MAX_COMPONENTS_PER_FRAME`` is 8. An optimal assignment
    over at most 8x8 differs from ascending-greedy only when two components pass within a
    few pixels of each other, and in that case both tracks are contaminated whichever way
    the assignment falls -- so the optimal solver would buy a dependency-free
    reimplementation of the auction algorithm and no behaviour.

    **The gate is the stabiliser's own limit, reused rather than re-fitted.** Frames
    reach this module already registered, so a component that has moved further between
    consecutive frames than the largest camera shift ``stabilise.py`` is willing to
    believe is not the same component. Fitting a second distance gate here would create
    two numbers that mean the same thing and are free to drift apart.

    Tracks are dropped after ``window`` frames without a match -- the same window the
    statistics fit over, because a track with no sample inside its own fitting window has
    nothing left to contribute.
    """

    def __init__(self, window: int = STAT_WINDOW_FRAMES) -> None:
        if window < 2:
            raise ValueError(f"window must be at least 2 frames, got {window}")
        self.window = int(window)
        self._tracks: dict[int, Track] = {}
        self._last_seen: dict[int, int] = {}
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        """Every live track, oldest first. Read-only view for the caller's diagnostics."""
        return [self._tracks[tid] for tid in sorted(self._tracks)]

    def update(self, frame_index: int, components: list[Component]) -> list[Track]:
        """Associate this frame's components with existing tracks.

        Returns one ``Track`` **per input component, in the input order**, so the caller
        can ``zip(components, tracker.update(i, components))`` without a lookup table.
        Tracks that got no component this frame are not returned -- they are still alive
        and still hold their history, they simply have nothing to say about this minute.
        """
        for tid in [t for t, seen in self._last_seen.items() if frame_index - seen > self.window]:
            del self._tracks[tid]
            del self._last_seen[tid]

        candidates: list[tuple[float, int, int]] = []
        for ci, comp in enumerate(components):
            for tid, track in self._tracks.items():
                last = track.last
                distance = math.hypot(comp.cx - last.cx, comp.cy - last.cy)
                if distance <= MAX_PLAUSIBLE_SHIFT_PX:
                    candidates.append((distance, ci, tid))
        # Ties broken by component index then track id so a replay is bit-reproducible.
        candidates.sort()

        assigned: dict[int, int] = {}
        taken_tracks: set[int] = set()
        for distance, ci, tid in candidates:
            if ci in assigned or tid in taken_tracks:
                continue
            assigned[ci] = tid
            taken_tracks.add(tid)

        out: list[Track] = []
        for ci, comp in enumerate(components):
            tid = assigned.get(ci)  # type: ignore[assignment]  # narrowed to int by the None-check below
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = Track(track_id=tid)
            track = self._tracks[tid]
            track.add(frame_index, comp, self.window)
            self._last_seen[tid] = frame_index
            out.append(track)
        return out


# ---------------------------------------------------------------------------------
# The one call the pipeline makes
# ---------------------------------------------------------------------------------


def measure(
    track: Track,
    grey: np.ndarray,
    prev_grey: np.ndarray | None = None,
) -> Stats:
    """All four statistics for a track's most recent component.

    ``grey`` is the stabilised working-resolution frame the track's last component was
    found in; ``prev_grey`` is the one before it, and S4 is skipped without complaint
    when it is absent -- the first frame of a replay has no predecessor and that is not
    an error.

    ``Stats.s3`` and ``Stats.raw_s3`` are ``nan`` together when the component is too thin
    to hold a boundary band. That pair is a **flag, not a score**: a caller that fuses it
    will get a ``nan`` hazard, which is the intended loud failure. It is not clamped to
    0.0 here, because a component that cannot be measured and a component measured as
    hard-edged are different facts and the ranking must not conflate them.
    """
    if len(track) == 0:
        raise ValueError(f"track {track.track_id} has no components to measure")

    component = track.last
    s3_squashed, s3_raw = s3_diffuse_boundary(grey, component)
    return Stats(
        s1=s1_anchoring(track),
        s2=s2_anisotropy(track),
        s3=s3_squashed,
        s4=s4_drift(grey, prev_grey, component),
        raw_s3=s3_raw,
    )

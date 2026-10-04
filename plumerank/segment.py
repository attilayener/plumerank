"""Foreground components: the one stage that decides what the statistic even looks at.

The plume is a temporal object. It does not exist in a single frame -- it is defined
entirely by what changed against sky that used to be there -- so the first thing the
pipeline needs is a background model, not an edge detector. This module owns that model
and nothing else: one stage, one file. It takes a stabilised greyscale frame and returns
the blobs that ``statistic.py`` is allowed to measure.

Order of operations, and why it is that order
---------------------------------------------
``createCLAHE`` equalise -> background subtraction -> ``morphologyEx`` OPEN -> CLOSE ->
``connectedComponentsWithStats``.

CLAHE runs on the *source*, before subtraction, because the discriminating signal at dawn
is a few grey levels of semi-transparent haze against a low-contrast sky, and MOG2's
variance test cannot see a difference that the sensor never encoded. A global
``equalizeHist`` would be the obvious alternative and is the wrong one: its transfer
function is recomputed from the whole frame every minute, so a brightening sky rewrites
the mapping under the background model and every pixel looks like foreground. CLAHE is
tile-local and clip-limited, which bounds both the drift and the noise it amplifies.

OPEN before CLOSE, never the reverse. OPEN with a 3 px kernel removes the single-pixel
sensor speckle that survives subtraction; CLOSE with a 7 px kernel then merges what is
left. Closing first would weld the speckle into the plume and hand S3 a boundary band
made of noise -- and S3 is the statistic the project rests on.

Three subtractors, and why all three are measured
-------------------------------------------------
``method`` selects ``cv2.createBackgroundSubtractorMOG2``,
``cv2.bgsegm.createBackgroundSubtractorLSBP`` or
``cv2.bgsegm.createBackgroundSubtractorGSOC``. These are not three names in a "Built With"
list. Smoke is a slow, semi-transparent change, which is precisely the regime where MOG2's
per-pixel Gaussian mixture is weakest and where LSBP's local-SVD binary pattern and GSOC
are the better-suited family. ``scripts/bench.py --bg {mog2,lsbp,gsoc}`` runs all three on
the same DEV arm and ``DEMO.md`` publishes TTD, alert rate and per-frame latency for each,
**whichever wins** -- whichever wins is also what ships. Naming a contrib module and never
measuring it is the decorative-integration failure this project exists not to commit.

MOG2 is constructed from the project's three frozen constants. LSBP and GSOC are
constructed at their published defaults, deliberately: the comparison being run is "does a
better-suited algorithm at its default beat a tuned MOG2", and hand-tuning two more
subtractors would spend the DEV arm three times over on knobs nobody would then trust.

Warm-up is a state, not a silence
---------------------------------
``apply`` runs and returns components from frame 1. Before ``warm`` the background model
has seen fewer than ``WARMUP_FRAMES`` frames and its output is not yet meaningful, so the
UI shows a designed warming state (``ui.md`` section 7) with the components it actually
got. Returning ``[]`` during warm-up would look like a working detector finding nothing,
which is the one thing a dispatcher must never be shown.
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np

from .constants import (
    CC_CONNECTIVITY,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    MASK_FOREGROUND_VALUE,
    MAX_COMPONENTS_PER_FRAME,
    MIN_COMPONENT_AREA_PX,
    MIN_COMPONENT_EXTENT_PX,
    MOG2_DETECT_SHADOWS,
    MOG2_HISTORY,
    MOG2_VAR_THRESHOLD,
    MORPH_CLOSE_KERNEL,
    MORPH_OPEN_KERNEL,
    WARMUP_FRAMES,
)
from .types import Component

__all__ = ["METHODS", "Segmenter"]


#: The three subtractors ``bench.py --bg`` sweeps. Ordered as the published table is.
METHODS: Final[tuple[str, ...]] = ("mog2", "lsbp", "gsoc")


class Segmenter:
    """Stateful per-camera foreground segmenter. One instance per camera, for the night.

    The state is the background model, which is the whole point: forty cameras means forty
    of these, and the spec section 11 budgets them at roughly 11 MB each at 480x270.

    Args:
        method: ``"mog2"`` | ``"lsbp"`` | ``"gsoc"``. Anything else raises immediately --
            a typo silently falling back to MOG2 would publish a bench row labelled
            ``lsbp`` that was really MOG2, which is worse than a crash.

    Raises:
        ValueError: on an unknown ``method``.
    """

    __slots__ = (
        "_method",
        "_subtractor",
        "_clahe",
        "_open_kernel",
        "_close_kernel",
        "_frames_seen",
        "_shape",
    )

    def __init__(self, method: str = "mog2") -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        self._method = method
        self._subtractor = _new_subtractor(method)

        # CLAHE object and both structuring elements are built once and reused. They are
        # pure functions of constants, and rebuilding them per frame is the easiest
        # milliseconds in the pipeline to waste at 40 cameras x 81 frames.
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=(CLAHE_TILE_GRID, CLAHE_TILE_GRID),
        )
        # Ellipse, not rectangle: a rectangular CLOSE kernel bridges preferentially along
        # the image axes, which would square off a plume that is drifting diagonally and
        # feed S2 a width that came from the kernel rather than from the smoke.
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL)
        )
        self._close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL)
        )
        self._frames_seen = 0
        # The frame shape this model was built against, latched on the first apply. See
        # apply's shape check for why an instance refuses to change size mid-night.
        self._shape: tuple[int, int] | None = None

    # -----------------------------------------------------------------------------

    @property
    def method(self) -> str:
        """Which subtractor this instance is actually running -- what a bench row is labelled with."""
        return self._method

    @property
    def frames_seen(self) -> int:
        """Frames handed to ``apply`` since construction or the last ``reset``."""
        return self._frames_seen

    @property
    def warm(self) -> bool:
        """True once the background model has seen ``WARMUP_FRAMES`` frames.

        This is advisory, not a gate: ``apply`` returns components either way and the UI
        renders the warming state around them.
        """
        return self._frames_seen >= WARMUP_FRAMES

    def reset(self) -> None:
        """Drop the background model and the frame count. Used between replay windows.

        The subtractor is rebuilt rather than cleared because none of the three exposes a
        clear: MOG2, LSBP and GSOC all accumulate their history internally with no public
        way back to frame zero. Reusing an instance across two camera-nights would carry
        one night's sky into the next night's variance, so a new object is the only honest
        reset available.
        """
        self._subtractor = _new_subtractor(self._method)
        self._frames_seen = 0
        self._shape = None

    # -----------------------------------------------------------------------------

    def apply(self, grey: np.ndarray) -> list[Component]:
        """Segment one stabilised greyscale frame into foreground components.

        Args:
            grey: 2-D ``uint8`` image, already stabilised onto the previous frame. The
                stabilisation is a precondition and is not checked here -- MOG2 in an
                unstabilised frame sees the whole ridgeline as foreground the moment the
                camera twitches, and there is no way to detect that from one frame.

        Returns:
            Components with area at least ``MIN_COMPONENT_AREA_PX``, largest first, at
            most ``MAX_COMPONENTS_PER_FRAME`` of them. Possibly empty. The list is
            returned before ``warm`` as well as after.

        Raises:
            ValueError: if ``grey`` is not a non-empty 2-D ``uint8`` array, or if its
                shape differs from the first frame this instance saw. A colour frame or a
                float image would still produce a plausible-looking mask, and a plausible
                wrong answer here propagates into every statistic downstream.
        """
        if not isinstance(grey, np.ndarray):
            raise ValueError(f"grey must be a numpy array, got {type(grey).__name__}")
        if grey.ndim != 2:
            raise ValueError(f"grey must be 2-D single-channel, got shape {grey.shape}")
        if grey.dtype != np.uint8:
            raise ValueError(f"grey must be uint8, got {grey.dtype}")
        if grey.size == 0:
            raise ValueError("grey is empty")

        shape = (int(grey.shape[0]), int(grey.shape[1]))
        if self._shape is None:
            self._shape = shape
        elif shape != self._shape:
            # All three subtractors silently reallocate and restart when handed a
            # different frame size: apply() would return a plausible mask, frames_seen
            # would keep counting and warm would keep saying True over a model that has
            # in fact seen one frame. That is the exact shape of a silent wrong answer,
            # so it is a raise. A genuine resolution change is a new camera-night and
            # therefore a reset(), which is the caller's decision to make explicitly.
            raise ValueError(
                f"grey shape {shape} differs from this segmenter's {self._shape}; "
                "call reset() to start a new background model"
            )

        equalised = self._clahe.apply(grey)
        raw_mask = self._subtractor.apply(equalised)
        self._frames_seen += 1

        binary = _binarise(raw_mask)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._open_kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._close_kernel)

        return _components(closed)


# ---------------------------------------------------------------------------------
# Construction and labelling
# ---------------------------------------------------------------------------------


def _new_subtractor(method: str) -> cv2.BackgroundSubtractor:
    """Build one subtractor. Separate from ``__init__`` so ``reset`` cannot drift from it."""
    if method == "mog2":
        return cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
    if method == "lsbp":
        return cv2.bgsegm.createBackgroundSubtractorLSBP()
    if method == "gsoc":
        return cv2.bgsegm.createBackgroundSubtractorGSOC()
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def _binarise(raw_mask: np.ndarray) -> np.ndarray:
    """Turn a subtractor's raw mask into the strict 0/255 mask morphologyEx wants.

    ``compare(..., CMP_EQ)`` rather than ``raw_mask > 0``, and equality rather than a
    threshold, because a subtractor that is labelling shadows returns them as 127. With
    MOG2_DETECT_SHADOWS False today the two are identical on every frame -- which is
    precisely why this is a named function with its own test rather than an inline
    expression: the day the flag flips, ``> 0`` would promote every cloud shadow at dawn
    into a component and nothing would fail.

    ``cv2.compare`` over a numpy comparison because numpy yields a bool array that
    ``morphologyEx`` cannot take, so the alternative is a second conversion pass per
    frame per camera.
    """
    return cv2.compare(raw_mask, MASK_FOREGROUND_VALUE, cv2.CMP_EQ)  # type: ignore[call-overload]


def _components(mask: np.ndarray) -> list[Component]:
    """Label a cleaned binary mask and return the components worth measuring.

    ``connectedComponentsWithStats`` rather than ``findContours`` because bbox, area and
    centroid all arrive from the same single pass -- and because S1 is literally a claim
    about ``CC_STAT_TOP + CC_STAT_HEIGHT`` holding still.
    """
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=CC_CONNECTIVITY, ltype=cv2.CV_32S
    )

    out: list[Component] = []
    # Label 0 is background by definition of the algorithm, so the range starts at 1.
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA_PX:
            continue
        # Area is not enough. A 60x1 sliver clears the area floor and then has no interior
        # once S3 takes its boundary band off both sides, so statistic.measure returns a
        # nan S3 and rank.fuse -- correctly -- refuses to score it. Both of those are right
        # and the pipeline still stops, because nobody owned the question "which components
        # is the primitive defined on?". It is owned here: a component narrower than
        # MIN_COMPONENT_EXTENT_PX in either dimension is below the size at which ANY of the
        # four statistics can say something (S1's anchor tolerance alone is half of it), so
        # it is not a plume candidate at this working resolution. Filtering it here is the
        # honest fix; clamping the nan to 0.0 downstream would have published "hard edge,
        # flat interior" -- a confident answer -- about a shape nothing measured.
        if (
            int(stats[label, cv2.CC_STAT_WIDTH]) < MIN_COMPONENT_EXTENT_PX
            or int(stats[label, cv2.CC_STAT_HEIGHT]) < MIN_COMPONENT_EXTENT_PX
        ):
            continue
        out.append(
            Component(
                x=int(stats[label, cv2.CC_STAT_LEFT]),
                y=int(stats[label, cv2.CC_STAT_TOP]),
                w=int(stats[label, cv2.CC_STAT_WIDTH]),
                h=int(stats[label, cv2.CC_STAT_HEIGHT]),
                area=area,
                cx=float(centroids[label, 0]),
                cy=float(centroids[label, 1]),
            )
        )

    # Largest first, then truncate: the cap exists to bound the per-frame cost of the
    # statistic, and spending it on the eight smallest blobs in a noisy frame would drop
    # the plume in favour of speckle that only just cleared the area floor.
    out.sort(key=lambda c: c.area, reverse=True)
    return out[:MAX_COMPONENTS_PER_FRAME]

"""The dataclasses the pipeline stages hand each other.

One file so that the seam between stages is written down once. Every stage takes one of
these and returns another; nothing passes a bare tuple across a module boundary.

    stabilise.stabilise(prev, cur)      -> Stabilised
    segment.Segmenter.apply(frame)      -> list[Component]
    statistic.measure(track)            -> Stats
    rank.fuse(stats, escalation)        -> float          (the hazard)
    rank.emit(scores, threshold, state) -> list[WatchEntry]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence as _Sequence

import numpy as np

# ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Stabilised:
    """One frame, registered onto the previous one."""

    frame: np.ndarray
    """The warped frame, same shape and dtype as the input."""

    dx: float
    dy: float
    """Estimated camera shift in working pixels. Positive dx means the scene moved right."""

    response: float
    """cv2.phaseCorrelate's peak response. Below constants.PHASE_RESPONSE_MIN the
    fallback ran, and ``method`` says so."""

    method: str
    """``"phase"`` | ``"phase+iterative"`` | ``"affine"`` | ``"passthrough"``.
    ``passthrough`` means the estimate was rejected as implausible and the frame was
    left alone -- an honest failure that the residual-error distribution publishes,
    rather than a wrong warp applied quietly."""

    @property
    def shift_px(self) -> float:
        return float(np.hypot(self.dx, self.dy))


@dataclass(frozen=True, slots=True)
class Component:
    """One foreground blob, straight out of connectedComponentsWithStats."""

    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float

    @property
    def bottom(self) -> int:
        """Lowest extent in image coordinates -- y grows downward, so this is y + h.
        S1 is a claim about *this* number staying still."""
        return self.y + self.h

    @property
    def top(self) -> int:
        return self.y

    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass(slots=True)
class Track:
    """One component followed across frames, which is what makes S1 and S2 possible.

    A statistic about growth cannot be computed from a single frame, so the tracker is
    part of the primitive rather than a convenience.
    """

    track_id: int
    history: list[Component] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)
    """Frame indices parallel to ``history``, so a gap is visible rather than smoothed
    over."""

    def add(self, frame_index: int, component: Component, window: int) -> None:
        self.history.append(component)
        self.frames.append(frame_index)
        if len(self.history) > window:
            del self.history[0]
            del self.frames[0]

    @property
    def last(self) -> Component:
        return self.history[-1]

    def __len__(self) -> int:
        return len(self.history)


@dataclass(frozen=True, slots=True)
class Stats:
    """The anchored plume-growth statistic. The thing OpenCV does not ship."""

    s1: float
    """Base anchoring in [0, 1]: 1.0 when the lowest extent has not moved while the top
    climbed. A fire is tied to the ground; a cloud is not."""

    s2: float
    """Vertical growth anisotropy in [0, 1]: d(height)/dt over d(width)/dt, squashed.
    A plume rises faster than it spreads."""

    s3: float
    """Diffuse-boundary contrast in [0, 1]: interior Laplacian energy over boundary-band
    Sobel energy, squashed. Smoke's edge is soft; a cloud shadow's is hard. THIS is the
    statistic the Sep 1 gate tested, and the only one that separates smoke from fog."""

    s4: float
    """DIS optical-flow drift inside the bounding box, in [0, 1]. Under ablation gate."""

    raw_s3: float = 0.0
    """S3 before squashing -- the units the gate reported in, kept so the published AUC
    and the running system are talking about the same number."""

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.s1, self.s2, self.s3, self.s4


@dataclass(frozen=True, slots=True)
class CameraScore:
    """One camera's answer for one minute, before ranking."""

    camera_id: str
    frame_index: int
    hazard: float
    stats: Stats
    component: Component | None
    escalated: bool = False
    escalation_score: float = 0.0
    stabilisation: Stabilised | None = None


@dataclass(frozen=True, slots=True)
class WatchEntry:
    """One row of the watch list -- what the dispatcher actually reads, and what the SSE
    stream carries."""

    rank: int
    camera_id: str
    hazard: float
    s1: float
    s2: float
    s3: float
    s4: float
    escalated: bool
    alerted: bool
    """True when this entry spent one of the camera's two alerts for the night."""
    budget_spent: int
    """Alerts spent on this camera so far tonight, 0..2."""
    evidence_url: str | None = None

    def to_json(self) -> dict:
        return {
            "rank": self.rank,
            "camera_id": self.camera_id,
            "hazard": round(self.hazard, 4),
            "s1": round(self.s1, 4),
            "s2": round(self.s2, 4),
            "s3": round(self.s3, 4),
            "s4": round(self.s4, 4),
            "escalated": self.escalated,
            "alerted": self.alerted,
            "budget_spent": self.budget_spent,
            "evidence_url": self.evidence_url,
        }


@dataclass(frozen=True, slots=True)
class Sequence:
    """One replay window: 81 frames on one camera, with an ignition clock if it has one.

    ``t0_index`` is ground truth and is NEVER visible to the pipeline -- only to
    metrics.py, after the fact. A CONFUSE window has ``t0_index is None``: nothing ever
    ignites, which is exactly what makes it a false-alarm denominator.
    """

    sequence_id: str
    camera_id: str
    arm: str
    """``"dev"`` | ``"hold"`` | ``"confuse"`` | ``"micro"``."""
    frame_paths: _Sequence[str]
    t0_index: int | None = None

    @property
    def is_fire(self) -> bool:
        return self.t0_index is not None

    def __len__(self) -> int:
        return len(self.frame_paths)

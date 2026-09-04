"""Time-to-detection against the ignition clock. The first thing in this repo that ran.

This module was written before any detector existed, and that ordering is the project's
central methodological claim rather than a scheduling accident. The literature reports F1
on individual frames. A dispatcher does not experience F1; she experiences *how many
minutes after ignition somebody looked at the right camera, given that she will only
tolerate a fixed number of interruptions per camera per night*. Nobody publishes that
trade-off, every deployed system lives somewhere on it, and it is the number this project
is scored on.

Ground truth here is mechanical and free: HPWREN FIgLib records the ignition time of each
sequence, so time-to-detection needs zero labelling -- no raters, no rig, no institutional
partner. That is why this metric is affordable and why an F1 metric is not the honest one.

Two definitions that the rest of the codebase must not quietly re-invent
-----------------------------------------------------------------------
**A detection** is the first minute at which the true camera is BOTH inside the top-k
watch list AND firing an alert. Being ranked 27th of 40 is not a detection; a dispatcher
never reads row 27. Both halves are required, and ``detection_index`` takes both.

**A miss** is a sequence with no detection at any point in its window. Misses are NOT
dropped and NOT imputed. ``percentiles`` treats a miss as ``+inf``, which propagates
honestly: if more than half the sequences are missed, the reported p50 *is* infinite, and
that is the correct thing for a system that fails half the time. ``TTDSummary`` carries
``n_missed`` alongside every percentile so the number can never be read without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from .constants import (
    ALERTS_PER_CAMERA_PER_NIGHT,
    FRAMES_PER_WINDOW,
    MINUTES_PER_FRAME,
)

__all__ = [
    "TTDSummary",
    "detection_index",
    "time_to_detection",
    "percentiles",
    "summarise",
    "alerts_per_camera_night",
    "Latency",
]


# ---------------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------------


def detection_index(
    in_top_k: Sequence[bool],
    alerted: Sequence[bool],
    t0_index: int,
) -> int | None:
    """Index of the first frame at or after ``t0_index`` that is a detection.

    Args:
        in_top_k: per-frame, whether the true camera was inside the top-k watch list.
        alerted: per-frame, whether the true camera fired an alert that minute.
        t0_index: the ignition frame. Ground truth; never visible to the pipeline.

    Returns:
        The frame index, or ``None`` if the sequence was missed.

    Frames before ``t0_index`` are ignored rather than counted as false alarms here --
    a pre-ignition alert is a real event and it is counted, but it is counted where it
    belongs, in the alert *rate* (``alerts_per_camera_night``), not as a negative
    detection latency. Crediting a system for alerting before the fire started would be
    the single easiest way to fake this metric.
    """
    if len(in_top_k) != len(alerted):
        raise ValueError(
            f"in_top_k and alerted must be parallel: {len(in_top_k)} != {len(alerted)}"
        )
    if t0_index < 0:
        raise ValueError(f"t0_index must be non-negative, got {t0_index}")

    for i in range(t0_index, len(in_top_k)):
        if in_top_k[i] and alerted[i]:
            return i
    return None


def time_to_detection(
    in_top_k: Sequence[bool],
    alerted: Sequence[bool],
    t0_index: int,
    minutes_per_frame: float = MINUTES_PER_FRAME,
) -> float:
    """Minutes from ignition to detection. ``inf`` for a miss, never ``None``.

    Returning ``inf`` rather than ``None`` is deliberate: a miss is an infinitely late
    detection, it sorts correctly, and it cannot be silently dropped by a downstream
    ``[x for x in ttds if x is not None]``.
    """
    idx = detection_index(in_top_k, alerted, t0_index)
    if idx is None:
        return math.inf
    return (idx - t0_index) * minutes_per_frame


# ---------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TTDSummary:
    """A headline number that cannot be quoted without its denominator."""

    p50: float
    p95: float
    n: int
    n_missed: int
    n_detected: int
    mean_detected: float
    """Mean over detected sequences only. Reported for completeness; it is NOT the
    headline, because a mean over survivors is exactly how a miss-heavy system flatters
    itself."""

    @property
    def detection_rate(self) -> float:
        return self.n_detected / self.n if self.n else 0.0

    def format_line(self, label: str) -> str:
        p50 = "  miss" if math.isinf(self.p50) else f"{self.p50:6.1f}"
        p95 = "  miss" if math.isinf(self.p95) else f"{self.p95:6.1f}"
        return (
            f"{label:<16} p50 {p50} min   p95 {p95} min"
            f"   ({self.n_detected}/{self.n} detected)"
        )

    def to_json(self) -> dict:
        def num(v: float) -> float | None:
            return None if math.isinf(v) or math.isnan(v) else round(v, 3)

        return {
            "p50_minutes": num(self.p50),
            "p95_minutes": num(self.p95),
            "n": self.n,
            "n_detected": self.n_detected,
            "n_missed": self.n_missed,
            "detection_rate": round(self.detection_rate, 4),
            "mean_detected_minutes": num(self.mean_detected),
        }


def percentiles(ttds: Iterable[float], qs: Sequence[float] = (50.0, 95.0)) -> list[float]:
    """Percentiles over a set that may contain ``inf``.

    ``numpy.percentile`` with linear interpolation produces ``nan`` the moment an ``inf``
    is involved in an interpolation, which would turn "half our sequences were missed"
    into "not a number". So the ranking is done on order statistics instead: sort, index,
    and let an ``inf`` at the chosen position stay an ``inf``. The result is the *lower*
    order statistic at each quantile -- pessimistic by at most one sequence, and never
    silently absent.
    """
    values = sorted(float(v) for v in ttds)
    if not values:
        raise ValueError("percentiles() of an empty set -- the caller has lost its arm")

    out: list[float] = []
    n = len(values)
    for q in qs:
        if not 0.0 <= q <= 100.0:
            raise ValueError(f"percentile {q} outside [0, 100]")
        # Nearest-rank: the smallest value at or above which q% of the set sits.
        rank = max(1, math.ceil(q / 100.0 * n))
        out.append(values[rank - 1])
    return out


def summarise(ttds: Iterable[float]) -> TTDSummary:
    """The headline, with its denominator attached."""
    values = [float(v) for v in ttds]
    if not values:
        raise ValueError("summarise() of an empty arm")

    detected = [v for v in values if math.isfinite(v)]
    p50, p95 = percentiles(values, (50.0, 95.0))
    return TTDSummary(
        p50=p50,
        p95=p95,
        n=len(values),
        n_missed=len(values) - len(detected),
        n_detected=len(detected),
        mean_detected=float(np.mean(detected)) if detected else math.inf,
    )


# ---------------------------------------------------------------------------------
# The budget side of the metric
# ---------------------------------------------------------------------------------


def alerts_per_camera_night(n_alerts: int, window_lengths: Sequence[int]) -> float:
    """Realised alert rate, in the unit the whole project is calibrated to.

    One camera-night is ``FRAMES_PER_WINDOW`` frames on one camera. DEV, HOLD and CONFUSE
    windows are the same span on the same cameras at the same time of day, which is what
    makes this denominator comparable across the arms -- and what makes a like-for-like
    comparison at a fixed operating point possible at all.

    **This takes the length of every window, not a count of them, and that is the whole
    point of the signature.** FIgLib does not publish a uniform window: of the 30
    committed DEV sequences 21 carry 80 frames and 9 carry 81, and HOLD splits 33/27. A
    scalar ``frames_per_window`` was the earlier shape, and against that corpus it
    published a rate biased high by ~0.9% -- inside ``BUDGET_DRIFT_TOLERANCE``, therefore
    invisible, and sitting underneath the one number the whole project calibrates to. The
    lengths are already in the caller's hands in every case, so asking for them removes
    the failure rather than documenting it.

    A window shorter than a full night counts as the fraction of a night it actually is.
    """
    if n_alerts < 0:
        raise ValueError(f"n_alerts must be non-negative, got {n_alerts}")
    lengths = list(window_lengths)
    if not lengths:
        raise ValueError("no windows -- the caller has lost its arm")
    if any(n < 1 for n in lengths):
        raise ValueError(f"every window must have at least one frame, got {lengths!r}")
    return n_alerts / (sum(lengths) / FRAMES_PER_WINDOW)


def budget_ceiling(n_cameras: int) -> int:
    """Total alerts a night may contain before the budget is spent everywhere."""
    return n_cameras * ALERTS_PER_CAMERA_PER_NIGHT


# ---------------------------------------------------------------------------------
# Latency -- measured with OpenCV's own clock, not with time.time()
# ---------------------------------------------------------------------------------


class Latency:
    """Per-stage wall clock via ``cv2.TickMeter``.

    OpenCV's tick counter rather than ``time.perf_counter`` on purpose: the published
    per-frame latency table is compared against OpenCV's own zoo benchmarks (MobileNetV2
    at 224x224 = 57.91 ms on a Cortex-A72), and a number measured with the library's own
    clock is the one that belongs in that table.

    Usage::

        lat = Latency()
        with lat.stage("classical"):
            ...
        lat.report()["classical"]["p50_ms"]
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}

    class _Stage:
        __slots__ = ("_owner", "_name", "_meter")

        def __init__(self, owner: "Latency", name: str) -> None:
            self._owner = owner
            self._name = name
            self._meter = cv2.TickMeter()

        def __enter__(self) -> "Latency._Stage":
            self._meter.start()
            return self

        def __exit__(self, *exc: object) -> None:
            self._meter.stop()
            ms = self._meter.getTimeMilli()
            self._owner._samples.setdefault(self._name, []).append(float(ms))

    def stage(self, name: str) -> "Latency._Stage":
        return Latency._Stage(self, name)

    def report(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, samples in sorted(self._samples.items()):
            arr = np.asarray(samples, dtype=float)
            out[name] = {
                "n": int(arr.size),
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "mean_ms": round(float(arr.mean()), 3),
                "total_ms": round(float(arr.sum()), 3),
            }
        return out

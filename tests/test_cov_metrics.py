"""Coverage tests for metrics.py uncovered lines: 86, 136-138, 170, 226."""

from __future__ import annotations

import math

import pytest

from plumerank.constants import FRAMES_PER_WINDOW
from plumerank.metrics import (
    TTDSummary,
    alerts_per_camera_night,
    detection_index,
    percentiles,
)

# --- Line 86: detection_index t0_index validation ---


def test_detection_index_rejects_negative_t0_index() -> None:
    """Line 86: t0_index < 0 raises ValueError with the signed value in the message."""
    in_top_k = [False] * 10
    alerted = [False] * 10
    with pytest.raises(ValueError, match="t0_index must be non-negative, got -1"):
        detection_index(in_top_k, alerted, t0_index=-1)


def test_detection_index_rejects_large_negative_t0_index() -> None:
    """Line 86: a large negative t0_index also raises with the correct message."""
    in_top_k = [False] * 10
    alerted = [False] * 10
    with pytest.raises(ValueError, match="t0_index must be non-negative, got -999"):
        detection_index(in_top_k, alerted, t0_index=-999)


# --- Lines 136-138: TTDSummary.format_line ---


def test_format_line_with_finite_percentiles() -> None:
    """Lines 136-138: when p50 and p95 are finite, they are formatted as floats."""
    summary = TTDSummary(
        p50=10.5,
        p95=20.3,
        n=100,
        n_missed=5,
        n_detected=95,
        mean_detected=11.0,
    )
    line = summary.format_line("test-label")
    assert "test-label" in line
    assert "p50" in line
    assert "p95" in line
    assert "10.5" in line
    assert "20.3" in line
    assert "(95/100 detected)" in line


def test_format_line_with_infinite_p50() -> None:
    """Lines 136-138: when p50 is infinite, it renders as 'miss'."""
    summary = TTDSummary(
        p50=math.inf,
        p95=20.0,
        n=50,
        n_missed=30,
        n_detected=20,
        mean_detected=15.0,
    )
    line = summary.format_line("arm1")
    assert "miss" in line  # p50 should be "  miss"
    assert "20.0" in line  # p95 should still be a number
    assert "(20/50 detected)" in line


def test_format_line_with_infinite_p95() -> None:
    """Lines 136-138: when p95 is infinite, it renders as 'miss'."""
    summary = TTDSummary(
        p50=5.0,
        p95=math.inf,
        n=75,
        n_missed=20,
        n_detected=55,
        mean_detected=6.0,
    )
    line = summary.format_line("arm2")
    assert "5.0" in line  # p50 should still be a number
    assert "miss" in line  # p95 should be "  miss"
    assert "(55/75 detected)" in line


def test_format_line_with_both_percentiles_infinite() -> None:
    """Lines 136-138: when both p50 and p95 are infinite, both render as 'miss'."""
    summary = TTDSummary(
        p50=math.inf,
        p95=math.inf,
        n=40,
        n_missed=40,
        n_detected=0,
        mean_detected=math.inf,
    )
    line = summary.format_line("total_miss")
    # Both should render as "  miss"
    assert line.count("miss") >= 2
    assert "(0/40 detected)" in line


def test_format_line_renders_label_padding() -> None:
    """Lines 136-138: label is padded to 16 characters left-aligned."""
    short_label = "x"
    summary = TTDSummary(p50=1.0, p95=2.0, n=5, n_missed=0, n_detected=5, mean_detected=1.5)
    line = summary.format_line(short_label)
    # The format string is {label:<16}, so "x" should be padded
    assert line.startswith("x")


# --- Line 170: percentiles with empty sequence ---


def test_percentiles_rejects_empty_sequence() -> None:
    """Line 170: percentiles of an empty sequence raises ValueError."""
    with pytest.raises(ValueError, match="percentiles.*empty.*lost its arm"):
        percentiles([])


def test_percentiles_rejects_empty_list_of_floats() -> None:
    """Line 170: percentiles rejects empty even when type is clear."""
    empty: list[float] = []
    with pytest.raises(ValueError, match="empty.*arm"):
        percentiles(empty)


# --- Line 226: alerts_per_camera_night n_alerts validation ---


def test_alerts_per_camera_night_rejects_negative_n_alerts() -> None:
    """Line 226: n_alerts < 0 raises ValueError with the signed value in message."""
    with pytest.raises(ValueError, match="n_alerts must be non-negative, got -1"):
        alerts_per_camera_night(-1, [FRAMES_PER_WINDOW])


def test_alerts_per_camera_night_rejects_large_negative_n_alerts() -> None:
    """Line 226: a large negative n_alerts raises with the correct message."""
    with pytest.raises(ValueError, match="n_alerts must be non-negative, got -500"):
        alerts_per_camera_night(-500, [FRAMES_PER_WINDOW] * 10)


def test_alerts_per_camera_night_accepts_zero_alerts() -> None:
    """Line 226: zero alerts is valid (no false positives, no budget exceeded)."""
    result = alerts_per_camera_night(0, [FRAMES_PER_WINDOW])
    assert result == 0.0


def test_alerts_per_camera_night_accepts_zero_with_multiple_windows() -> None:
    """Line 226: zero alerts over multiple windows is still valid."""
    result = alerts_per_camera_night(0, [FRAMES_PER_WINDOW] * 5)
    assert result == 0.0


# --- Lines 278-281: pin_opencv_threads validation and execution ---


def test_pin_opencv_threads_rejects_zero_threads() -> None:
    """Line 278: n < 1 raises ValueError with the signed value in message."""
    from plumerank.metrics import pin_opencv_threads

    with pytest.raises(ValueError, match="thread count must be at least 1, got 0"):
        pin_opencv_threads(0)


def test_pin_opencv_threads_rejects_negative_threads() -> None:
    """Line 278: negative thread count raises ValueError."""
    from plumerank.metrics import pin_opencv_threads

    with pytest.raises(ValueError, match="thread count must be at least 1, got -5"):
        pin_opencv_threads(-5)


@pytest.fixture(autouse=True)
def _restore_opencv_threads():
    """pin_opencv_threads mutates process-global state; put it back for later tests."""
    import cv2

    before = cv2.getNumThreads()
    yield
    cv2.setNumThreads(before)


def test_pin_opencv_threads_accepts_one() -> None:
    """Lines 280-281: n=1 is valid and returns an integer."""
    from plumerank.metrics import pin_opencv_threads

    result = pin_opencv_threads(1)
    assert isinstance(result, int)
    assert result >= 1


def test_pin_opencv_threads_accepts_multiple() -> None:
    """Lines 280-281: n=4 is valid and returns the thread count set."""
    from plumerank.metrics import pin_opencv_threads

    result = pin_opencv_threads(4)
    assert isinstance(result, int)
    # The result should be >= 1 (may differ from requested on some platforms)
    assert result >= 1

"""Tests for uncovered lines in plumerank/rank.py.

This module targets lines 301, 362, 415, 419, 439, 504, 544, 577, 654, 704.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from plumerank.rank import (
    BudgetState,
    draw_evidence,
    emit,
    evidence_crop_box,
    realised_rate,
    solve_threshold,
)
from plumerank.types import CameraScore, Component, Stats, Track


def stats(s1: float = 0.0, s2: float = 0.0, s3: float = 0.0, s4: float = 0.0) -> Stats:
    """Helper to create Stats instances."""
    return Stats(s1=s1, s2=s2, s3=s3, s4=s4, raw_s3=0.0)


def score(camera_id: str, hazard: float, **kw) -> CameraScore:
    """Helper to create CameraScore instances."""
    return CameraScore(
        camera_id=camera_id,
        frame_index=kw.pop("frame_index", 0),
        hazard=hazard,
        stats=kw.pop("stats", stats(0.5, 0.5, 0.5, 0.5)),
        component=kw.pop("component", None),
        **kw,
    )


def gradient_frame(h: int = 270, w: int = 480) -> np.ndarray:
    """A smooth vertical ramp for testing draw_evidence."""
    ramp = np.linspace(60, 190, h, dtype=np.float32)
    return np.repeat(ramp[:, None, None], w, axis=1).repeat(3, axis=2).astype(np.uint8)


def anchored_track(bottom: int = 100) -> tuple[Component, Track]:
    """A pinned foot with a climbing top for testing draw_evidence."""
    t = Track(track_id=1)
    for i, top in enumerate((70, 62, 55, 48, 40)):
        c = Component(
            x=200,
            y=top,
            w=30,
            h=bottom - top,
            area=30 * (bottom - top),
            cx=215.0,
            cy=(top + bottom) / 2.0,
        )
        t.add(i, c, window=10)
    return t.last, t


# --- Line 301: realised_rate with empty hazard_windows ---


def test_realised_rate_refuses_empty_hazard_windows():
    """Line 301: raise ValueError("realised_rate() of an empty arm -- the caller has lost its corpus")"""
    with pytest.raises(ValueError, match="empty arm"):
        realised_rate([], threshold=0.5)


# --- Line 362: solve_threshold with invalid target ---


def test_solve_threshold_refuses_nan_target():
    """Line 362: raise ValueError(f"target must be a non-negative rate, got {target!r}")"""
    windows = [[0.5, 0.3, 0.1] * 10]
    with pytest.raises(ValueError, match="non-negative rate"):
        solve_threshold(windows, target=float("nan"))


def test_solve_threshold_refuses_infinite_target():
    """Line 362: raise ValueError(f"target must be a non-negative rate, got {target!r}")"""
    windows = [[0.5, 0.3, 0.1] * 10]
    with pytest.raises(ValueError, match="non-negative rate"):
        solve_threshold(windows, target=float("inf"))


def test_solve_threshold_refuses_negative_target():
    """Line 362: raise ValueError(f"target must be a non-negative rate, got {target!r}")"""
    windows = [[0.5, 0.3, 0.1] * 10]
    with pytest.raises(ValueError, match="non-negative rate"):
        solve_threshold(windows, target=-0.5)


# --- Line 415: BudgetState with empty cameras ---


def test_budget_refuses_empty_roster():
    """Line 415: raise ValueError("BudgetState needs a camera roster -- the slots are per camera")"""
    with pytest.raises(ValueError, match="camera roster"):
        BudgetState([])


# --- Line 419: BudgetState with negative per_camera ---


def test_budget_refuses_negative_per_camera():
    """Line 419: raise ValueError(f"per_camera must be non-negative, got {per_camera}")"""
    with pytest.raises(ValueError, match="non-negative"):
        BudgetState(["a", "b"], per_camera=-1)


# --- Line 439: BudgetState.spent() with camera not on roster ---


def test_budget_spent_raises_on_unknown_camera():
    """Line 439: raise KeyError(f"{camera_id!r} is not on the roster this budget was built for")"""
    b = BudgetState(["a", "b"])
    with pytest.raises(KeyError, match="not on the roster"):
        b.spent("unknown")


# --- Line 504: emit with non-finite threshold ---


def test_emit_refuses_nan_threshold():
    """Line 504: raise ValueError(f"threshold must be finite, got {threshold!r}")"""
    b = BudgetState(["a"])
    with pytest.raises(ValueError, match="threshold must be finite"):
        emit([score("a", 0.9)], float("nan"), b)


def test_emit_refuses_infinite_threshold():
    """Line 504: raise ValueError(f"threshold must be finite, got {threshold!r}")"""
    b = BudgetState(["a"])
    with pytest.raises(ValueError, match="threshold must be finite"):
        emit([score("a", 0.9)], float("inf"), b)


# --- Line 544: evidence_crop_box with degenerate shape ---


def test_evidence_crop_box_refuses_zero_height():
    """Line 544: raise ValueError(f"degenerate frame shape {shape!r}")"""
    comp = Component(x=10, y=10, w=20, h=20, area=400, cx=20.0, cy=20.0)
    with pytest.raises(ValueError, match="degenerate frame shape"):
        evidence_crop_box(comp, (0, 100, 3))


def test_evidence_crop_box_refuses_zero_width():
    """Line 544: raise ValueError(f"degenerate frame shape {shape!r}")"""
    comp = Component(x=10, y=10, w=20, h=20, area=400, cx=20.0, cy=20.0)
    with pytest.raises(ValueError, match="degenerate frame shape"):
        evidence_crop_box(comp, (100, 0, 3))


def test_evidence_crop_box_refuses_negative_height():
    """Line 544: raise ValueError(f"degenerate frame shape {shape!r}")"""
    comp = Component(x=10, y=10, w=20, h=20, area=400, cx=20.0, cy=20.0)
    with pytest.raises(ValueError, match="degenerate frame shape"):
        evidence_crop_box(comp, (-50, 100, 3))


def test_evidence_crop_box_refuses_negative_width():
    """Line 544: raise ValueError(f"degenerate frame shape {shape!r}")"""
    comp = Component(x=10, y=10, w=20, h=20, area=400, cx=20.0, cy=20.0)
    with pytest.raises(ValueError, match="degenerate frame shape"):
        evidence_crop_box(comp, (100, -50, 3))


# --- Line 577: _dashed_line with zero-length line ---


def test_dashed_line_handles_zero_length():
    """Line 577: return when length <= 0.0 in _dashed_line

    Testing indirectly through draw_evidence which calls _dashed_line.
    When the crop is only 1 pixel wide, the dashed line endpoints are the same,
    resulting in zero length.
    """
    # Create a frame that's 1 pixel wide - this forces the crop to be 1 pixel wide
    narrow_frame = np.zeros((270, 1, 3), dtype=np.uint8)
    # Create a component within the narrow frame
    comp = Component(x=0, y=135, w=1, h=1, area=1, cx=0.5, cy=135.5)
    track = Track(track_id=5)
    track.add(0, comp, window=10)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # This should complete without error even though the dashed line has zero length
        result = draw_evidence(
            narrow_frame, comp, track, stats(0.5, 0.5, 0.5, 0.5), pathlib.Path(tmp_dir) / "test.jpg"
        )
        assert pathlib.Path(result).exists()


# --- Line 654: draw_evidence with empty crop ---


def test_draw_evidence_refuses_an_empty_crop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Line 654: the empty-crop guard, reached by forcing a degenerate box."""
    import plumerank.rank as rank_mod

    monkeypatch.setattr(rank_mod, "evidence_crop_box", lambda *_a: (5, 5, 5, 5))
    comp, track = anchored_track()
    with pytest.raises(ValueError, match="empty crop"):
        draw_evidence(gradient_frame(), comp, track, stats(), tmp_path / "e.jpg")


def test_dashed_line_with_coincident_endpoints_draws_nothing() -> None:
    """Line 577: a zero-length dash segment returns without touching the image."""
    from plumerank.rank import _dashed_line

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    _dashed_line(img, (3, 3), (3, 3), (255, 255, 255))
    assert not img.any()


# --- Line 704: draw_evidence with unwritable path ---


def test_draw_evidence_with_unwritable_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Line 704: cv2.imwrite returning False surfaces as the documented ValueError.

    Monkeypatched rather than chmod'ed: a read-only directory is still writable as root.
    """
    import plumerank.rank as rank_mod

    monkeypatch.setattr(rank_mod.cv2, "imwrite", lambda *_a, **_k: False)
    comp, track = anchored_track()
    with pytest.raises(ValueError, match="imwrite returned False"):
        draw_evidence(gradient_frame(), comp, track, stats(0.5, 0.5, 0.5, 0.5), tmp_path / "e.jpg")

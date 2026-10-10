"""Tests for plumerank.types dataclasses, targeting uncovered lines 199 and 202."""

from __future__ import annotations

import numpy as np
import pytest

from plumerank.types import (
    CameraScore,
    Component,
    Sequence,
    Stabilised,
    Stats,
    Track,
    WatchEntry,
)

# --- Sequence ------------------------------------------------------------------


def test_sequence_is_fire_true_when_t0_index_is_not_none():
    """Line 199: Sequence.is_fire returns True when t0_index is set."""
    seq = Sequence(
        sequence_id="seq1",
        camera_id="cam1",
        arm="dev",
        frame_paths=["f1.png", "f2.png"],
        t0_index=10,
    )
    assert seq.is_fire is True


def test_sequence_is_fire_false_when_t0_index_is_none():
    """Line 199: Sequence.is_fire returns False when t0_index is None."""
    seq = Sequence(
        sequence_id="seq2",
        camera_id="cam2",
        arm="confuse",
        frame_paths=["f1.png", "f2.png"],
        t0_index=None,
    )
    assert seq.is_fire is False


def test_sequence_len_returns_frame_paths_length():
    """Line 202: Sequence.__len__ returns the length of frame_paths."""
    frame_paths = ["frame1.png", "frame2.png", "frame3.png"]
    seq = Sequence(
        sequence_id="seq3",
        camera_id="cam3",
        arm="hold",
        frame_paths=frame_paths,
    )
    assert len(seq) == 3


def test_sequence_len_with_empty_frame_paths():
    """Sequence.__len__ works with empty frame_paths."""
    seq = Sequence(
        sequence_id="seq4",
        camera_id="cam4",
        arm="micro",
        frame_paths=[],
    )
    assert len(seq) == 0


def test_sequence_len_with_single_frame():
    """Sequence.__len__ works with a single frame."""
    seq = Sequence(
        sequence_id="seq5",
        camera_id="cam5",
        arm="dev",
        frame_paths=["single.png"],
    )
    assert len(seq) == 1


# --- Stabilised ----------------------------------------------------------------


def test_stabilised_shift_px_property():
    """Stabilised.shift_px computes the Euclidean distance."""
    stab = Stabilised(
        frame=np.zeros((480, 640), dtype=np.uint8),
        dx=3.0,
        dy=4.0,
        response=0.9,
        method="phase",
    )
    assert stab.shift_px == pytest.approx(5.0)


def test_stabilised_shift_px_zero():
    """Stabilised.shift_px is 0 when no motion."""
    stab = Stabilised(
        frame=np.zeros((480, 640), dtype=np.uint8),
        dx=0.0,
        dy=0.0,
        response=0.85,
        method="passthrough",
    )
    assert stab.shift_px == pytest.approx(0.0)


# --- Component -----------------------------------------------------------------


def test_component_bottom_property():
    """Component.bottom returns y + h."""
    comp = Component(x=10, y=20, w=100, h=50, area=5000, cx=60.0, cy=45.0)
    assert comp.bottom == 70


def test_component_top_property():
    """Component.top returns y."""
    comp = Component(x=10, y=20, w=100, h=50, area=5000, cx=60.0, cy=45.0)
    assert comp.top == 20


def test_component_bbox_method():
    """Component.bbox returns (x, y, w, h)."""
    comp = Component(x=10, y=20, w=100, h=50, area=5000, cx=60.0, cy=45.0)
    assert comp.bbox() == (10, 20, 100, 50)


# --- Track ---------------------------------------------------------------------


def test_track_add_and_last():
    """Track.add appends and Track.last retrieves the last component."""
    track = Track(track_id=1)
    comp1 = Component(x=10, y=20, w=100, h=50, area=5000, cx=60.0, cy=45.0)
    comp2 = Component(x=11, y=21, w=100, h=50, area=5000, cx=61.0, cy=46.0)

    track.add(0, comp1, window=10)
    assert track.last == comp1
    assert len(track) == 1

    track.add(1, comp2, window=10)
    assert track.last == comp2
    assert len(track) == 2


def test_track_window_trimming():
    """Track.add removes oldest entries when window is exceeded."""
    track = Track(track_id=2)
    comp = Component(x=10, y=20, w=100, h=50, area=5000, cx=60.0, cy=45.0)
    window = 3

    for i in range(5):
        track.add(i, comp, window=window)

    assert len(track) == window
    assert track.frames == [2, 3, 4]


# --- Stats ---------------------------------------------------------------------


def test_stats_as_tuple():
    """Stats.as_tuple returns (s1, s2, s3, s4)."""
    stats = Stats(s1=0.9, s2=0.8, s3=0.7, s4=0.6, raw_s3=0.65)
    assert stats.as_tuple() == (0.9, 0.8, 0.7, 0.6)


def test_stats_raw_s3_default():
    """Stats.raw_s3 defaults to 0.0."""
    stats = Stats(s1=0.5, s2=0.5, s3=0.5, s4=0.5)
    assert stats.raw_s3 == 0.0


# --- CameraScore ---------------------------------------------------------------


def test_camera_score_creation():
    """CameraScore can be created with all fields."""
    comp = Component(x=10, y=20, w=100, h=50, area=5000, cx=60.0, cy=45.0)
    stats = Stats(s1=0.9, s2=0.8, s3=0.7, s4=0.6)
    score = CameraScore(
        camera_id="cam1",
        frame_index=42,
        hazard=0.75,
        stats=stats,
        component=comp,
        escalated=True,
        escalation_score=0.5,
    )
    assert score.camera_id == "cam1"
    assert score.hazard == 0.75
    assert score.escalated is True


def test_camera_score_defaults():
    """CameraScore has proper defaults for escalation fields."""
    stats = Stats(s1=0.5, s2=0.5, s3=0.5, s4=0.5)
    score = CameraScore(
        camera_id="cam2",
        frame_index=10,
        hazard=0.5,
        stats=stats,
        component=None,
    )
    assert score.escalated is False
    assert score.escalation_score == 0.0
    assert score.stabilisation is None


# --- WatchEntry ----------------------------------------------------------------


def test_watch_entry_to_json():
    """WatchEntry.to_json returns a dict with all fields rounded."""
    entry = WatchEntry(
        rank=1,
        camera_id="cam1",
        hazard=0.7654321,
        s1=0.1234567,
        s2=0.2345678,
        s3=0.3456789,
        s4=0.4567890,
        escalated=True,
        alerted=False,
        budget_spent=1,
        evidence_url="http://example.com/evidence.jpg",
    )
    js = entry.to_json()
    assert js["rank"] == 1
    assert js["camera_id"] == "cam1"
    assert js["hazard"] == pytest.approx(0.7654, abs=0.0001)
    assert js["s1"] == pytest.approx(0.1235, abs=0.0001)
    assert js["escalated"] is True
    assert js["evidence_url"] == "http://example.com/evidence.jpg"


def test_watch_entry_to_json_with_none_evidence():
    """WatchEntry.to_json handles None evidence_url."""
    entry = WatchEntry(
        rank=2,
        camera_id="cam2",
        hazard=0.5,
        s1=0.1,
        s2=0.2,
        s3=0.3,
        s4=0.4,
        escalated=False,
        alerted=True,
        budget_spent=0,
        evidence_url=None,
    )
    js = entry.to_json()
    assert js["evidence_url"] is None

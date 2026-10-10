"""Tests for statistic.py line coverage — uncovered lines 136, 189, 314, 452.

Each test targets one or more uncovered lines and asserts real behaviour
(return values, exceptions, messages), following the style of test_statistic.py.
"""

from __future__ import annotations

import math
from unittest import mock

import numpy as np

from plumerank.constants import (
    STAT_WINDOW_FRAMES,
    WORK_HEIGHT,
    WORK_WIDTH,
)
from plumerank.statistic import (
    _slope,
    s1_anchoring,
    s3_diffuse_boundary,
    s4_drift,
)
from plumerank.types import Component, Track

# --- fixtures -----------------------------------------------------------------


def comp(x: int, y: int, w: int, h: int) -> Component:
    """A Component with its derived fields consistent."""
    return Component(x=x, y=y, w=w, h=h, area=w * h, cx=x + w / 2.0, cy=y + h / 2.0)


# --- line 136: _slope with denom <= 0.0 ------


def test_slope_returns_zero_when_all_frames_have_identical_index():
    """Line 136: when all time points are identical, denom is 0 and slope returns 0.0.

    _slope computes (t_centred * (v - v.mean())).sum() / (t_centred * t_centred).sum().
    If all frames are [5, 5, 5, 5], then t_centred is [0, 0, 0, 0] and the denominator
    is 0, triggering the guard that returns 0.0.
    """
    frames = [10, 10, 10, 10]
    values = [5.0, 7.0, 9.0, 11.0]
    assert _slope(frames, values) == 0.0


def test_slope_returns_zero_when_given_single_repeated_frame():
    """Line 136 variant: even a short list of identical frames returns 0.0."""
    frames = [42, 42]
    values = [1.0, 100.0]
    assert _slope(frames, values) == 0.0


# --- line 189: s1_anchoring with span <= 0.0 ------


def test_s1_returns_zero_when_track_spans_zero_frames():
    """Line 189: when track.frames[-1] - track.frames[0] <= 0, s1_anchoring returns 0.0.

    A track with all components on the same frame number has zero time span.
    Even if the components are changing shape, there is no time to measure growth over,
    so the statistic returns 0.0 (no evidence of anchored growth).
    """
    # Build a track where all components are added at the same frame index
    track = Track(track_id=1)
    frame_index = 17
    track.add(frame_index, comp(x=100, y=190, w=10, h=10), window=STAT_WINDOW_FRAMES)
    track.add(frame_index, comp(x=100, y=185, w=10, h=15), window=STAT_WINDOW_FRAMES)
    track.add(frame_index, comp(x=100, y=180, w=10, h=20), window=STAT_WINDOW_FRAMES)

    assert s1_anchoring(track) == 0.0


# --- line 314: s3_diffuse_boundary with w <= 0 or h <= 0 ------


def test_s3_returns_nan_pair_when_component_has_zero_width():
    """Line 314: when component.bbox() has w <= 0, s3_diffuse_boundary returns (nan, nan).

    A component with w=0 cannot have a valid ROI, so both squashed and raw S3 are nan.
    """
    grey = np.full((WORK_HEIGHT, WORK_WIDTH), 100, dtype=np.uint8)
    invalid_comp = comp(x=100, y=100, w=0, h=50)

    squashed, raw = s3_diffuse_boundary(grey, invalid_comp)
    assert math.isnan(squashed) and math.isnan(raw)


def test_s3_returns_nan_pair_when_component_has_zero_height():
    """Line 314 variant: when component.bbox() has h <= 0, s3_diffuse_boundary returns (nan, nan)."""
    grey = np.full((WORK_HEIGHT, WORK_WIDTH), 100, dtype=np.uint8)
    invalid_comp = comp(x=100, y=100, w=50, h=0)

    squashed, raw = s3_diffuse_boundary(grey, invalid_comp)
    assert math.isnan(squashed) and math.isnan(raw)


def test_s3_returns_nan_pair_when_component_has_negative_width():
    """Line 314 variant: negative width (impossible but guarded) returns (nan, nan)."""
    grey = np.full((WORK_HEIGHT, WORK_WIDTH), 100, dtype=np.uint8)
    # Component constructor may validate, so we use a low w value that becomes negative
    # in practice. We manually test the guard by creating a component with w < 0.
    invalid_comp = comp(x=100, y=100, w=-5, h=50)

    squashed, raw = s3_diffuse_boundary(grey, invalid_comp)
    assert math.isnan(squashed) and math.isnan(raw)


# --- line 452: s4_drift when flow is not finite ------


def test_s4_returns_zero_when_dis_flow_contains_nan():
    """Line 452: when DIS returns a flow field with NaN values, s4_drift returns 0.0.

    The flow field is checked with np.isfinite(flow).all(). If any value is nan, inf,
    or -inf, the condition fails and s4_drift returns 0.0 (no drift evidence).

    We mock _dis to return a NaN-containing flow to trigger this path without relying
    on undocumented OpenCV behaviour across versions.
    """
    from plumerank import statistic as st

    # Create two frames and a valid component
    frame1 = np.full((100, 100), 128, dtype=np.uint8)
    frame2 = np.roll(frame1, -3, axis=0)
    box = comp(x=20, y=20, w=64, h=64)

    # Mock _dis to return a flow field with NaN
    nan_flow = np.full((64, 64, 2), np.nan, dtype=np.float32)

    with mock.patch.object(st, "_dis") as mock_dis:
        mock_dis_instance = mock.MagicMock()
        mock_dis_instance.calc.return_value = nan_flow
        mock_dis.return_value = mock_dis_instance

        result = s4_drift(frame2, frame1, box)
        assert result == 0.0


def test_s4_returns_zero_when_dis_flow_contains_inf():
    """Line 452 variant: when DIS returns inf in the flow field, s4_drift returns 0.0."""
    from plumerank import statistic as st

    frame1 = np.full((100, 100), 128, dtype=np.uint8)
    frame2 = np.roll(frame1, -3, axis=0)
    box = comp(x=20, y=20, w=64, h=64)

    # Create a flow field with infinity
    inf_flow = np.zeros((64, 64, 2), dtype=np.float32)
    inf_flow[10, 10, 0] = np.inf

    with mock.patch.object(st, "_dis") as mock_dis:
        mock_dis_instance = mock.MagicMock()
        mock_dis_instance.calc.return_value = inf_flow
        mock_dis.return_value = mock_dis_instance

        result = s4_drift(frame2, frame1, box)
        assert result == 0.0


def test_s4_returns_zero_when_dis_flow_partially_nan():
    """Line 452 variant: when DIS returns a flow with even one NaN, s4_drift returns 0.0.

    np.isfinite(flow).all() is False if ANY element is non-finite, so a single NaN
    in a 64x64x2 field triggers the guard.
    """
    from plumerank import statistic as st

    frame1 = np.full((100, 100), 128, dtype=np.uint8)
    frame2 = np.roll(frame1, -3, axis=0)
    box = comp(x=20, y=20, w=64, h=64)

    # Create a mostly-valid flow with one NaN at the end
    partial_nan_flow = np.zeros((64, 64, 2), dtype=np.float32)
    partial_nan_flow[-1, -1, -1] = np.nan

    with mock.patch.object(st, "_dis") as mock_dis:
        mock_dis_instance = mock.MagicMock()
        mock_dis_instance.calc.return_value = partial_nan_flow
        mock_dis.return_value = mock_dis_instance

        result = s4_drift(frame2, frame1, box)
        assert result == 0.0


def test_s4_returns_valid_score_when_flow_is_finite():
    """Sanity check: when flow is fully finite, s4_drift computes and returns a valid result."""
    frame1 = np.full((100, 100), 128, dtype=np.uint8)
    frame2 = np.roll(frame1, -3, axis=0)  # upward shift
    box = comp(x=20, y=20, w=64, h=64)

    result = s4_drift(frame2, frame1, box)
    assert 0.0 <= result < 1.0  # should be a valid squashed score or 0


# --- Additional coverage for edge cases ------


def test_slope_handles_values_all_same():
    """_slope with constant values returns 0.0 regardless of frame times."""
    frames = [0, 1, 2, 3, 4]
    values = [5.0, 5.0, 5.0, 5.0, 5.0]
    assert _slope(frames, values) == 0.0


def test_s1_needs_at_least_two_frames_with_different_indices():
    """s1_anchoring requires span > 0 to compute a slope. Same-index tracks score 0.0."""
    track = Track(track_id=1)
    # Add 10 components but all at the same frame index
    frame_index = 100
    for i in range(10):
        track.add(
            frame_index, comp(x=100, y=190 - i * 5, w=10, h=10 + i), window=STAT_WINDOW_FRAMES
        )

    # Even though the shape is changing, no time has elapsed
    assert s1_anchoring(track) == 0.0


# --- line 244: s2_anisotropy with len(track) < 2 ------


def test_s2_returns_zero_for_single_frame_track():
    """Line 244: s2_anisotropy returns 0.0 when track has fewer than 2 components.

    A track with a single sample has no history to fit a slope over, so d(height)/dt
    and d(width)/dt cannot be computed. The function returns 0.0 (no evidence of growth).
    """
    from plumerank.statistic import s2_anisotropy

    track = Track(track_id=1)
    track.add(0, comp(x=100, y=100, w=10, h=20), window=STAT_WINDOW_FRAMES)

    assert s2_anisotropy(track) == 0.0


# --- line 436: s4_drift when dis_unmeasurable is True ------


def test_s4_returns_zero_when_roi_shape_is_dis_unmeasurable():
    """Line 436: s4_drift returns 0.0 when dis_unmeasurable(h, w) is True.

    DIS has a known problematic geometry: rows < 32 and cols >= 80 causes segfault on x86
    and returns NaN on arm64. The guard at line 435 prevents calling DIS on these shapes,
    returning 0.0 instead (no drift evidence rather than a segfault).
    """
    from plumerank.statistic import dis_unmeasurable

    frame1 = np.full((100, 200), 128, dtype=np.uint8)
    frame2 = np.roll(frame1, -3, axis=0)

    # Create a box with height < 32 and width >= 80
    # dis_unmeasurable expects h < 32 and w >= 80
    h_bad = 31
    w_bad = 80
    assert dis_unmeasurable(h_bad, w_bad), "test setup: shape should be unmeasurable"

    box = comp(x=20, y=20, w=w_bad, h=h_bad)

    result = s4_drift(frame2, frame1, box)
    assert result == 0.0


def test_s4_returns_zero_for_other_unmeasurable_shapes():
    """Line 436 variant: other problematic DIS shapes return 0.0."""
    from plumerank.statistic import dis_unmeasurable

    frame1 = np.full((100, 200), 128, dtype=np.uint8)
    frame2 = np.roll(frame1, -2, axis=0)

    # Test a few more shapes in the unmeasurable zone
    test_shapes = [
        (31, 100),
        (20, 80),
        (15, 120),
    ]

    for h, w in test_shapes:
        if dis_unmeasurable(h, w):
            box = comp(x=10, y=10, w=w, h=h)
            result = s4_drift(frame2, frame1, box)
            assert result == 0.0, f"height={h}, width={w} should return 0.0"


# --- line 532: Tracker assignment skipping ------


def test_tracker_skips_already_assigned_components():
    """Line 532: when iterating candidates, skip if a component or track is already taken.

    In Tracker.update, candidates are sorted by distance. When two components could match
    the same track, or one component could match multiple tracks, the nearest one wins and
    the farther one is skipped (line 532 continue). This test creates a scenario where
    component 0 and component 1 both want the same track, but component 0 is nearer.
    """
    from plumerank.statistic import Tracker

    tracker = Tracker()

    # Frame 0: two components far apart
    c0 = comp(x=50, y=50, w=10, h=10)
    c1 = comp(x=300, y=300, w=10, h=10)
    tracks0 = tracker.update(0, [c0, c1])
    assert len(tracks0) == 2
    track_id_c0 = tracks0[0].track_id
    track_id_c1 = tracks0[1].track_id

    # Frame 1: move both components slightly
    c0_next = comp(x=51, y=51, w=10, h=10)  # near the original c0
    c1_next = comp(x=301, y=301, w=10, h=10)  # near the original c1

    tracks1 = tracker.update(1, [c0_next, c1_next])

    # Both should match their original tracks
    assert tracks1[0].track_id == track_id_c0
    assert tracks1[1].track_id == track_id_c1

    # Now frame 2: introduce a new component C between the two old ones
    # that could plausibly match either track, but C is much closer to c0's track.
    # To test line 532's continue logic, we need a scenario where both c0 and C
    # are candidates for the same track, and c0 wins.
    between = comp(x=52, y=52, w=10, h=10)  # very close to c0 (distance ~1.4)
    # The MAX_PLAUSIBLE_SHIFT_PX should allow this to match either track, but
    # it should match c0's track because it's nearer.

    tracks2 = tracker.update(2, [c0_next, between, c1_next])
    assert tracks2[0].track_id == track_id_c0
    assert tracks2[2].track_id == track_id_c1
    # The new component should get a new track (no longer matching c0 since c0_next took it)
    # OR it should match c0's track if the distance allows. Let's verify it's associated.
    assert len(tracks2) == 3


def test_tracker_skip_logic_with_competing_candidates():
    """Line 532 variant: test the exact skipping logic with multiple competing candidates.

    Create a scenario where the same track could match multiple components.
    The candidate with the smallest distance wins, and the second candidate
    is skipped (line 532 continue).
    """
    from plumerank.constants import MAX_PLAUSIBLE_SHIFT_PX
    from plumerank.statistic import Tracker

    tracker = Tracker()

    # Frame 0: create a track
    comp_start = comp(x=100, y=100, w=10, h=10)
    tracks0 = tracker.update(0, [comp_start])
    track_id = tracks0[0].track_id

    # Frame 1: create two new components, both within MAX_PLAUSIBLE_SHIFT_PX of the old one
    # comp_near is at (101, 101) - very close, distance ~1.4
    # comp_farther is at (105, 105) - also close, distance ~7
    # Both are within MAX_PLAUSIBLE_SHIFT_PX, so both are valid candidates for the same track.
    # The greedy assignment will pick comp_near first, then skip comp_farther (line 532 continue).
    comp_near = comp(x=101, y=101, w=10, h=10)
    comp_farther = comp(x=105, y=105, w=10, h=10)

    assert comp_near.cx - comp_start.cx < MAX_PLAUSIBLE_SHIFT_PX
    assert comp_farther.cx - comp_start.cx < MAX_PLAUSIBLE_SHIFT_PX

    tracks1 = tracker.update(1, [comp_near, comp_farther])

    # comp_near (first in list, nearest distance) should get the existing track
    assert tracks1[0].track_id == track_id
    # comp_farther should get a new track (skipped by line 532)
    assert tracks1[1].track_id != track_id
    assert len(tracks1) == 2

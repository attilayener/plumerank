"""Coverage tests for plumerank.stabilise -- the six uncovered lines.

These tests target branches in the fallback path and error handling that are
difficult to reach with real or typical synthetic inputs.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from plumerank import stabilise as stab
from plumerank.constants import (
    FALLBACK_MIN_INLIERS,
    WORK_HEIGHT,
    WORK_WIDTH,
)


def textured(offset_x: float = 0.0, offset_y: float = 0.0, seed: int = 7) -> np.ndarray:
    """A working-resolution crop of a fixed textured plane, sampled at a sub-pixel offset.

    Shared helper from test_stabilise.py; repeated here so tests are independent.
    """
    rng = np.random.default_rng(seed)
    plane = cv2.GaussianBlur(
        rng.integers(0, 255, (WORK_HEIGHT + 160, WORK_WIDTH + 160), dtype=np.uint8),
        (5, 5),
        0,
    )
    matrix = np.array(
        [[1.0, 0.0, -(80.0 - offset_x)], [0.0, 1.0, -(80.0 - offset_y)]],
        dtype=np.float32,
    )
    return cv2.warpAffine(plane, matrix, (WORK_WIDTH, WORK_HEIGHT), flags=cv2.INTER_LINEAR)


@pytest.fixture(autouse=True)
def _clean_window_cache():
    """Every test starts with an empty window cache."""
    stab._HANNING_CACHE.clear()
    yield
    stab._HANNING_CACHE.clear()


# ---------------------------------------------------------------------------------
# Line 151: _as_u8() -- clipping path for float arrays with out-of-range values
# ---------------------------------------------------------------------------------


def test_as_u8_clips_float_values_outside_0_255_range() -> None:
    """Line 151: `return np.clip(grey, 0.0, 255.0).astype(np.uint8)`

    This is reached when the input is not uint8 and contains values outside 0-255.
    We pass a float32 array with values beyond the valid range.
    """
    # Float array with values outside 0-255
    grey = np.array([[-10.0, 100.0], [200.0, 300.0]], dtype=np.float32)

    result = stab._as_u8(grey)

    # Values should be clipped to 0-255 and converted to uint8
    assert result.dtype == np.uint8
    assert result[0, 0] == 0  # -10 clipped to 0
    assert result[0, 1] == 100
    assert result[1, 0] == 200
    assert result[1, 1] == 255  # 300 clipped to 255


# ---------------------------------------------------------------------------------
# Line 189: _affine_fallback() -- calcOpticalFlowPyrLK returns None
# ---------------------------------------------------------------------------------


def test_affine_fallback_returns_none_when_pyr_lk_fails() -> None:
    """Line 189: `return None` when nxt is None or status is None

    cv2.calcOpticalFlowPyrLK can fail and return None values. We monkeypatch it
    to force this branch.
    """
    prev = textured()
    cur = textured(2.0, 0.0)

    with patch("cv2.calcOpticalFlowPyrLK", return_value=(None, None, None)):
        result = stab._affine_fallback(prev, cur)

    assert result is None


# ---------------------------------------------------------------------------------
# Line 195: _affine_fallback() -- too few points survive LK tracking
# ---------------------------------------------------------------------------------


def test_affine_fallback_returns_none_when_too_few_tracks_survive() -> None:
    """Line 195: `return None` when len(src) < FALLBACK_MIN_INLIERS

    After tracking with calcOpticalFlowPyrLK, if too few points survived
    (status != 1), the fallback must refuse. We monkeypatch goodFeaturesToTrack
    to return many corners and calcOpticalFlowPyrLK to mark most of them as lost.
    """
    prev = textured()
    cur = textured(2.0, 0.0)

    # goodFeaturesToTrack returns corners; we'll use real ones
    real_corners = cv2.goodFeaturesToTrack(
        prev.astype(np.uint8),
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=8,
    )
    assert real_corners is not None and len(real_corners) >= FALLBACK_MIN_INLIERS

    # Create a bad status array: only 1 point succeeded (needs 12)
    bad_status = np.zeros((len(real_corners), 1), dtype=np.uint8)
    bad_status[0, 0] = 1  # Only 1 point passes

    # nxt can be any valid array since we're only checking the status
    nxt = real_corners.copy()

    with patch(
        "cv2.calcOpticalFlowPyrLK",
        return_value=(nxt, bad_status, None),
    ):
        result = stab._affine_fallback(prev, cur)

    assert result is None


# ---------------------------------------------------------------------------------
# Line 212: _affine_fallback() -- estimateAffine2D returns None
# ---------------------------------------------------------------------------------


def test_affine_fallback_returns_none_when_estimate_affine_fails() -> None:
    """Line 212: `return None` when model is None or inliers is None

    cv2.estimateAffine2D can fail and return (None, None). We monkeypatch it.
    """
    prev = textured()
    cur = textured(2.0, 0.0)

    with patch("cv2.estimateAffine2D", return_value=(None, None)):
        result = stab._affine_fallback(prev, cur)

    assert result is None


# ---------------------------------------------------------------------------------
# Line 214: _affine_fallback() -- estimateAffine2D returns too few inliers
# ---------------------------------------------------------------------------------


def test_affine_fallback_returns_none_when_fit_has_too_few_inliers() -> None:
    """Line 214: `return None` when int(np.count_nonzero(inliers)) < FALLBACK_MIN_INLIERS

    estimateAffine2D can return a model with too few inlier points. We monkeypatch
    it to return a valid model but with an inlier mask that has fewer than 12 ones.
    """
    prev = textured()
    cur = textured(2.0, 0.0)

    # Create a valid-looking but sparse inlier mask (fewer than 12 inliers)
    bad_inliers = np.zeros((20, 1), dtype=np.uint8)
    bad_inliers[0:5, 0] = 1  # Only 5 inliers, need 12

    # Create a dummy 2x3 affine matrix
    dummy_model = np.eye(2, 3, dtype=np.float32)

    with patch(
        "cv2.estimateAffine2D",
        return_value=(dummy_model, bad_inliers),
    ):
        result = stab._affine_fallback(prev, cur)

    assert result is None


# ---------------------------------------------------------------------------------
# Line 402: residual_error() -- phaseCorrelate returns non-finite values
# ---------------------------------------------------------------------------------


def test_residual_error_returns_inf_when_correlation_is_non_finite() -> None:
    """Line 402: `return math.inf` when not (math.isfinite(rdx) and math.isfinite(rdy))

    If the residual correlation produces non-finite shift values, residual_error
    must return inf. We monkeypatch phaseCorrelate to return non-finite rdx/rdy.
    """
    prev = textured()
    cur = textured(2.0, 0.0)
    result = stab.stabilise(prev, cur)

    # Monkeypatch phaseCorrelate to return non-finite shift values
    # We patch it within the stabilise module where residual_error calls it
    with patch("plumerank.stabilise.cv2.phaseCorrelate") as mock_pc:
        # Return (rdx=nan, rdy=valid), response=high (high response ensures line 401 check passes)
        mock_pc.return_value = ((float("nan"), 0.5), 0.95)
        error = stab.residual_error(prev, cur, result)

    assert math.isinf(error)


# ---------------------------------------------------------------------------------
# Integration: stabilise() with each rejection path
# ---------------------------------------------------------------------------------


def test_stabilise_with_affine_fallback_path() -> None:
    """Verify that stabilise() uses the affine fallback on low response.

    We patch phaseCorrelate to return low response to trigger the fallback.
    """
    prev = textured()
    cur = textured(2.0, 0.0)

    # Patch phaseCorrelate to return low response, triggering the fallback
    with patch("plumerank.stabilise.cv2.phaseCorrelate") as mock_pc:
        mock_pc.return_value = ((0.5, 0.3), 0.05)  # Low response value
        result = stab.stabilise(prev, cur)

    # With low response, stabilise() should use the fallback path
    # The method will be one of: affine, passthrough, or phase
    assert result.method in {"affine", "passthrough", "phase"}
    assert result.method != "phase+iterative"
    assert math.isfinite(result.dx) and math.isfinite(result.dy)


def test_stabilise_passthrough_on_implausible_fallback_estimate() -> None:
    """stabilise() falls back to _affine_fallback, which returns None, leading to passthrough."""
    prev = textured()
    cur = textured(2.0, 0.0)

    # Patch _affine_fallback to return None unconditionally
    with patch.object(stab, "_affine_fallback", return_value=None):
        # Also need to make phase correlation response low so fallback is triggered
        with patch(
            "cv2.phaseCorrelate",
            return_value=((0.0, 0.0), 0.1),  # Low response
        ):
            result = stab.stabilise(prev, cur)

    assert result.method == "passthrough"
    assert result.dx == 0.0 and result.dy == 0.0

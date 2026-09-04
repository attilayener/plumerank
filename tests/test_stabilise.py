"""Tests for plumerank.stabilise.

The four things worth asserting about a stabiliser, in order of how much damage getting
them wrong would do:

1. When there IS a shift and the scene has texture, it is recovered to well under a
   pixel -- the contract's bar is 0.5 px on +/-3.5 px, and the phase path clears it by
   two orders of magnitude.
2. When there is NOTHING to correlate, no confident number comes out. This is the
   expensive failure: a fabricated warp manufactures foreground along every high-contrast
   edge, and S1's base-anchoring statistic lives on exactly those edges.
3. The Hanning window is built once per shape. Not a correctness property; the
   per-frame classical budget is 30 ms and the window is 0.04 ms of it, but a cache that
   silently stops caching is the kind of regression nobody notices until the wall is
   running 40 cameras.
4. It works on real HPWREN frames, including a pair with genuine camera drift.
"""

from __future__ import annotations

import math
import pathlib

import cv2
import numpy as np
import pytest

from plumerank import stabilise as stab
from plumerank.constants import (
    MAX_PLAUSIBLE_SHIFT_PX,
    PHASE_RESPONSE_MIN,
    WORK_HEIGHT,
    WORK_WIDTH,
)
from plumerank.types import Stabilised

FRAMES = pathlib.Path(__file__).resolve().parents[1] / "spike" / "frames"

#: A pair of real consecutive frames from the same HPWREN camera (Palisades Fire,
#: 69bravo-e-mobo-c) that carries about 4.6 px of genuine camera drift. Chosen because
#: most of the corpus is nearly still, and a stabiliser that returned 0.0 unconditionally
#: would pass a test written on any of those.
DRIFT_PAIR = ("1736275741_p01500.jpg", "1736276342_p02101.jpg")

#: A near-static pair on a different camera (Del Mar Fire, tdlln-n-mobo-c), phase
#: response ~0.96. The other half of the same question: a stabiliser that always found
#: motion would pass the drift test and fail this one.
STILL_PAIR = ("1719350441_p00900.jpg", "1719351041_p01500.jpg")


# ---------------------------------------------------------------------------------
# Fixtures -- all synthetic images are seeded, because a stabiliser that passes 90% of
# the time is a stabiliser whose failures nobody will ever reproduce.
# ---------------------------------------------------------------------------------


def textured(offset_x: float = 0.0, offset_y: float = 0.0, seed: int = 7) -> np.ndarray:
    """A working-resolution crop of a fixed textured plane, sampled at a sub-pixel offset.

    Cropping out of an oversized plane rather than warping the crop itself keeps the
    border free of replicated or blank pixels, so the recovered shift is not being scored
    against an image the test itself damaged.
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
    return cv2.warpAffine(
        plane, matrix, (WORK_WIDTH, WORK_HEIGHT), flags=cv2.INTER_LINEAR
    )


def sky(seed: int, sigma: float = 1.5) -> np.ndarray:
    """A featureless dawn sky: constant grey plus faint sensor noise, nothing else.

    This is the input phase correlation has no answer for, and the input the whole
    fallback branch exists to survive.
    """
    rng = np.random.default_rng(seed)
    return np.clip(
        128.0 + rng.normal(0.0, sigma, (WORK_HEIGHT, WORK_WIDTH)), 0.0, 255.0
    ).astype(np.uint8)


def real(name: str) -> np.ndarray:
    """One HPWREN JPEG at working resolution, greyscale. INTER_AREA because the
    downscale is 6.4x and any other interpolation aliases the ridgeline into texture the
    camera never saw."""
    path = FRAMES / name
    if not path.exists():  # pragma: no cover - the corpus is committed
        pytest.skip(f"missing real frame {path}")
    full = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert full is not None, f"unreadable JPEG: {path}"
    return cv2.resize(full, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA)


@pytest.fixture(autouse=True)
def _clean_window_cache():
    """Every test starts with an empty window cache so the caching tests measure the
    cache and not the order pytest happened to run in."""
    stab._HANNING_CACHE.clear()
    yield
    stab._HANNING_CACHE.clear()


# ---------------------------------------------------------------------------------
# 1. Recovery -- the contract's bar
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shift_x, shift_y",
    [(3.5, 0.0), (-3.5, 0.0), (0.0, 3.5), (0.0, -3.5), (3.5, -3.5), (-3.5, 3.5)],
)
def test_synthetic_shift_recovered_to_under_half_a_pixel(
    shift_x: float, shift_y: float
) -> None:
    prev = textured()
    cur = textured(shift_x, shift_y)

    result = stab.stabilise(prev, cur)

    assert result.method == "phase+iterative"
    assert result.response >= PHASE_RESPONSE_MIN
    error = math.hypot(result.dx - shift_x, result.dy - shift_y)
    assert error < 0.5, f"recovered ({result.dx}, {result.dy}) for ({shift_x}, {shift_y})"


def test_recovered_shift_actually_registers_the_frame() -> None:
    """Recovering the number is not the job; removing the motion is.

    Asserted against the unstabilised residual rather than against a fixed tolerance, so
    the test says "it got better" in the units the estimator is graded in.
    """
    prev = textured()
    cur = textured(3.5, -2.0)

    result = stab.stabilise(prev, cur)
    untouched = Stabilised(
        frame=cur, dx=0.0, dy=0.0, response=0.0, method="passthrough"
    )

    after = stab.residual_error(prev, cur, result)
    before = stab.residual_error(prev, cur, untouched)
    assert before > 3.0
    assert after < 0.25
    assert after < before / 10.0


def test_sign_convention_positive_dx_means_the_scene_moved_right() -> None:
    """Getting this backwards warps by double the motion instead of undoing it, and the
    result still looks like a plausible number in the log."""
    prev = textured()
    cur = textured(4.0, 0.0)

    result = stab.stabilise(prev, cur)

    assert result.dx > 3.0
    assert abs(result.dy) < 0.5


def test_float32_input_is_accepted_and_the_output_keeps_its_dtype() -> None:
    prev = textured().astype(np.float32)
    cur = textured(3.5, 0.0).astype(np.float32)

    result = stab.stabilise(prev, cur)

    assert result.frame.dtype == np.float32
    assert abs(result.dx - 3.5) < 0.5


def test_the_callers_frames_are_not_modified() -> None:
    """cv2.phaseCorrelate applies its ``window`` argument by multiplying its inputs IN
    PLACE whenever the frame needs no DFT padding, and 480x270 needs none.

    A CV_32F caller therefore gets its own frames replaced by vignetted copies -- and,
    because ``stabilise`` warps ``cur_grey`` at the end, the frame handed downstream is
    that vignette: a black border tens of pixels wide that MOG2 reports as foreground
    along every edge. Nothing in the return value looks wrong, which is why this is
    asserted on the inputs.
    """
    prev = textured().astype(np.float32)
    cur = textured(3.5, -2.0).astype(np.float32)
    prev_before, cur_before = prev.copy(), cur.copy()

    result = stab.stabilise(prev, cur)

    assert np.array_equal(prev, prev_before), "stabilise modified prev_grey in place"
    assert np.array_equal(cur, cur_before), "stabilise modified cur_grey in place"

    stab.residual_error(prev, cur, result)
    assert np.array_equal(prev, prev_before), "residual_error modified prev_grey"
    assert np.array_equal(cur, cur_before), "residual_error modified cur_grey"


def test_the_returned_frame_is_the_registered_frame_and_not_a_windowed_one() -> None:
    """The output frame has to be ``cur_grey`` moved, with its brightness intact.

    A frame that has been multiplied by the Hann window still has the right shape, the
    right dtype and a plausible shift beside it; it is only the pixels that are ruined,
    and the ruin is worst exactly at the frame edge where the ridgeline sits.
    """
    prev = textured().astype(np.float32)
    cur = textured(3.5, -2.0).astype(np.float32)

    result = stab.stabilise(prev, cur)

    assert np.allclose(result.frame, stab.warp(cur, result.dx, result.dy))
    # The Hann window is ~0 on the first and last rows; an unwindowed frame is not.
    assert result.frame[0].mean() > 64.0
    assert result.frame[-1].mean() > 64.0
    assert abs(float(result.frame.mean()) - float(cur.mean())) < 8.0


# ---------------------------------------------------------------------------------
# 2. Nothing to correlate -- the expensive failure
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_featureless_sky_never_returns_a_confident_wrong_shift(seed: int) -> None:
    """On an empty sky the estimator may fall back, or refuse, or report a coarse peak it
    has labelled as coarse -- but it must never hand back a sub-pixel-refined number, and
    it must never hand back a large one. Both frames are pure noise around the same
    constant, so the honest answer is "no measurable motion"."""
    prev = sky(1000 + 2 * seed)
    cur = sky(1001 + 2 * seed)

    result = stab.stabilise(prev, cur)

    assert result.method in {"affine", "passthrough", "phase"}
    assert result.method != "phase+iterative"
    assert result.shift_px < 3.0, f"{result.method} claimed {result.shift_px} px of shake"


def test_featureless_sky_takes_the_fallback_across_a_seed_sweep() -> None:
    """The single branch has to actually fire, and fire on most of these inputs -- a
    fallback that never runs is a fallback nobody has tested."""
    methods = [
        stab.stabilise(sky(1000 + 2 * s), sky(1001 + 2 * s)).method for s in range(32)
    ]

    assert "phase+iterative" not in methods
    assert sum(m == "affine" for m in methods) >= 20
    assert all(m in {"affine", "passthrough", "phase"} for m in methods)


def test_a_constant_frame_yields_no_motion() -> None:
    """A flat grey field is the degenerate input: the cross-power spectrum is 0/0.

    Zero motion is the true answer here, so the assertion is on the number and on the
    pixels rather than on the label -- which of ``"phase"`` and ``"phase+iterative"``
    comes back is an artifact of how the refinement behaves on a signal that carries no
    information, and pinning it would pin the artifact instead of the requirement.
    """
    flat = np.full((WORK_HEIGHT, WORK_WIDTH), 128, dtype=np.uint8)

    result = stab.stabilise(flat, flat)

    assert result.method in {"phase", "phase+iterative"}
    assert result.shift_px < 0.5
    assert np.allclose(result.frame, flat)


def test_non_finite_pixels_end_in_a_passthrough_not_a_nan_shift() -> None:
    """A corrupt CV_32F frame propagates NaN through the DFT into the shift.

    ``warp`` would raise on it, three stages after the frame that caused it was read, so
    the refusal is made here and labelled: passthrough, zero shift, frame untouched.
    """
    prev = textured().astype(np.float32)
    cur = textured(3.5, 0.0).astype(np.float32)
    cur[10, 10] = np.nan

    result = stab.stabilise(prev, cur)

    assert result.method == "passthrough"
    assert result.dx == 0.0 and result.dy == 0.0


def test_the_fallback_chain_recovers_a_shift_on_textured_input() -> None:
    """goodFeaturesToTrack -> calcOpticalFlowPyrLK -> estimateAffine2D(USAC_MAGSAC)
    exercised directly, because on the sky inputs that reach it in production the right
    answer is zero and a broken chain would look identical to a working one."""
    estimate = stab._affine_fallback(textured(), textured(3.0, -2.0))

    assert estimate is not None
    assert math.hypot(estimate[0] - 3.0, estimate[1] + 2.0) < 0.5


def test_the_fallback_reports_the_shift_at_the_centre_not_at_the_origin() -> None:
    """A 6-DoF fit hides a rotation in its translation column, multiplied by the lever
    arm from the corner.

    This pair is a pure roll about the image centre: the honest answer is "the scene did
    not translate". Reading ``model[:, 2]`` instead returns about 4.8 px of phantom pan,
    which is inside MAX_PLAUSIBLE_SHIFT_PX and so would have been applied to the frame
    rather than rejected.
    """
    prev = textured()
    centre = ((WORK_WIDTH - 1) * 0.5, (WORK_HEIGHT - 1) * 0.5)
    rotation = cv2.getRotationMatrix2D(centre, 1.0, 1.0)
    cur = cv2.warpAffine(
        prev, rotation, (WORK_WIDTH, WORK_HEIGHT), borderMode=cv2.BORDER_REPLICATE
    )

    estimate = stab._affine_fallback(prev, cur)

    assert estimate is not None
    assert math.hypot(*estimate) < 1.0, f"phantom pan of {estimate} from a pure roll"


@pytest.mark.parametrize(
    "field",
    [
        np.full((WORK_HEIGHT, WORK_WIDTH), 128, dtype=np.uint8),
        np.tile(
            np.linspace(60, 200, WORK_HEIGHT, dtype=np.float32)[:, None],
            (1, WORK_WIDTH),
        ).astype(np.uint8),
    ],
)
def test_the_fallback_refuses_a_corner_free_field(field: np.ndarray) -> None:
    """``None``, not a best-effort number. A flat field and a pure ramp both have a
    singular structure tensor everywhere, so there is nothing to track and saying so is
    the only honest return value."""
    assert stab._affine_fallback(field, field) is None


def test_runaway_subpixel_refinement_is_rejected_and_labelled() -> None:
    """The failure mode _SUBPIXEL_AGREEMENT_PX exists for, pinned to a fixed pair.

    Raw ``phaseCorrelateIterative`` walks to ~83 px on this input while the coarse peak
    sits at 0.49 px. Without the agreement check that 83 px would be returned as a
    sub-pixel-accurate camera shake.
    """
    prev, cur = sky(1020), sky(1021)
    window = stab._hanning_window(prev.shape)
    raw_dx, raw_dy = cv2.phaseCorrelateIterative(
        prev.astype(np.float32) * window, cur.astype(np.float32) * window
    )
    assert math.hypot(raw_dx, raw_dy) > 10.0, "fixture no longer exercises the runaway"

    result = stab.stabilise(prev, cur)

    assert result.method == "phase"
    assert result.shift_px < 3.0


def test_implausible_shift_is_rejected_rather_than_applied() -> None:
    """60 px at 480x270 is a pan or a corrupt frame, not shake. The phase peak is strong
    and correct here -- the point is that a correct-but-implausible estimate is still
    refused, because the pipeline downstream has no model for a scene that jumped."""
    prev = textured()
    cur = textured(60.0, 0.0)

    result = stab.stabilise(prev, cur)

    assert result.method == "passthrough"
    assert result.dx == 0.0 and result.dy == 0.0
    assert result.response >= PHASE_RESPONSE_MIN
    assert np.array_equal(result.frame, cur)


def test_a_shift_just_inside_the_plausible_envelope_is_kept() -> None:
    """The other side of the same cut, so the test suite pins the threshold rather than
    just the rejection."""
    inside = MAX_PLAUSIBLE_SHIFT_PX - 10.0
    result = stab.stabilise(textured(), textured(inside, 0.0))

    assert result.method in {"phase", "phase+iterative"}
    assert abs(result.dx - inside) < 0.5


# ---------------------------------------------------------------------------------
# 3. The Hanning window is built once
# ---------------------------------------------------------------------------------


def test_hanning_window_is_cached_across_calls() -> None:
    prev, cur = textured(), textured(2.0, 0.0)

    stab.stabilise(prev, cur)
    first = stab._hanning_window(prev.shape)
    stab.stabilise(prev, cur)
    second = stab._hanning_window(prev.shape)

    assert list(stab._HANNING_CACHE) == [(WORK_HEIGHT, WORK_WIDTH)]
    assert first is second, "the window was rebuilt instead of being handed back"


def test_hanning_cache_keeps_one_entry_per_shape() -> None:
    """40 cameras at one resolution means one entry; a second resolution must not evict
    the first, or an alternating pair of shapes rebuilds on every frame."""
    stab.stabilise(textured(), textured(2.0, 0.0))
    small = np.zeros((64, 96), dtype=np.uint8)
    stab.stabilise(small, small)

    assert set(stab._HANNING_CACHE) == {(WORK_HEIGHT, WORK_WIDTH), (64, 96)}


def test_the_window_is_applied_exactly_once_to_both_estimators() -> None:
    """The coarse peak and its refinement must be measured on the same two arrays.

    ``phaseCorrelate`` windows its inputs internally when handed a ``window``;
    ``phaseCorrelateIterative`` has no window argument at all. Windowing for the second
    call on top of the first applies the Hann coefficients twice, and then the agreement
    check that gates ``"phase+iterative"`` is comparing two different images -- which is
    invisible in the output, because a squared Hann window is still a valid window.
    """
    prev, cur = real(STILL_PAIR[0]), real(STILL_PAIR[1])
    window = stab._hanning_window(prev.shape)
    prev_w = np.multiply(prev, window, dtype=np.float32)
    cur_w = np.multiply(cur, window, dtype=np.float32)

    result = stab.stabilise(prev, cur)
    once_dx, once_dy = cv2.phaseCorrelateIterative(prev_w, cur_w)
    twice = cv2.phaseCorrelateIterative(prev_w * window, cur_w * window)

    assert result.method == "phase+iterative"
    assert (result.dx, result.dy) == (pytest.approx(once_dx), pytest.approx(once_dy))
    assert math.hypot(once_dx - twice[0], once_dy - twice[1]) > 1e-6, (
        "double windowing no longer changes the answer, so this test proves nothing"
    )


def test_cached_window_is_opencvs_own_and_the_axes_are_not_swapped() -> None:
    """createHanningWindow takes (width, height). On a 480x270 frame the transposed call
    would still return an array of plausible values and every shift would be subtly
    wrong, so this compares against a freshly built window rather than a shape."""
    window = stab._hanning_window((WORK_HEIGHT, WORK_WIDTH))
    expected = cv2.createHanningWindow((WORK_WIDTH, WORK_HEIGHT), cv2.CV_32F)

    assert window.shape == (WORK_HEIGHT, WORK_WIDTH)
    assert np.allclose(window, expected)


# ---------------------------------------------------------------------------------
# 4. warp, residual_error, and the input contract
# ---------------------------------------------------------------------------------


def test_warp_undoes_the_shift_it_is_handed() -> None:
    """``warp`` takes the shift the scene underwent, not the translation to apply -- the
    negation lives inside so no call site has to get the sign right."""
    prev = textured()
    cur = textured(6.0, -4.0)

    registered = stab.warp(cur, 6.0, -4.0)

    # Compared away from the replicated border, which carries no information either way.
    inner = (slice(20, -20), slice(20, -20))
    def mad(a: np.ndarray) -> float:
        return float(np.mean(np.abs(a[inner].astype(float) - prev[inner].astype(float))))

    assert mad(registered) < mad(cur) / 4.0
    assert registered.shape == cur.shape and registered.dtype == cur.dtype


def test_warp_replicates_the_border_rather_than_filling_it_with_zeros() -> None:
    """The vacated strip must carry the edge pixel, not black.

    A zero band is a step of ~128 grey levels down one whole side of the frame, and MOG2
    reports it as foreground for the full length of that side -- a fabricated detection
    on every stabilised frame, in the literal sense.
    """
    frame = textured()

    registered = stab.warp(frame, -12.0, 0.0)

    band = registered[:, :12]
    assert band.min() > 0
    assert np.array_equal(band, np.repeat(registered[:, 12:13], 12, axis=1))


def test_warp_of_zero_returns_a_copy_not_the_original() -> None:
    """The passthrough path takes this branch on every rejected frame, and the caller
    owns the array it gets back."""
    frame = textured()

    out = stab.warp(frame, 0.0, 0.0)

    assert out is not frame
    assert np.array_equal(out, frame)


def test_warp_refuses_a_non_finite_shift() -> None:
    with pytest.raises(ValueError):
        stab.warp(textured(), float("nan"), 0.0)
    with pytest.raises(ValueError):
        stab.warp(textured(), 0.0, float("inf"))


def test_residual_error_of_an_uncorrelated_pair_is_infinite() -> None:
    """``inf`` rather than a large float, so metrics.percentiles cannot interpolate it
    into a plausible small number."""
    rng = np.random.default_rng(11)
    prev = cv2.GaussianBlur(
        rng.integers(0, 255, (WORK_HEIGHT, WORK_WIDTH), dtype=np.uint8), (9, 9), 0
    )
    cur = cv2.GaussianBlur(
        rng.integers(0, 255, (WORK_HEIGHT, WORK_WIDTH), dtype=np.uint8), (9, 9), 0
    )
    untouched = Stabilised(
        frame=cur, dx=0.0, dy=0.0, response=0.0, method="passthrough"
    )

    assert math.isinf(stab.residual_error(prev, cur, untouched))


def test_residual_error_reports_the_error_it_was_handed_in_pixels() -> None:
    """The units have to be pixels of leftover shift, not an arbitrary score.

    A deliberately wrong registration is fed in, 2 px off the truth, and the function has
    to come back with 2 px. Asserting this rather than "it is small after stabilise" is
    what stops the number from being a photometric score, or a self-fulfilling zero: on
    the phase paths ``residual_error`` grades an estimate its own instrument produced, so
    the only honest way to test it is with an estimate that instrument did not make.
    """
    prev = textured()
    cur = textured(5.0, 0.0)
    wrong = Stabilised(frame=cur, dx=3.0, dy=0.0, response=1.0, method="phase")

    assert stab.residual_error(prev, cur, wrong) == pytest.approx(2.0, abs=0.1)


def test_residual_error_of_a_passthrough_reports_the_whole_motion() -> None:
    prev = textured()
    cur = textured(5.0, 0.0)
    untouched = Stabilised(
        frame=cur, dx=0.0, dy=0.0, response=0.0, method="passthrough"
    )

    assert stab.residual_error(prev, cur, untouched) > 4.0


@pytest.mark.parametrize(
    "prev, cur",
    [
        (np.zeros((10, 10), np.uint8), np.zeros((12, 10), np.uint8)),
        (np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10, 3), np.uint8)),
        (np.zeros((1, 1), np.uint8), np.zeros((1, 1), np.uint8)),
    ],
)
def test_unusable_input_raises_instead_of_returning_a_number(
    prev: np.ndarray, cur: np.ndarray
) -> None:
    """A caller who hands this stage a colour frame has a bug, and finding it four
    stages later as a strange hazard score costs far more than finding it here."""
    with pytest.raises(ValueError):
        stab.stabilise(prev, cur)


def test_a_non_array_raises_type_error() -> None:
    with pytest.raises(TypeError):
        stab.stabilise([[0, 0], [0, 0]], np.zeros((2, 2), np.uint8))


# ---------------------------------------------------------------------------------
# 5. Real HPWREN frames
# ---------------------------------------------------------------------------------


def test_real_consecutive_frames_with_camera_drift_are_registered() -> None:
    """The smoke test that matters: two real frames from one HPWREN camera carrying
    genuine drift, and the residual measured before and after."""
    prev, cur = real(DRIFT_PAIR[0]), real(DRIFT_PAIR[1])

    result = stab.stabilise(prev, cur)
    untouched = Stabilised(
        frame=cur, dx=0.0, dy=0.0, response=0.0, method="passthrough"
    )
    before = stab.residual_error(prev, cur, untouched)
    after = stab.residual_error(prev, cur, result)

    assert result.method in {"phase", "phase+iterative"}
    assert 1.0 < result.shift_px < MAX_PLAUSIBLE_SHIFT_PX
    assert before > 3.0
    assert after < before / 4.0
    assert result.frame.shape == prev.shape and result.frame.dtype == prev.dtype


def test_real_still_frames_report_no_motion() -> None:
    """The same code on a camera that did not move. A stabiliser that always finds
    something would pass the drift test and fail this one."""
    prev, cur = real(STILL_PAIR[0]), real(STILL_PAIR[1])

    result = stab.stabilise(prev, cur)

    assert result.method == "phase+iterative"
    assert result.response > PHASE_RESPONSE_MIN
    assert result.shift_px < 1.0


def test_real_frame_recovers_an_injected_subpixel_shift() -> None:
    """+/-3.5 px on a real ridgeline rather than on synthetic texture -- the contract's
    bar is 0.5 px and the phase path clears it by two orders of magnitude on the actual
    imagery the system runs on."""
    prev = real(STILL_PAIR[0])
    matrix = np.array([[1.0, 0.0, 3.5], [0.0, 1.0, -3.5]], dtype=np.float32)
    cur = cv2.warpAffine(
        prev, matrix, (WORK_WIDTH, WORK_HEIGHT), borderMode=cv2.BORDER_REPLICATE
    )

    result = stab.stabilise(prev, cur)

    assert result.method == "phase+iterative"
    assert math.hypot(result.dx - 3.5, result.dy + 3.5) < 0.5

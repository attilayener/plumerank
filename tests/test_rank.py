"""Tests for the fusion, the solved threshold, the budget and the evidence crop.

Two of these are not really unit tests but assertions about the product's promises:
``test_emit_never_suppresses_a_spent_camera`` is spec.md section 9 safeguard 2, and
``test_solve_threshold_closed_form`` is the only reason to believe the published operating
point is the one the system actually runs at.
"""

from __future__ import annotations

import math
import pathlib

import cv2
import numpy as np
import pytest

from plumerank import rank
from plumerank.constants import (
    ALERTS_PER_CAMERA_PER_NIGHT,
    BASELINE_SD_FLOOR,
    BASELINE_WARMUP_FRAMES,
    BASELINE_WINDOW_FRAMES,
    BASELINE_Z_SCALE,
    FRAMES_PER_WINDOW,
    FUSION_BIAS,
    FUSION_W_ESCALATION,
    FUSION_W_S1,
    FUSION_W_S2,
    FUSION_W_S3,
    FUSION_W_S4,
    S4_ENABLED,
    THRESHOLD_SEARCH_HI,
    THRESHOLD_SEARCH_LO,
    THRESHOLD_SEARCH_TOL,
)
from plumerank.rank import (
    BudgetState,
    draw_evidence,
    emit,
    evidence_crop_box,
    fuse,
    realised_rate,
    solve_threshold,
)
from plumerank.types import CameraScore, Component, Stats, Track

SPIKE_FRAMES = pathlib.Path(__file__).resolve().parents[1] / "spike" / "frames"


def stats(s1: float = 0.0, s2: float = 0.0, s3: float = 0.0, s4: float = 0.0) -> Stats:
    return Stats(s1=s1, s2=s2, s3=s3, s4=s4, raw_s3=0.0)


def score(camera_id: str, hazard: float, **kw) -> CameraScore:
    return CameraScore(
        camera_id=camera_id,
        frame_index=kw.pop("frame_index", 0),
        hazard=hazard,
        stats=kw.pop("stats", stats(0.5, 0.5, 0.5, 0.5)),
        component=kw.pop("component", None),
        **kw,
    )


# --- fuse --------------------------------------------------------------------------


def test_fuse_is_the_documented_logistic():
    """Closed form, recomputed from constants.py rather than from a remembered value."""
    s = stats(0.9, 0.7, 0.6, 0.4)
    z = FUSION_BIAS + FUSION_W_S1 * 0.9 + FUSION_W_S2 * 0.7 + FUSION_W_S3 * 0.6
    if S4_ENABLED:
        z += FUSION_W_S4 * 0.4
    z += FUSION_W_ESCALATION * 0.3
    assert fuse(s, escalation_score=0.3) == pytest.approx(1.0 / (1.0 + math.exp(-z)), abs=1e-12)


def test_fuse_stays_inside_the_unit_interval_at_both_extremes():
    assert 0.0 < fuse(stats(0, 0, 0, 0)) < fuse(stats(1, 1, 1, 1), 1.0) < 1.0


def test_fuse_is_monotone_in_every_statistic():
    """Every weight is positive, so more of any evidence can only raise the hazard --
    the property that lets a row's chips explain its rank."""
    base = fuse(stats(0.2, 0.2, 0.2, 0.2))
    assert fuse(stats(0.9, 0.2, 0.2, 0.2)) > base
    assert fuse(stats(0.2, 0.9, 0.2, 0.2)) > base
    assert fuse(stats(0.2, 0.2, 0.9, 0.2)) > base
    assert fuse(stats(0.2, 0.2, 0.2, 0.2), escalation_score=0.9) > base


def test_s3_carries_the_most_weight():
    """The fog discriminator dominates by construction, not by accident."""
    only_s1 = fuse(stats(1.0, 0.0, 0.0, 0.0))
    only_s3 = fuse(stats(0.0, 0.0, 1.0, 0.0))
    assert only_s3 > only_s1


def test_fuse_raises_on_a_nan_statistic():
    """S3 returns nan when the component is too small to carry a boundary band. A NaN
    hazard would sort arbitrarily and scramble the watch list silently."""
    with pytest.raises(ValueError, match="non-finite"):
        fuse(stats(0.5, 0.5, float("nan"), 0.5))


def test_fuse_raises_on_an_unsquashed_statistic():
    with pytest.raises(ValueError, match="outside"):
        fuse(stats(0.5, 4.0, 0.5, 0.5))
    with pytest.raises(ValueError, match="outside"):
        fuse(stats(0.5, 0.5, 0.5, 0.5), escalation_score=-0.3)


# --- solve_threshold ---------------------------------------------------------------


def two_hot_windows(n_windows: int, frames: int = FRAMES_PER_WINDOW,
                    hi: float = 0.9, lo: float = 0.1) -> list[list[float]]:
    """Windows with exactly two high frames each -- so the answer is closed-form: any
    threshold in (lo, hi] realises exactly 2 alerts per window and nothing else does."""
    return [[hi, hi] + [lo] * (frames - 2) for _ in range(n_windows)]


def test_realised_rate_matches_hand_arithmetic():
    windows = two_hot_windows(5)
    assert realised_rate(windows, 0.5) == pytest.approx(2.0)
    assert realised_rate(windows, 0.95) == pytest.approx(0.0)
    assert realised_rate(windows, 0.0) == pytest.approx(float(FRAMES_PER_WINDOW))


def test_realised_rate_is_monotone_non_increasing_in_the_threshold():
    rng = np.random.default_rng(7)
    windows = [list(rng.random(FRAMES_PER_WINDOW)) for _ in range(12)]
    rates = [realised_rate(windows, t) for t in np.linspace(0.0, 1.0, 41)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_solve_threshold_closed_form():
    """Two alerts per window by construction; the solved threshold must be the infimum of
    the interval that realises them, to within the search tolerance."""
    windows = two_hot_windows(30)
    t = solve_threshold(windows, target=2.0)
    assert 0.1 < t <= 0.1 + THRESHOLD_SEARCH_TOL
    assert realised_rate(windows, t) == pytest.approx(2.0)


def test_solve_threshold_hits_the_project_operating_point_by_default():
    windows = two_hot_windows(30)
    assert realised_rate(windows, solve_threshold(windows)) == pytest.approx(2.0)


def test_solve_threshold_is_monotone_in_the_target():
    """A looser budget can never demand a stricter threshold."""
    rng = np.random.default_rng(11)
    windows = [list(rng.random(FRAMES_PER_WINDOW)) for _ in range(20)]
    thresholds = [solve_threshold(windows, target=x) for x in (0.5, 1.0, 2.0, 4.0, 8.0, 20.0)]
    assert all(a >= b for a, b in zip(thresholds, thresholds[1:]))


def test_solve_threshold_terminates_at_the_low_bound_when_the_budget_is_unreachably_generous():
    windows = two_hot_windows(4)
    assert solve_threshold(windows, target=1000.0) == THRESHOLD_SEARCH_LO


def test_solve_threshold_terminates_at_the_high_bound_when_the_target_is_unreachable():
    """A window whose hazard saturates at 1.0 alerts even at the strictest threshold, so
    a zero-alert budget cannot be met and the bound is returned rather than a pretence."""
    windows = [[1.0] + [0.0] * (FRAMES_PER_WINDOW - 1) for _ in range(6)]
    t = solve_threshold(windows, target=0.0)
    assert t == THRESHOLD_SEARCH_HI
    assert realised_rate(windows, t) > 0.0


def test_solve_threshold_reports_in_camera_night_units_for_short_windows():
    """The micro corpus has shorter windows; the rate is scaled, not redefined."""
    frames = 27
    windows = two_hot_windows(10, frames=frames)
    scale = FRAMES_PER_WINDOW / frames
    assert realised_rate(windows, 0.5, frames_per_window=frames) == pytest.approx(2.0 * scale)
    t = solve_threshold(windows, target=2.0 * scale, frames_per_window=frames)
    assert realised_rate(windows, t, frames_per_window=frames) == pytest.approx(2.0 * scale)


def test_solve_threshold_refuses_an_empty_arm():
    with pytest.raises(ValueError, match="empty arm"):
        solve_threshold([])


def test_realised_rate_refuses_a_non_finite_hazard():
    with pytest.raises(ValueError, match="non-finite"):
        realised_rate([[0.5, float("nan")]], 0.4)


# --- BudgetState -------------------------------------------------------------------


def test_budget_allows_exactly_two_alerts_per_camera():
    b = BudgetState(["a", "b"])
    assert [b.spend("a") for _ in range(4)] == [True] * ALERTS_PER_CAMERA_PER_NIGHT + [False, False]
    assert b.spent("a") == ALERTS_PER_CAMERA_PER_NIGHT


def test_budget_is_per_camera_not_global():
    b = BudgetState(["a", "b"])
    for _ in range(3):
        b.spend("a")
    assert b.spend("b") is True
    assert b.spent("b") == 1
    assert b.total_spent() == ALERTS_PER_CAMERA_PER_NIGHT + 1


def test_budget_capacity_is_the_countable_slot_grid():
    assert BudgetState([f"c{i}" for i in range(40)]).capacity == 80


def test_budget_raises_on_a_camera_off_the_roster():
    b = BudgetState(["a"])
    with pytest.raises(KeyError):
        b.spend("typo")


def test_budget_refuses_a_duplicated_roster():
    with pytest.raises(ValueError, match="duplicate"):
        BudgetState(["a", "a"])


# --- emit --------------------------------------------------------------------------


def test_emit_ranks_by_hazard_descending_from_one():
    b = BudgetState(["a", "b", "c"])
    out = emit([score("a", 0.2), score("b", 0.9), score("c", 0.5)], 2.0, b)
    assert [e.camera_id for e in out] == ["b", "c", "a"]
    assert [e.rank for e in out] == [1, 2, 3]


def test_emit_breaks_ties_deterministically():
    """An unstable sort would make equal-hazard cameras swap places on the wall for no
    reason a viewer could explain, and would make bench.py non-reproducible."""
    b = BudgetState(["a", "b", "c"])
    forward = emit([score("c", 0.4), score("a", 0.4), score("b", 0.4)], 2.0, b)
    b2 = BudgetState(["a", "b", "c"])
    reverse = emit([score("b", 0.4), score("c", 0.4), score("a", 0.4)], 2.0, b2)
    assert [e.camera_id for e in forward] == [e.camera_id for e in reverse] == ["a", "b", "c"]


def test_emit_alerts_only_at_or_above_the_threshold():
    b = BudgetState(["a", "b"])
    out = {e.camera_id: e for e in emit([score("a", 0.60), score("b", 0.59)], 0.60, b)}
    assert out["a"].alerted is True
    assert out["b"].alerted is False
    assert b.total_spent() == 1


def test_emit_does_not_spend_budget_below_the_threshold():
    b = BudgetState(["a"])
    for _ in range(5):
        emit([score("a", 0.1)], 0.9, b)
    assert b.spent("a") == 0


def test_emit_never_suppresses_a_spent_camera():
    """spec.md section 9 safeguard 2: the system reorders, it never suppresses. A camera
    whose two alerts are gone still occupies its rank, carries its budget_spent count, and
    is merely not alerting -- because a row the operator cannot see is a row she cannot
    overrule."""
    b = BudgetState(["a", "b"])
    seen = []
    for _ in range(4):
        out = emit([score("a", 0.95), score("b", 0.10)], 0.5, b)
        assert [e.camera_id for e in out] == ["a", "b"]  # full roster, every tick
        seen.append(out[0])

    assert [e.alerted for e in seen] == [True, True, False, False]
    assert [e.budget_spent for e in seen] == [1, 2, 2, 2]
    assert seen[-1].rank == 1 and seen[-1].hazard == pytest.approx(0.95)


def test_emit_returns_the_full_roster_not_the_top_k():
    """TOP_K is a metrics concept -- what counts as a detection -- and truncating here
    would delete the evidence a human needs."""
    ids = [f"c{i:02d}" for i in range(40)]
    b = BudgetState(ids)
    out = emit([score(c, i / 100.0) for i, c in enumerate(ids)], 0.99, b)
    assert len(out) == 40
    assert {e.camera_id for e in out} == set(ids)


def test_emit_carries_the_chips_the_ui_displays():
    b = BudgetState(["a"])
    e = emit([score("a", 0.7, stats=stats(0.11, 0.22, 0.33, 0.44))], 0.5, b)[0]
    assert (e.s1, e.s2, e.s3, e.s4) == (0.11, 0.22, 0.33, 0.44)
    assert e.to_json()["s3"] == 0.33


def test_emit_raises_on_a_camera_seen_twice_in_one_tick():
    b = BudgetState(["a"])
    with pytest.raises(ValueError, match="twice"):
        emit([score("a", 0.9), score("a", 0.8)], 0.5, b)


def test_emit_raises_on_a_non_finite_hazard():
    b = BudgetState(["a"])
    with pytest.raises(ValueError, match="non-finite"):
        emit([score("a", float("inf"))], 0.5, b)


# --- draw_evidence -----------------------------------------------------------------


def gradient_frame(h: int = 270, w: int = 480) -> np.ndarray:
    """A smooth vertical ramp: JPEG encodes it almost losslessly, so any pixel difference
    the test finds is the overlay and not the codec."""
    ramp = np.linspace(60, 190, h, dtype=np.float32)
    return np.repeat(ramp[:, None, None], w, axis=1).repeat(3, axis=2).astype(np.uint8)


def anchored_track(bottom: int = 100) -> tuple[Component, Track]:
    """A pinned foot with a climbing top -- the thing the overlay is drawn to show."""
    t = Track(track_id=1)
    for i, top in enumerate((70, 62, 55, 48, 40)):
        c = Component(x=200, y=top, w=30, h=bottom - top, area=30 * (bottom - top),
                      cx=215.0, cy=(top + bottom) / 2.0)
        t.add(i, c, window=10)
    return t.last, t


def test_evidence_crop_box_stays_inside_the_frame():
    frame = gradient_frame()
    for comp in (
        Component(x=0, y=0, w=4, h=4, area=16, cx=2.0, cy=2.0),
        Component(x=476, y=266, w=4, h=4, area=16, cx=478.0, cy=268.0),
        Component(x=200, y=40, w=30, h=60, area=1800, cx=215.0, cy=70.0),
    ):
        x0, y0, x1, y1 = evidence_crop_box(comp, frame.shape)
        assert 0 <= x0 < x1 <= frame.shape[1]
        assert 0 <= y0 < y1 <= frame.shape[0]


def test_draw_evidence_writes_a_readable_jpeg_that_differs_from_the_crop(tmp_path):
    frame = gradient_frame()
    comp, track = anchored_track()
    out = draw_evidence(frame, comp, track, stats(0.92, 0.61, 0.74, 0.30),
                        tmp_path / "evidence" / "cam01_040.jpg")

    assert pathlib.Path(out).exists()
    written = cv2.imread(out, cv2.IMREAD_COLOR)
    assert written is not None, "cv2.imread could not decode what draw_evidence wrote"

    x0, y0, x1, y1 = evidence_crop_box(comp, frame.shape)
    undrawn = frame[y0:y1, x0:x1]
    assert written.shape == undrawn.shape

    diff = np.abs(written.astype(np.int16) - undrawn.astype(np.int16))
    assert diff.max() > 60, "nothing was drawn"
    assert (diff.max(axis=2) > 20).mean() > 0.02, "the overlay covers implausibly few pixels"


def test_the_two_overlay_lines_carry_their_own_tokens(tmp_path):
    """#FFD166 is red-dominant and #56C2FF is blue-dominant, so a swap of the anchor and
    growth tokens -- which would invert the whole visual argument -- is detectable through
    JPEG chroma subsampling without matching exact colours."""
    frame = gradient_frame()
    comp, track = anchored_track()
    out = draw_evidence(frame, comp, track, stats(0.9, 0.6, 0.7, 0.3), tmp_path / "e.jpg")
    written = cv2.imread(out, cv2.IMREAD_COLOR).astype(np.float32)

    x0, y0, _, _ = evidence_crop_box(comp, frame.shape)
    foot_row = written[100 - y0]          # median bottom of the track
    top_row = written[comp.top - y0]

    assert foot_row[:, 2].mean() > foot_row[:, 0].mean() + 20   # anchor: R over B
    assert top_row[:, 0].mean() > top_row[:, 2].mean() + 20     # growth: B over R


def test_draw_evidence_on_a_real_hpwren_frame(tmp_path):
    """Synthetic ramps cannot show that the overlay survives real sky texture."""
    jpegs = sorted(SPIKE_FRAMES.glob("*.jpg"))
    if not jpegs:
        pytest.skip("no HPWREN frames on disk")
    full = cv2.imread(str(jpegs[0]), cv2.IMREAD_COLOR)
    assert full is not None
    frame = cv2.resize(full, (480, 270), interpolation=cv2.INTER_AREA)

    comp, track = anchored_track()
    out = draw_evidence(frame, comp, track, stats(0.88, 0.55, 0.69, 0.21), tmp_path / "hpwren.jpg")
    written = cv2.imread(out, cv2.IMREAD_COLOR)
    x0, y0, x1, y1 = evidence_crop_box(comp, frame.shape)
    assert written is not None and written.shape == frame[y0:y1, x0:x1].shape
    assert np.abs(written.astype(np.int16) - frame[y0:y1, x0:x1].astype(np.int16)).max() > 60


def test_draw_evidence_refuses_an_empty_track(tmp_path):
    with pytest.raises(ValueError, match="non-empty track"):
        draw_evidence(gradient_frame(), anchored_track()[0], Track(track_id=2),
                      stats(0.5, 0.5, 0.5, 0.5), tmp_path / "x.jpg")


def test_draw_evidence_refuses_a_single_channel_frame(tmp_path):
    comp, track = anchored_track()
    with pytest.raises(ValueError, match="BGR"):
        draw_evidence(np.zeros((270, 480), np.uint8), comp, track,
                      stats(0.5, 0.5, 0.5, 0.5), tmp_path / "x.jpg")


def test_draw_evidence_raises_rather_than_silently_failing_to_write(tmp_path):
    """A row whose thumbnail silently failed to render is a row nobody can overrule."""
    comp, track = anchored_track()
    with pytest.raises(ValueError, match="imwrite"):
        draw_evidence(gradient_frame(), comp, track, stats(0.5, 0.5, 0.5, 0.5),
                      tmp_path / "evidence.unknownext")


# --- adversarial: the paths that would return a plausible number ---------------------


def test_realised_rate_refuses_a_window_with_no_hazards():
    """An empty window is a stage failure, not a quiet camera-night: CameraScore carries
    ``component=None``, so the pipeline produces a hazard for every camera on every frame.
    Counting it as a zero-alert night dilutes the denominator invisibly."""
    with pytest.raises(ValueError, match="no hazards"):
        realised_rate([[0.5] * 10, []], 0.4)


def test_solve_threshold_refuses_an_arm_of_empty_windows():
    """The whole-corpus version of the same defect, and the expensive one: the rate is 0.0
    at every threshold, the budget is met at the low bound, and THRESHOLD_SEARCH_LO --
    'alert on every frame' -- gets published as a calibrated operating point."""
    with pytest.raises(ValueError, match="no hazards"):
        solve_threshold([[], [], []])


def test_realised_rate_refuses_a_non_positive_window_length():
    """metrics.alerts_per_camera_night swallows a zero into scale=1.0 and lets a negative
    through into a negative rate, which then satisfies any budget at the low bound."""
    windows = two_hot_windows(4)
    for bad in (0, -FRAMES_PER_WINDOW):
        with pytest.raises(ValueError, match="frames_per_window"):
            realised_rate(windows, 0.5, frames_per_window=bad)


def test_emit_refuses_an_empty_tick():
    """emit's contract is the FULL roster. An empty list is a lost roster, and returning
    it renders an empty wall that looks exactly like a quiet night."""
    with pytest.raises(ValueError, match="no cameras"):
        emit([], 0.5, BudgetState(["a"]))


def test_emit_refuses_a_non_finite_chip():
    """The hazard was already checked; the chips were not. WatchEntry.to_json would emit a
    bare NaN, which is not JSON -- the browser's SSE parse throws and the wall silently
    stops updating."""
    b = BudgetState(["a"])
    with pytest.raises(ValueError, match="non-finite"):
        emit([score("a", 0.9, stats=stats(0.5, float("nan"), 0.5, 0.5))], 0.5, b)


def test_fuse_ignores_s4_when_the_ablation_gate_is_closed(monkeypatch):
    """S4_ENABLED is a model switch, not an input that gets zeroed: with the gate closed the
    hazard must not depend on S4 at all, and a broken S4 must still raise rather than hide
    behind the closed gate."""
    monkeypatch.setattr(rank, "S4_ENABLED", False)
    assert fuse(stats(0.3, 0.3, 0.3, 0.0)) == fuse(stats(0.3, 0.3, 0.3, 1.0))
    with pytest.raises(ValueError, match="non-finite"):
        fuse(stats(0.3, 0.3, 0.3, float("nan")))


def test_draw_evidence_does_not_mutate_the_caller_frame(tmp_path):
    """A full-width crop is a contiguous VIEW of the frame, so drawing into it draws into
    the frame the rest of the tick still has to use. The case is not exotic -- a fog bank
    spanning the frame is what the CONFUSE arm is made of."""
    frame = gradient_frame()
    comp = Component(x=8, y=60, w=464, h=90, area=464 * 90, cx=240.0, cy=105.0)
    track = Track(track_id=3)
    track.add(0, comp, window=10)
    before = frame.copy()

    draw_evidence(frame, comp, track, stats(0.5, 0.5, 0.5, 0.5), tmp_path / "wide.jpg")

    assert evidence_crop_box(comp, frame.shape)[0] == 0, "this case must be the full-width one"
    assert np.array_equal(frame, before), "draw_evidence drew into its caller's frame"


def test_the_pinned_foot_line_is_the_window_median_not_the_last_frame(tmp_path):
    """One noisy segmentation at the end of the window must not drag the anchor line -- and
    with it the caption's gap -- down the image. Mean and last-frame implementations both
    land on rows this test checks are clean."""
    frame = gradient_frame()
    track = Track(track_id=4)
    for i, bottom in enumerate((100, 100, 100, 100, 148)):
        c = Component(x=200, y=40, w=30, h=bottom - 40, area=30 * (bottom - 40),
                      cx=215.0, cy=(40 + bottom) / 2.0)
        track.add(i, c, window=10)
    comp = track.last

    out = draw_evidence(frame, comp, track, stats(0.9, 0.6, 0.7, 0.3), tmp_path / "median.jpg")
    written = cv2.imread(out, cv2.IMREAD_COLOR).astype(np.float32)
    _, y0, _, _ = evidence_crop_box(comp, frame.shape)

    def anchor_strength(src_row: int) -> float:
        row = written[src_row - y0]
        return float(row[:, 2].mean() - row[:, 0].mean())

    assert anchor_strength(100) > 20.0, "the anchor is not on the median bottom"
    assert anchor_strength(148) < 20.0, "the anchor followed the last frame"
    assert anchor_strength(110) < 20.0, "the anchor followed the mean"


def test_evidence_crop_box_clamps_to_a_frame_smaller_than_the_minimum_crop():
    """The minimum crop is a floor on legibility, not a promise the frame can keep."""
    comp = Component(x=10, y=10, w=20, h=20, area=400, cx=20.0, cy=20.0)
    assert evidence_crop_box(comp, (90, 100, 3)) == (0, 0, 100, 90)


# ---------------------------------------------------------------------------------
# Per-camera baseline (gate 0, 2026-09-16) -- what is ranked is a camera's departure
# from its own night, and the wall and the benchmark must compute it identically
# ---------------------------------------------------------------------------------


def _night(seed: int, n: int = 79) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in np.clip(rng.normal(0.45, 0.06, n), 0.0, 1.0)]


def test_baseline_live_equals_offline_tick_for_tick():
    """``Baseline.observe`` and ``rebase_series`` are the same function of the same history.

    The benchmark solves the threshold on ``rebase_series``; the wall runs on ``Baseline``.
    A single frame of disagreement is a threshold the wall never runs at.
    """
    live = rank.Baseline()
    for cam, seed in (("a", 1), ("b", 2), ("c", 3)):
        series = _night(seed)
        assert [live.observe(cam, h) for h in series] == rank.rebase_series(series)


def test_baseline_is_zero_through_the_warm_up_and_only_then():
    series = _night(4)
    out = rank.rebase_series(series)
    assert out[: BASELINE_WARMUP_FRAMES] == [0.0] * BASELINE_WARMUP_FRAMES
    assert all(0.0 < v < 1.0 for v in out[BASELINE_WARMUP_FRAMES:])


def test_baseline_never_scores_a_frame_against_itself():
    """A step up on an otherwise flat camera must read as a departure, not as the new normal."""
    flat = [0.4] * 20
    out = rank.rebase_series(flat + [0.9])
    assert out[-1] > 0.99, "the step frame is scored against the flat past only"
    assert out[BASELINE_WARMUP_FRAMES:-1] == pytest.approx([0.5] * (len(flat) - BASELINE_WARMUP_FRAMES)), "z == 0 on the flat run"


def test_baseline_is_causal():
    """Frame t reads frames < t only: changing the future changes nothing before it."""
    a = _night(5)
    b = list(a)
    b[40:] = [1.0] * (len(b) - 40)
    assert rank.rebase_series(a)[:40] == rank.rebase_series(b)[:40]


def test_baseline_sd_floor_stops_a_flat_camera_from_maximal_alerting_on_a_blip():
    """A camera with no component emits a constant; the first blip must not be an alert."""
    flat = [0.35] * 20
    blip = rank.rebase_series(flat + [0.35 + BASELINE_SD_FLOOR])[-1]
    # one floor-width above a flat past is z == 1, squashed: well below any solved threshold
    assert blip == pytest.approx(1.0 / (1.0 + math.exp(-1.0 / BASELINE_Z_SCALE)))
    assert blip < 0.75


def test_baseline_is_monotone_in_the_hazard_given_the_same_past():
    past = _night(6, n=BASELINE_WINDOW_FRAMES)
    values = [rank.rebase(h, past) for h in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert values == sorted(values) and len(set(values)) == 5


def test_baseline_window_is_the_frozen_constant_and_forgets_beyond_it():
    """A departure older than the window no longer lifts the camera's floor."""
    series = [0.4] * 10 + [0.95] + [0.4] * (BASELINE_WINDOW_FRAMES + 1)
    out = rank.rebase_series(series)
    # while the spike is inside the window the flat frames sit BELOW their own mean (z < 0)
    assert out[12] < 0.5
    # once it has left the window they are back on the mean exactly
    assert out[-1] == pytest.approx(0.5)


def test_baseline_reset_forgets_every_camera():
    live = rank.Baseline()
    for h in _night(7, n=10):
        live.observe("a", h)
    live.reset()
    assert live.observe("a", 0.9) == 0.0, "a fresh night starts in the warm-up"


def test_baseline_refuses_a_non_finite_hazard():
    with pytest.raises(ValueError, match="non-finite"):
        rank.Baseline().observe("a", float("nan"))


def test_baseline_refuses_a_zero_window():
    with pytest.raises(ValueError):
        rank.Baseline(window=0)

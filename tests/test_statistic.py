"""Tests for THE PRIMITIVE.

Every fixture here has an analytically known answer, because a statistic tested only
against real imagery is a statistic tested against whatever it happens to do. A wedge
whose foot is nailed to row 200 has S1 = 1 by construction; a disc that grows equally in
both directions has S2 = 0 by construction; a rectangle with a step edge and a flat
interior has S3 near 0 by construction. If any of those three stops being true, the
statistic has stopped meaning what its name says.

``test_s3_matches_the_gate`` is the load-bearing one, and it is not a unit test of S3 --
it is a test of the project's own honesty. The GO decision was taken on an AUC produced
by ``spike/s3_separation.py``. That AUC is evidence about the shipped system only while
the shipped system returns the same float, so this file imports the gate script by path
and diffs the two.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import cv2
import numpy as np
import pytest

from plumerank.constants import (
    MAX_PLAUSIBLE_SHIFT_PX,
    S2_ISOTROPIC_RATIO,
    S2_WIDTH_RATE_FLOOR_PX_PER_FRAME,
    S4_MIN_BBOX_PX,
    STAT_WINDOW_FRAMES,
    WORK_HEIGHT,
    WORK_WIDTH,
)
from plumerank.statistic import (
    Tracker,
    measure,
    s1_anchoring,
    s2_anisotropy,
    s3_diffuse_boundary,
    s4_drift,
    squash,
)
from plumerank.types import Component, Track

REPO = pathlib.Path(__file__).resolve().parents[1]
SPIKE_PATH = REPO / "spike" / "s3_separation.py"
SPIKE_FRAMES = REPO / "spike" / "frames"


# --- fixtures with known answers -------------------------------------------------


def comp(x: int, y: int, w: int, h: int) -> Component:
    """A Component with its derived fields consistent, which the tracker relies on."""
    return Component(x=x, y=y, w=w, h=h, area=w * h, cx=x + w / 2.0, cy=y + h / 2.0)


def track_from(components: list[Component], start: int = 0) -> Track:
    tr = Track(track_id=1)
    for i, c in enumerate(components):
        tr.add(start + i, c, window=STAT_WINDOW_FRAMES)
    return tr


def anchored_wedge() -> Track:
    """Foot nailed to row 200, apex climbing 3 px/frame, width fixed. S1 = 1, S2 high."""
    return track_from([comp(x=100, y=190 - 3 * i, w=6, h=10 + 3 * i) for i in range(10)])


def translating_blob() -> Track:
    """A rigid 20x20 body rising 3 px/frame -- a cloud. Foot moves exactly as far as top."""
    return track_from([comp(x=100, y=190 - 3 * i, w=20, h=20) for i in range(10)])


def isotropic_disc() -> Track:
    """Radius growing 1 px/frame about a fixed centre: d(h)/dt == d(w)/dt exactly."""
    out = []
    for i in range(10):
        r = 5 + i
        out.append(comp(x=100 - r, y=100 - r, w=2 * r, h=2 * r))
    return track_from(out)


def hard_edged_roi(seed: int = 3) -> np.ndarray:
    """Flat bright interior inside a step edge that lands in the boundary band.

    b = int(0.15 * 48) = 7, so the ring is rows/cols 0..6 and 41..47. The step at index 3
    and 44 sits inside it; the interior sees only the noise floor. Low Laplacian over high
    Sobel -- a cloud shadow, or the leading edge of a fog bank.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((48, 48), dtype=np.float64)
    img[3:45, 3:45] = 255.0
    img += rng.normal(0.0, 1.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def soft_edged_roi(seed: int = 3) -> np.ndarray:
    """Turbulent interior that fades to flat before the boundary band -- smoke.

    The texture mask is confined to [12:36] and blurred at sigma 2, so by index 7 (the
    ring's inner edge) it is 2.5 sigma out and contributes nothing. High Laplacian over
    low Sobel.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((48, 48), dtype=np.float64)
    mask[12:36, 12:36] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
    img = 128.0 + rng.normal(0.0, 45.0, mask.shape) * mask
    img += rng.normal(0.0, 1.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def whole_roi(roi: np.ndarray) -> Component:
    return comp(0, 0, roi.shape[1], roi.shape[0])


# --- the squash ------------------------------------------------------------------


def test_squash_is_monotone_and_bounded():
    xs = [0.0, 0.1, 1.0, 3.0, 40.0, 1e6]
    ys = [squash(x) for x in xs]
    assert ys == sorted(ys)
    assert all(0.0 <= y < 1.0 for y in ys)
    assert squash(0.0) == 0.0
    assert squash(1.0) == pytest.approx(0.5)


def test_squash_floors_negative_evidence_and_propagates_nan():
    assert squash(-5.0) == 0.0
    assert math.isnan(squash(math.nan))
    assert squash(math.inf) == 1.0


# --- S1 --------------------------------------------------------------------------


def test_s1_is_one_for_an_anchored_growing_wedge():
    assert s1_anchoring(anchored_wedge()) == pytest.approx(1.0, abs=1e-6)


def test_s1_is_zero_for_a_translating_blob():
    """Foot drifts exactly as far as the top climbs, so nothing was anchored."""
    assert s1_anchoring(translating_blob()) == 0.0


@pytest.mark.parametrize("rise_px_per_frame", [0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
def test_s1_rejects_rigid_translation_at_every_speed(rise_px_per_frame):
    """The regression that matters: S1 must not reward a fog bank for drifting slowly.

    A cloud, a fog bank and a lens flare all translate as rigid bodies -- foot and top
    move together -- and S1 exists to score exactly that at 0. Charging the dead band to
    the foot but not to the top made the leak grow as the motion slowed: 0.15 at
    3 px/frame but **0.89 at 0.5 px/frame**, because the tolerance ate the whole drift and
    none of the climb. 0.5 px/frame at 480x270 is an ordinary dawn fog bank, and S1 is the
    second-heaviest term in the fusion, so the leak sat squarely on the false-alarm class
    the alert budget is calibrated against. Parametrised over speed on purpose: a single
    speed is what let this through, because 3 px/frame is where the leak is smallest.
    """
    rigid = track_from(
        [comp(x=100, y=int(190 - rise_px_per_frame * i), w=20, h=20) for i in range(10)]
    )
    assert s1_anchoring(rigid) == pytest.approx(0.0, abs=1e-9)


def test_s1_falls_monotonically_as_the_foot_gives_way():
    """Partial anchoring must be ordered, or the ranking cannot use S1 to order anything."""
    climb = 30
    scores = []
    for slip in (0, 5, 10, 20, 29):
        # The top climbs 30 px in every case; only how far the foot follows it changes.
        scores.append(
            s1_anchoring(
                track_from(
                    [
                        comp(
                            x=100,
                            y=190 - climb * i // 9,
                            w=20,
                            h=20 + climb * i // 9 - slip * i // 9,
                        )
                        for i in range(10)
                    ]
                )
            )
        )
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] > 0.9 and scores[-1] < 0.2


def test_s1_is_zero_when_the_top_does_not_climb():
    """A component sinking down the frame is not a plume, whatever its foot does."""
    sinking = track_from([comp(x=100, y=100 + 3 * i, w=10, h=10) for i in range(6)])
    assert s1_anchoring(sinking) == 0.0


def test_s1_tolerates_jitter_inside_the_dead_band():
    """A foot wobbling by less than the tolerance must cost exactly nothing."""
    wobble = [0, 1, -1, 2, -2, 1, 0, 1, -1, 0]  # peak-to-peak 4 == the tolerance
    jittery = track_from(
        [comp(x=100, y=190 - 3 * i, w=6, h=10 + 3 * i + wobble[i]) for i in range(10)]
    )
    assert s1_anchoring(jittery) == pytest.approx(1.0, abs=1e-6)


def test_s1_needs_two_frames():
    assert s1_anchoring(track_from([comp(100, 100, 10, 10)])) == 0.0


# --- S2 --------------------------------------------------------------------------


def test_s2_is_high_for_a_wedge_that_rises_without_spreading():
    """d(h)/dt = 3, d(w)/dt = 0 -> floored to 0.25 -> ratio 12 -> squash(11)."""
    assert s2_anisotropy(anchored_wedge()) == pytest.approx(11.0 / 12.0, abs=1e-9)


def test_s2_is_zero_for_an_isotropically_growing_disc():
    """The name is the assertion: an isotropic thing has no anisotropy."""
    assert s2_anisotropy(isotropic_disc()) == 0.0


def test_s2_has_an_absolute_growth_floor():
    """A documented consequence of the floored denominator, asserted so it stays known.

    Because the ratio's denominator is floored at S2_WIDTH_RATE_FLOOR_PX_PER_FRAME and the
    squash is fed the excess over S2_ISOTROPIC_RATIO, a component whose HEIGHT rate is
    below that floor scores 0.0 however little it widens. Intended -- two pixels of growth
    across the ten minute window is segmentation jitter -- but it is a real dead zone and a
    reader of the ranking is entitled to find it written down rather than infer it.
    """
    slow = S2_WIDTH_RATE_FLOOR_PX_PER_FRAME * 0.8
    vertical_only = track_from(
        [comp(x=100, y=100, w=10, h=10 + slow * i) for i in range(10)]
    )
    assert s2_anisotropy(vertical_only) == 0.0

    fast = S2_WIDTH_RATE_FLOOR_PX_PER_FRAME * 4.0
    assert s2_anisotropy(
        track_from([comp(x=100, y=100, w=10, h=10 + fast * i) for i in range(10)])
    ) == pytest.approx(squash(4.0 - S2_ISOTROPIC_RATIO))


def test_s2_orders_wedges_by_how_much_they_out_climb_their_spread():
    """Ordering, not a single point: the ranking consumes S2 as an order relation."""
    scores = [
        s2_anisotropy(
            track_from([comp(x=100, y=100, w=10 + spread * i, h=10 + 4 * i) for i in range(10)])
        )
        for spread in (0, 1, 2, 4)
    ]
    assert scores == sorted(scores, reverse=True), scores


def test_s2_is_zero_for_a_body_that_does_not_grow():
    assert s2_anisotropy(translating_blob()) == 0.0


def test_s2_treats_shrinking_as_no_growth_not_negative_growth():
    shrinking = track_from([comp(100, 100, 30, 30 - 2 * i) for i in range(8)])
    assert s2_anisotropy(shrinking) == 0.0


# --- S3: THE GATE ----------------------------------------------------------------


def load_gate():
    """Import the GO/NO-GO script by file path, exactly as it sat on Sep 1."""
    assert SPIKE_PATH.exists(), f"the gate script is missing: {SPIKE_PATH}"
    spec = importlib.util.spec_from_file_location("s3_gate", SPIKE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s3_matches_the_gate():
    """The published AUC is evidence about this code only if this code agrees with it.

    Forty ROIs of varied size and texture, including six too small to hold a boundary
    band so that the nan path is compared too. Agreement is asserted to 1e-9; the two
    implementations perform the same float32 operations in the same order, so in practice
    it is exact, and a tolerance that loose would still catch any reordering that matters.
    """
    gate = load_gate()
    rng = np.random.default_rng(20260901)

    numeric_comparisons = 0
    nan_comparisons = 0
    for _ in range(36):
        h = int(rng.integers(8, 91))
        w = int(rng.integers(8, 91))
        # Varied texture scale: a uniform ROI and a heavily blurred one exercise very
        # different numerator/denominator magnitudes, and the ratio must survive both.
        roi = (rng.random((h, w)) * 255).astype(np.uint8)
        if rng.random() < 0.5:
            roi = cv2.GaussianBlur(roi, (0, 0), float(rng.uniform(0.5, 4.0)))

        expected = gate.s3(roi, "v0")
        _, raw = s3_diffuse_boundary(roi, whole_roi(roi))
        assert np.isfinite(expected), "fixture should be large enough to score"
        assert raw == pytest.approx(expected, rel=1e-9, abs=1e-9)
        numeric_comparisons += 1

    for _ in range(6):
        h = int(rng.integers(2, 8))
        w = int(rng.integers(2, 8))
        roi = (rng.random((h, w)) * 255).astype(np.uint8)
        expected = gate.s3(roi, "v0")
        squashed, raw = s3_diffuse_boundary(roi, whole_roi(roi))
        assert math.isnan(expected)
        assert math.isnan(raw) and math.isnan(squashed)
        nan_comparisons += 1

    assert numeric_comparisons >= 30
    assert nan_comparisons == 6


def test_s3_matches_the_gate_on_a_crop_out_of_a_larger_frame():
    """Production never scores a whole array -- it scores a bbox inside a 480x270 frame.

    The other agreement test hands the ROI in as the entire ``grey``, so it cannot catch a
    crop that is off by a row, transposed, or taken with x and y swapped. Here the gate is
    given ``frame[y:y+h, x:x+w]`` directly and production is given the frame and the
    Component; only identical cropping makes the two floats agree.
    """
    gate = load_gate()
    rng = np.random.default_rng(4242)
    frame = cv2.GaussianBlur((rng.random((WORK_HEIGHT, WORK_WIDTH)) * 255).astype(np.uint8),
                             (0, 0), 1.7)

    for _ in range(12):
        w = int(rng.integers(16, 120))
        h = int(rng.integers(16, 120))          # w != h on purpose: a transposed crop
        x = int(rng.integers(0, WORK_WIDTH - w))  # scores a different, plausible number
        y = int(rng.integers(0, WORK_HEIGHT - h))
        expected = gate.s3(frame[y : y + h, x : x + w], "v0")
        _, raw = s3_diffuse_boundary(frame, comp(x=x, y=y, w=w, h=h))
        assert np.isfinite(expected)
        assert raw == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_s3_separates_a_soft_edge_from_a_hard_one():
    soft_squashed, soft_raw = s3_diffuse_boundary(soft_edged_roi(), whole_roi(soft_edged_roi()))
    hard_squashed, hard_raw = s3_diffuse_boundary(hard_edged_roi(), whole_roi(hard_edged_roi()))

    assert soft_raw > hard_raw
    assert soft_squashed > 0.8
    assert hard_squashed < 0.1


def test_s3_is_nan_not_zero_for_a_component_too_thin_to_band():
    """0.0 is a confident answer here; a missing measurement must not borrow it."""
    grey = np.full((WORK_HEIGHT, WORK_WIDTH), 100, dtype=np.uint8)
    squashed, raw = s3_diffuse_boundary(grey, comp(x=10, y=10, w=60, h=5))
    assert math.isnan(squashed) and math.isnan(raw)


def test_s3_is_nan_when_the_bbox_runs_off_the_frame():
    grey = np.full((WORK_HEIGHT, WORK_WIDTH), 100, dtype=np.uint8)
    squashed, raw = s3_diffuse_boundary(grey, comp(x=WORK_WIDTH - 10, y=10, w=40, h=40))
    assert math.isnan(squashed) and math.isnan(raw)


def test_s3_reads_exactly_the_bbox_the_caller_asked_for():
    """x/y transposed or off by a row would still return a plausible number, so pin it.

    The patch is planted at a deliberately asymmetric offset (x=60, y=40) and the frame
    around it is flat, which scores 0.0. Reading the patch standalone and reading it out of
    the frame must give the SAME float; a swapped x/y or an off-by-one lands on flat pixels
    and returns 0.0 instead.
    """
    grey = np.full((WORK_HEIGHT, WORK_WIDTH), 30, dtype=np.uint8)
    patch = soft_edged_roi()
    grey[40:88, 60:108] = patch

    _, standalone = s3_diffuse_boundary(patch, whole_roi(patch))
    squashed, raw = s3_diffuse_boundary(grey, comp(x=60, y=40, w=48, h=48))

    assert raw == pytest.approx(standalone, rel=1e-12)
    assert squashed == pytest.approx(squash(raw))
    assert squashed > 0.8
    # The transposed read is a different, entirely plausible-looking answer.
    _, transposed = s3_diffuse_boundary(grey, comp(x=40, y=60, w=48, h=48))
    assert transposed != pytest.approx(raw)


@pytest.mark.skipif(
    not SPIKE_FRAMES.exists() or not list(SPIKE_FRAMES.glob("*.jpg")),
    reason="HPWREN frames are gitignored third-party imagery; regenerable via spike/urls.txt",
)
def test_s3_scores_a_real_hpwren_frame_at_working_resolution():
    """Real 3072x2048 pixels, downscaled to the resolution the pipeline actually runs at."""
    path = sorted(SPIKE_FRAMES.glob("*.jpg"))[0]
    full = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert full is not None
    grey = cv2.resize(full, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA)
    squashed, raw = s3_diffuse_boundary(grey, comp(x=120, y=60, w=80, h=60))
    assert math.isfinite(raw) and 0.0 <= squashed < 1.0


# --- S4 --------------------------------------------------------------------------


def moving_texture(dy: int, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """A textured frame and the same frame shifted by dy rows (negative == upward)."""
    rng = np.random.default_rng(seed)
    prev = (rng.random((WORK_HEIGHT, WORK_WIDTH)) * 255).astype(np.uint8)
    prev = cv2.GaussianBlur(prev, (0, 0), 1.5)  # DIS needs structure, not white noise
    cur = np.roll(prev, dy, axis=0)
    return prev, cur


def test_s4_is_high_for_coherent_upward_drift():
    prev, cur = moving_texture(dy=-3)
    box = comp(x=100, y=80, w=64, h=64)
    assert s4_drift(cur, prev, box) > 0.8


def test_s4_is_zero_for_downward_drift():
    """Subsidence is the opposite of a plume, and negative evidence floors at 0."""
    prev, cur = moving_texture(dy=3)
    box = comp(x=100, y=80, w=64, h=64)
    assert s4_drift(cur, prev, box) == 0.0


def test_s4_is_computed_inside_the_bbox_only():
    """The contract's reason for in-bbox flow, made falsifiable.

    The frame outside the box sinks 6 px while the box itself climbs 4. A frame-wide flow
    field would be dominated by the descent and score ~0; only flow computed on the crop
    sees the climb. This is the test that fails if someone "optimises" S4 into one
    whole-frame ``calc`` per frame with the box sliced out afterwards.
    """
    rng = np.random.default_rng(7)
    base = cv2.GaussianBlur((rng.random((WORK_HEIGHT, WORK_WIDTH)) * 255).astype(np.uint8),
                            (0, 0), 1.5)
    x, y, w, h = 100, 80, 64, 64
    cur = np.roll(base, 6, axis=0)                                  # the frame sinks
    cur[y : y + h, x : x + w] = np.roll(base, -4, axis=0)[y : y + h, x : x + w]  # box climbs

    assert s4_drift(cur, base, comp(x=x, y=y, w=w, h=h)) > 0.8


def test_s4_prefers_coherent_drift_over_incoherent_motion_of_the_same_speed():
    """S4 is a signal-to-noise ratio, not a speed meter -- otherwise a shaking camera wins.

    Both pairs carry the same mean upward displacement; the second has per-row jitter added
    on top. The incoherent one must score strictly lower, or the denominator is decorative.
    """
    rng = np.random.default_rng(21)
    base = cv2.GaussianBlur((rng.random((WORK_HEIGHT, WORK_WIDTH)) * 255).astype(np.uint8),
                            (0, 0), 1.5)
    box = comp(x=100, y=80, w=64, h=64)

    coherent = np.roll(base, -4, axis=0)
    incoherent = coherent.copy()
    for row in range(WORK_HEIGHT):
        incoherent[row] = np.roll(coherent[row], int(rng.integers(-6, 7)))

    assert s4_drift(incoherent, base, box) < s4_drift(coherent, base, box)


def test_s4_is_zero_for_a_motionless_pair():
    """No motion is no evidence, and it must not borrow the value of upward motion."""
    rng = np.random.default_rng(5)
    frame = cv2.GaussianBlur((rng.random((WORK_HEIGHT, WORK_WIDTH)) * 255).astype(np.uint8),
                             (0, 0), 1.5)
    assert s4_drift(frame, frame.copy(), comp(x=100, y=80, w=64, h=64)) == 0.0


def test_s4_is_zero_when_the_bbox_runs_off_the_frame():
    """Not nan: S4 is additive evidence under an ablation gate, not a load-bearing read."""
    prev, cur = moving_texture(dy=-3)
    assert s4_drift(cur, prev, comp(x=WORK_WIDTH - 10, y=80, w=64, h=64)) == 0.0


def test_s4_skips_the_first_frame_of_a_replay():
    prev, cur = moving_texture(dy=-3)
    assert s4_drift(cur, None, comp(x=100, y=80, w=64, h=64)) == 0.0


def test_s4_skips_a_box_too_small_for_the_patch_grid():
    prev, cur = moving_texture(dy=-3)
    small = S4_MIN_BBOX_PX - 1
    assert s4_drift(cur, prev, comp(x=100, y=80, w=small, h=64)) == 0.0
    assert s4_drift(cur, prev, comp(x=100, y=80, w=64, h=small)) == 0.0


# --- the tracker -----------------------------------------------------------------


def test_tracker_associates_components_across_frames():
    tracker = Tracker()
    ids_seen: list[tuple[int, int]] = []
    for i in range(5):
        left = comp(x=50 + 2 * i, y=50, w=12, h=12)
        right = comp(x=300, y=150 - 2 * i, w=14, h=14)
        tracks = tracker.update(i, [left, right])
        assert len(tracks) == 2
        ids_seen.append((tracks[0].track_id, tracks[1].track_id))

    assert len(set(ids_seen)) == 1, "both components should keep their identity"
    assert ids_seen[0][0] != ids_seen[0][1]
    assert all(len(t) == 5 for t in tracker.tracks)


def test_tracker_returns_tracks_in_input_component_order():
    """The caller zips the two lists; a reordered return would silently mislabel rows."""
    tracker = Tracker()
    a = comp(x=40, y=40, w=10, h=10)
    b = comp(x=300, y=200, w=10, h=10)
    first = tracker.update(0, [a, b])
    second = tracker.update(1, [b, a])  # same components, swapped order
    assert [t.track_id for t in second] == [first[1].track_id, first[0].track_id]


def test_tracker_honours_the_statistics_window():
    tracker = Tracker()
    n = STAT_WINDOW_FRAMES + 5
    for i in range(n):
        tracker.update(i, [comp(x=100, y=100 - i, w=10, h=10 + i)])
    (track,) = tracker.tracks
    assert len(track) == STAT_WINDOW_FRAMES
    assert track.frames == list(range(n - STAT_WINDOW_FRAMES, n))


def test_tracker_starts_a_new_track_beyond_the_stabiliser_gate():
    tracker = Tracker()
    first = tracker.update(0, [comp(x=40, y=40, w=10, h=10)])[0]
    jump = int(MAX_PLAUSIBLE_SHIFT_PX) + 20
    second = tracker.update(1, [comp(x=40 + jump, y=40, w=10, h=10)])[0]
    assert second.track_id != first.track_id


def test_tracker_drops_a_track_that_missed_its_whole_window():
    tracker = Tracker()
    box = comp(x=40, y=40, w=10, h=10)
    first = tracker.update(0, [box])[0]

    kept = Tracker()
    kept.update(0, [box])
    assert kept.update(STAT_WINDOW_FRAMES, [box])[0].track_id == 1

    revived = tracker.update(STAT_WINDOW_FRAMES + 1, [box])[0]
    assert revived.track_id != first.track_id
    assert len(tracker.tracks) == 1, "the stale track must be gone, not merely unmatched"


def test_tracker_rejects_a_window_too_short_to_fit_a_slope():
    with pytest.raises(ValueError):
        Tracker(window=1)


# --- measure ---------------------------------------------------------------------


def test_measure_returns_all_four_statistics_in_range():
    grey = np.full((WORK_HEIGHT, WORK_WIDTH), 30, dtype=np.uint8)
    grey[40:88, 60:108] = soft_edged_roi()
    prev = np.roll(grey, 3, axis=0)

    track = track_from([comp(x=60, y=88 - 4 * i, w=48, h=4 * i + 8) for i in range(9)])
    stats = measure(track, grey, prev)

    for value in stats.as_tuple():
        assert 0.0 <= value <= 1.0
    assert stats.raw_s3 > 0.0
    assert stats.s3 == pytest.approx(squash(stats.raw_s3))


def test_measure_carries_the_raw_s3_in_the_gates_units():
    grey = np.full((WORK_HEIGHT, WORK_WIDTH), 30, dtype=np.uint8)
    grey[40:88, 60:108] = hard_edged_roi()
    track = track_from([comp(x=60, y=40, w=48, h=48)] * 3)
    stats = measure(track, grey)
    _, raw = s3_diffuse_boundary(grey, comp(x=60, y=40, w=48, h=48))
    assert stats.raw_s3 == pytest.approx(raw)


def test_measure_flags_an_unmeasurable_s3_rather_than_scoring_it_zero():
    grey = np.full((WORK_HEIGHT, WORK_WIDTH), 100, dtype=np.uint8)
    # 60x5: wide enough to pass MIN_COMPONENT_AREA_PX, too short for a 3 px band a side.
    track = track_from([comp(x=10, y=10 - i, w=60, h=5) for i in range(4)])
    stats = measure(track, grey)
    assert math.isnan(stats.s3) and math.isnan(stats.raw_s3)
    assert math.isfinite(stats.s1) and math.isfinite(stats.s2)


def test_measure_scores_the_tracks_most_recent_component_not_its_first():
    """An off-by-one on which end of the history is read would score a stale minute.

    The track starts on a flat region (S3 = 0.0) and ends on the soft patch. Reading
    ``history[0]`` instead of ``history[-1]`` returns 0.0 -- a confident, defensible,
    completely wrong "hard edge, flat interior" -- rather than raising.
    """
    grey = np.full((WORK_HEIGHT, WORK_WIDTH), 30, dtype=np.uint8)
    grey[40:88, 60:108] = soft_edged_roi()

    stale = comp(x=200, y=200, w=48, h=48)
    fresh = comp(x=60, y=40, w=48, h=48)
    stats = measure(track_from([stale, fresh]), grey)

    assert stats.raw_s3 == pytest.approx(s3_diffuse_boundary(grey, fresh)[1])
    assert s3_diffuse_boundary(grey, stale)[1] == 0.0


def test_measure_refuses_an_empty_track():
    with pytest.raises(ValueError):
        measure(Track(track_id=9), np.zeros((WORK_HEIGHT, WORK_WIDTH), np.uint8))


def test_measure_skips_s4_without_a_previous_frame():
    grey = np.full((WORK_HEIGHT, WORK_WIDTH), 30, dtype=np.uint8)
    grey[40:88, 60:108] = soft_edged_roi()
    track = track_from([comp(x=60, y=40, w=48, h=48)] * 3)
    assert measure(track, grey).s4 == 0.0


def test_gate_scores_at_the_resolution_production_computes_at():
    """The gate and the pipeline must ask S3 the same question at the same scale.

    ``test_s3_matches_the_gate`` proves the two IMPLEMENTATIONS agree on one ROI. It cannot
    prove they are handed comparable ROIs, and until 2026-09-06 they were not: the gate read
    native 3072x2048 frames while ``segment.py`` hands ``_s3_raw`` a 480x270 working frame.
    S3 is not scale-invariant (fixed 3x3 Laplacian/Sobel kernels), so that gap was worth a
    Spearman rank correlation of ~0.63 between the two scales -- enough to move the gate's
    verdict across its own cutline. This asserts the default never drifts back.
    """
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("s3sep", root / "spike" / "s3_separation.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    from plumerank import constants as K

    assert (gate.WORK_W, gate.WORK_H) == (K.WORK_WIDTH, K.WORK_HEIGHT), (
        "the gate's working resolution has drifted from the pipeline's"
    )

    import inspect

    sig = inspect.signature(gate.score_box)
    assert sig.parameters["scale"].default == "work", (
        "the gate must default to production resolution; 'full' measures a statistic the "
        "product does not compute"
    )


def test_the_dis_cache_is_bounded_and_eviction_does_not_change_the_flow() -> None:
    """The per-shape DIS cache grew without bound on the live box (OOM). It is an LRU now,
    and a shape that was evicted must compute the same flow as one that never was."""
    import cv2

    from plumerank import statistic as st

    rng = np.random.default_rng(5)
    a = rng.integers(0, 255, (60, 90), dtype=np.uint8)
    b = np.roll(a, -1, axis=0)
    st._DIS_BY_SHAPE.clear()
    first = st._dis((60, 90)).calc(a, b, None)
    for h in range(32, 32 + st.DIS_CACHE_SIZE + 10):  # flood the cache past its size
        st._dis((h, 100))
    assert len(st._DIS_BY_SHAPE) <= st.DIS_CACHE_SIZE
    assert (60, 90) not in st._DIS_BY_SHAPE
    again = st._dis((60, 90)).calc(a, b, None)
    fresh = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(a, b, None)
    assert np.array_equal(first, again)
    assert np.array_equal(first, fresh)

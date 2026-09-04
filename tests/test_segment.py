"""Tests for plumerank.segment.

The fixtures are synthetic on purpose: a wedge whose area is a known increasing function
of the frame index is the only way to assert "the segmentation tracks growth" without a
labelled corpus, and the growth is exactly what S1 and S2 are computed from downstream.
The real HPWREN frames appear once, at the end, as a smoke test -- they prove the stage
survives 3072x2048 JPEGs downscaled to the working resolution, which no synthetic fixture
can prove, and they assert nothing about correctness because they carry no ground truth.
"""

from __future__ import annotations

import glob
import pathlib

import cv2
import numpy as np
import pytest

from plumerank.constants import (
    MASK_FOREGROUND_VALUE,
    MAX_COMPONENTS_PER_FRAME,
    MIN_COMPONENT_AREA_PX,
    WARMUP_FRAMES,
    WORK_HEIGHT,
    WORK_WIDTH,
)
from plumerank.segment import METHODS, Segmenter, _binarise
from plumerank.types import Component

# The wedge is measured from this frame onwards. The first frames are excluded because
# the background model is still converging there -- which is the honest reason the
# ``warm`` property exists, and asserting monotone growth through the convergence would
# be asserting that a warming model is already warm.
_MEASURE_FROM = 15
_N_FRAMES = 45


# ---------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------


def _static_background(seed: int = 7) -> np.ndarray:
    """A fixed textured sky. Textured, not flat: a flat background makes every subtractor
    look perfect, and the ridgeline texture is what MOG2 actually has to hold still."""
    rng = np.random.RandomState(seed)
    return (rng.rand(WORK_HEIGHT, WORK_WIDTH) * 60 + 90).astype(np.uint8)


def _wedge_frame(background: np.ndarray, k: int, base_y: int = 200) -> np.ndarray:
    """Background plus a bright wedge whose foot is pinned at ``base_y`` and whose top
    climbs with ``k``. This is the shape S1 claims a fire has: the lowest extent does not
    move, the highest one does."""
    frame = background.copy()
    half = 6 + k
    height = 4 + 3 * k
    for i in range(height):
        y = base_y - i
        if y < 0:
            break
        w = max(1, int(half * (1 - 0.4 * i / max(height, 1))))
        frame[y, 240 - w : 240 + w] = 235
    return frame


def _noise_frame(rng: np.random.RandomState, sigma: float = 6.0) -> np.ndarray:
    """A fresh draw of i.i.d. sensor noise -- no scene, nothing to detect."""
    return np.clip(140 + rng.randn(WORK_HEIGHT, WORK_WIDTH) * sigma, 0, 255).astype(np.uint8)


_BLOB_GAP = 12  # > the 7 px CLOSE kernel, so two squares can never weld into one


def _blob_frame(background: np.ndarray, sides: list[int]) -> np.ndarray:
    """Background plus squares of the given side lengths, laid out on a wrapping grid.

    The layout wraps into rows and asserts it fits. The first version of this fixture ran
    a single row off the right edge, where numpy slicing silently drops the write -- so a
    twelve-blob fixture was really a nine-blob fixture and the cap test was passing on a
    number nobody had checked. A fixture that lies is worse than no fixture.
    """
    frame = background.copy()
    x, y, row_height = 20, 20, 0
    for side in sides:
        if x + side > WORK_WIDTH - 20:
            x, y, row_height = 20, y + row_height + _BLOB_GAP, 0
        assert y + side <= WORK_HEIGHT - 20, f"blob layout does not fit: {sides}"
        frame[y : y + side, x : x + side] = 240
        x += side + _BLOB_GAP
        row_height = max(row_height, side)
    return frame


# ---------------------------------------------------------------------------------
# The growing wedge -- the contract test, run on all three subtractors
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_growing_wedge_yields_one_monotonically_growing_component(method: str) -> None:
    background = _static_background()
    segmenter = Segmenter(method)

    frames = [segmenter.apply(_wedge_frame(background, k)) for k in range(_N_FRAMES)]
    measured = frames[_MEASURE_FROM:]

    assert all(len(c) == 1 for c in measured), (
        f"{method}: expected exactly one component per frame, got "
        f"{[len(c) for c in measured]}"
    )

    areas = [c[0].area for c in measured]
    assert all(b >= a for a, b in zip(areas, areas[1:])), f"{method}: area not monotone: {areas}"
    assert areas[-1] > areas[0], f"{method}: wedge never grew: {areas[0]} -> {areas[-1]}"

    # The foot must stay pinned. This is not S1 -- S1 is statistic.py's job -- but if the
    # segmentation let the bottom edge wander, S1 would be measuring the morphology
    # rather than the fire.
    bottoms = [c[0].bottom for c in measured]
    assert max(bottoms) - min(bottoms) <= 4, f"{method}: foot drifted {bottoms}"

    # The top climbs: the other half of the same claim.
    tops = [c[0].top for c in measured]
    assert tops[-1] < tops[0], f"{method}: top never climbed: {tops[0]} -> {tops[-1]}"


@pytest.mark.parametrize("method", METHODS)
def test_every_method_constructs_its_own_subtractor(method: str) -> None:
    """The three are distinct objects, not one object behind three labels.

    Without this, a typo in ``_new_subtractor`` would publish a ``bench.py --bg lsbp``
    row that was really MOG2, and the comparison table -- the thing that makes bgsegm a
    measured alternative rather than a name-drop -- would be quietly false.
    """
    segmenter = Segmenter(method)
    assert segmenter.method == method
    assert method in type(segmenter._subtractor).__name__.lower()


def test_pure_noise_yields_no_components_after_warmup() -> None:
    """MOG2, the shipped default, must not invent a plume out of sensor noise.

    Asserted on ``mog2`` alone and deliberately so: LSBP and GSOC at their published
    defaults *do* flag i.i.d. noise at this amplitude, right up to the whole frame. That
    is a real property of those algorithms, it is what ``bench.py --bg`` exists to
    measure, and it goes in the DEMO.md table whichever way it falls -- it is not
    something to hide behind a laxer assertion here.
    """
    rng = np.random.RandomState(11)
    segmenter = Segmenter("mog2")

    post_warm: list[int] = []
    for k in range(40):
        components = segmenter.apply(_noise_frame(rng))
        if k >= WARMUP_FRAMES:
            post_warm.append(len(components))

    assert post_warm, "warm-up consumed the whole sequence -- the test asserts nothing"
    assert sum(post_warm) == 0, f"noise produced components: {post_warm}"


# ---------------------------------------------------------------------------------
# Warm-up is a state, not a silence
# ---------------------------------------------------------------------------------


def test_apply_runs_and_returns_before_warm() -> None:
    background = _static_background()
    segmenter = Segmenter()

    seen: list[list[Component]] = []
    for k in range(WARMUP_FRAMES):
        assert not segmenter.warm
        seen.append(segmenter.apply(_wedge_frame(background, k)))

    assert all(isinstance(c, list) for c in seen)
    # At least one pre-warm frame produced something. A stage that returned [] until warm
    # would look, from the UI's side, exactly like a working detector finding nothing.
    assert any(seen), "nothing at all was returned during warm-up"


def test_warm_flips_exactly_at_warmup_frames() -> None:
    background = _static_background()
    segmenter = Segmenter()

    assert segmenter.frames_seen == 0
    assert not segmenter.warm
    for k in range(WARMUP_FRAMES - 1):
        segmenter.apply(_wedge_frame(background, k))
    assert not segmenter.warm
    segmenter.apply(_wedge_frame(background, WARMUP_FRAMES))
    assert segmenter.frames_seen == WARMUP_FRAMES
    assert segmenter.warm


def test_reset_returns_the_model_to_frame_zero() -> None:
    """Not just the counter: the background model itself.

    The check is behavioural rather than an identity check on the object, because what
    matters is that last night's sky is gone. A reset segmenter re-fed the same frames
    must produce the same components a fresh one does -- if the model survived, the
    static background would already be learned and the first frames would differ.
    """
    background = _static_background()

    used = Segmenter()
    for k in range(20):
        used.apply(_wedge_frame(background, k))
    assert used.warm
    used.reset()
    assert used.frames_seen == 0
    assert not used.warm

    fresh = Segmenter()
    for k in range(12):
        after_reset = used.apply(_wedge_frame(background, k))
        from_fresh = fresh.apply(_wedge_frame(background, k))
        assert [c.area for c in after_reset] == [c.area for c in from_fresh]


# ---------------------------------------------------------------------------------
# The component filter
# ---------------------------------------------------------------------------------


def test_the_cap_drops_the_smallest_not_the_first() -> None:
    """The cap is applied AFTER the sort, over more candidates than it keeps.

    The previous version laid its blobs out in one row that ran off the frame, so only
    nine of the twelve squares existed and the assertion was weaker than it read. Here
    the four squares the cap must drop are shown -- in their own segmenter -- to be real
    components that clear the area floor, so the cap is doing the dropping. Component
    labels arrive in raster order, which puts side 30, 12, 26, 16 first: a bare
    truncation would keep the 12 px square and throw away the 32 px one.
    """
    background = _static_background()
    sides = [30, 12, 26, 16, 34, 14, 22, 18, 28, 20, 24, 32]
    biggest = sorted(sides, reverse=True)[:MAX_COMPONENTS_PER_FRAME]
    smallest = sorted(sides)[: len(sides) - MAX_COMPONENTS_PER_FRAME]

    dropped = Segmenter()
    for _ in range(WARMUP_FRAMES):
        dropped.apply(background)
    candidates = dropped.apply(_blob_frame(background, smallest))
    assert len(candidates) == len(smallest), (
        "the squares the cap is supposed to drop never became components, so this test "
        f"would pass with no cap at all: {[c.area for c in candidates]}"
    )

    capped = Segmenter()
    for _ in range(WARMUP_FRAMES):
        capped.apply(background)
    components = capped.apply(_blob_frame(background, sides))

    assert len(components) == MAX_COMPONENTS_PER_FRAME
    areas = [c.area for c in components]
    assert areas == sorted(areas, reverse=True), f"not largest-first: {areas}"
    # Each square keeps its side as its bbox width through OPEN then CLOSE, so the
    # survivors are identifiable one by one rather than only in aggregate.
    assert sorted((c.w for c in components), reverse=True) == biggest


def test_components_under_the_area_floor_are_dropped() -> None:
    """A 4x4 square survives OPEN as a 12 px component and must never reach S3 --
    the boundary band would eat the whole component and the statistic would be nan.

    The blob is deliberately one that morphology does NOT erase: if the tiny square were
    simply opened away, this test would pass with the area floor deleted and would be
    asserting nothing.
    """
    background = _static_background()
    segmenter = Segmenter()
    for _ in range(WARMUP_FRAMES):
        segmenter.apply(background)

    components = segmenter.apply(_blob_frame(background, [40, 4]))

    assert len(components) == 1
    assert all(c.area >= MIN_COMPONENT_AREA_PX for c in components)


def test_a_component_exactly_on_the_area_floor_is_kept() -> None:
    """The floor is ``area < MIN``, not ``<=``. Nothing else in this file separates them.

    An 8x8 square is the smallest that comes through OPEN/CLOSE with an area of exactly
    MIN_COMPONENT_AREA_PX under the pinned OpenCV, and a 7x7 comes through as 45 px. If
    the comparison were ``<=``, the smallest plume the statistic can still measure would
    be deleted at the frame it first became measurable -- which is exactly the frame
    time-to-detection is counted from.
    """
    background = _static_background()

    on_floor = Segmenter()
    for _ in range(WARMUP_FRAMES):
        on_floor.apply(background)
    small_areas = sorted(c.area for c in on_floor.apply(_blob_frame(background, [40, 8])))[:-1]
    assert small_areas == [MIN_COMPONENT_AREA_PX], small_areas

    under_floor = Segmenter()
    for _ in range(WARMUP_FRAMES):
        under_floor.apply(background)
    assert len(under_floor.apply(_blob_frame(background, [40, 7]))) == 1


# ---------------------------------------------------------------------------------
# Failing loudly
# ---------------------------------------------------------------------------------


def test_a_frame_of_a_different_size_raises_rather_than_restarting_the_model() -> None:
    """The silent-wrongness case this module is most exposed to.

    All three subtractors reallocate and start over when handed a different frame size.
    Nothing raises: ``apply`` returns a plausible mask, ``frames_seen`` keeps counting and
    ``warm`` keeps answering True over a model that has in fact seen one frame -- so a
    resized frame anywhere upstream would publish a bench row measured on a background
    model that never converged. A resolution change is a new camera-night and has to be
    an explicit ``reset``.
    """
    background = _static_background()
    segmenter = Segmenter()
    for _ in range(WARMUP_FRAMES + 4):
        segmenter.apply(background)
    assert segmenter.warm

    with pytest.raises(ValueError, match="shape"):
        segmenter.apply(background[:100, :100].copy())

    # The rejection left nothing half-updated: the model is still the one that converged
    # on the original size, and it still accepts that size.
    assert segmenter.frames_seen == WARMUP_FRAMES + 4
    segmenter.apply(background)
    assert segmenter.frames_seen == WARMUP_FRAMES + 5

    # reset() is the documented way through, and it really does clear the latch.
    segmenter.reset()
    segmenter.apply(background[:100, :100].copy())
    assert segmenter.frames_seen == 1
    assert not segmenter.warm


def test_a_rejected_frame_does_not_advance_the_model() -> None:
    """``frames_seen`` drives ``warm``, which drives the UI's warming state. If a rejected
    frame still counted, a camera could report itself warm on frames it never saw."""
    background = _static_background()
    segmenter = Segmenter()
    segmenter.apply(background)

    for bad in (background.astype(np.float32), np.zeros((4, 4, 3), dtype=np.uint8)):
        with pytest.raises(ValueError):
            segmenter.apply(bad)

    assert segmenter.frames_seen == 1


def test_shadow_pixels_never_become_foreground() -> None:
    """MOG2 labels shadows 127 when ``detectShadows`` is on. The mask is compared for
    EQUALITY with 255 so that flipping that constant cannot promote every cloud shadow at
    dawn into a component -- and a shadow is a CONFUSE class, so that promotion would eat
    the alert budget on exactly the mornings the system is judged on.

    Asserted on the binarisation directly, because with the constant False today no frame
    the pipeline sees carries a 127 and an end-to-end test could not tell ``== 255`` from
    ``> 0``.
    """
    raw = np.array([[0, 127, MASK_FOREGROUND_VALUE], [127, 255, 1]], dtype=np.uint8)
    binary = _binarise(raw)

    assert binary.dtype == np.uint8
    assert binary.tolist() == [[0, 0, 255], [0, 255, 0]]


def test_the_bgsegm_noise_floor_is_measured_not_assumed() -> None:
    """The claim DEMO.md publishes: on i.i.d. sensor noise MOG2 is silent and the two
    bgsegm arms are not. This is the noise-floor row of ``bench.py --bg``, asserted here
    so that it is a measurement rather than a sentence -- if a future OpenCV changes the
    ordering, the table is wrong and this fails before it is published.
    """
    foreground: dict[str, int] = {}
    for method in METHODS:
        rng = np.random.RandomState(11)
        segmenter = Segmenter(method)
        total = 0
        for k in range(40):
            components = segmenter.apply(_noise_frame(rng))
            if k >= WARMUP_FRAMES:
                total += sum(c.area for c in components)
        foreground[method] = total

    assert foreground["mog2"] == 0, foreground
    assert foreground["lsbp"] > foreground["mog2"], foreground
    assert foreground["gsoc"] > foreground["mog2"], foreground


def test_unknown_method_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        Segmenter("knn")


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8),
        np.zeros((WORK_HEIGHT, WORK_WIDTH), dtype=np.float32),
        np.zeros((0, 0), dtype=np.uint8),
        [[0, 0], [0, 0]],
    ],
    ids=["colour", "float32", "empty", "not-an-array"],
)
def test_apply_rejects_input_it_cannot_segment(bad: object) -> None:
    """Every one of these would produce a plausible-looking mask if it were coerced, and
    a plausible wrong mask propagates silently into every statistic downstream."""
    with pytest.raises(ValueError):
        Segmenter().apply(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------
# Real imagery
# ---------------------------------------------------------------------------------


def _real_frames(limit: int = 12) -> list[np.ndarray]:
    root = pathlib.Path(__file__).resolve().parents[1] / "spike" / "frames"
    paths = sorted(glob.glob(str(root / "*.jpg")))[:limit]
    if len(paths) < limit:
        pytest.skip(f"HPWREN frames not on disk at {root} (gitignored; run scripts/fetch.py)")

    frames = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        assert bgr is not None, f"unreadable frame {path}"
        small = cv2.resize(bgr, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    return frames


def test_smoke_over_real_hpwren_frames() -> None:
    """Structural invariants only -- these frames carry no ignition clock, so any
    assertion about *which* components appear would be an assertion about nothing."""
    frames = _real_frames()
    segmenter = Segmenter()

    for grey in frames:
        components = segmenter.apply(grey)
        assert len(components) <= MAX_COMPONENTS_PER_FRAME
        for c in components:
            assert c.area >= MIN_COMPONENT_AREA_PX
            assert 0 <= c.x and 0 <= c.y
            assert c.x + c.w <= WORK_WIDTH
            assert c.bottom <= WORK_HEIGHT
            assert c.w > 0 and c.h > 0
            assert c.area <= c.w * c.h  # a component cannot be denser than its own bbox

    assert segmenter.frames_seen == len(frames)
    assert segmenter.warm

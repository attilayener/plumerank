"""Tests for the ~1% learned path.

Two things make this file awkward and both are deliberate. There is no torch in the
runtime lockfile, and there is no committed ONNX file -- ``models/tile_classifier.onnx``
is produced by a training run that takes hours and is not part of ``make test``. So the
tests build a *minimal ONNX model by hand*, as protobuf bytes, and run the real
``cv2.dnn.readNetFromONNX(..., ENGINE_NEW)`` against it.

That is worth the ~60 lines of encoder below rather than mocking ``cv2.dnn``, because the
thing most likely to break in this module is exactly the thing a mock would hide: whether
the engine loads the graph at all. The hand-built net is
``GlobalAveragePool -> Flatten -> Gemm``, whose weights are chosen so that its output is
``3 * mean(R)/255`` -- an analytically known answer, which is what lets the grid geometry
and the invocation count be checked against arithmetic instead of against a golden file.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import cv2

from plumerank.constants import (
    B1_GRID_STRIDE,
    ESCALATION_FRACTION_MAX,
    ESCALATION_HAZARD_MIN,
    ESCALATION_TOP_K,
    FRAMES_PER_WINDOW,
    FUSION_BIAS,
    FUSION_W_S1,
    FUSION_W_S2,
    FUSION_W_S3,
    FUSION_W_S4,
    IGNITION_FRAME_INDEX,
    ONNX_ACCEPT_TOLERANCE,
    ONNX_OPSET,
    POSITIVE_OVERLAP_MIN,
    TILE_MEAN_BGR,
    TILE_PIXEL_SCALE,
    TILE_SIZE,
    TILE_SWAP_RB,
    WORK_HEIGHT,
    WORK_WIDTH,
)
from plumerank.escalate import TileClassifier, _probability, grid_origins, should_escalate

SEED = 20261026


# ---------------------------------------------------------------------------------
# A minimal ONNX file, written as protobuf bytes
# ---------------------------------------------------------------------------------


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _int_field(field: int, value: int) -> bytes:
    return _tag(field, 0) + _varint(value)


def _len_field(field: int, payload: bytes | str) -> bytes:
    raw = payload.encode() if isinstance(payload, str) else payload
    return _tag(field, 2) + _varint(len(raw)) + raw


def _value_info(name: str, dims: tuple[int, ...]) -> bytes:
    shape = b"".join(_len_field(1, _int_field(1, d)) for d in dims)
    tensor_type = _int_field(1, 1) + _len_field(2, shape)  # elem_type 1 == FLOAT
    return _len_field(1, name) + _len_field(2, _len_field(1, tensor_type))


def _initializer(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    body = b"".join(_int_field(1, d) for d in arr.shape)
    return body + _int_field(2, 1) + _len_field(8, name) + _len_field(9, arr.tobytes())


def _node(op: str, ins: tuple[str, ...], outs: tuple[str, ...], name: str) -> bytes:
    return (
        b"".join(_len_field(1, i) for i in ins)
        + b"".join(_len_field(2, o) for o in outs)
        + _len_field(3, name)
        + _len_field(4, op)
    )


def _minimal_onnx(path: pathlib.Path, out_dim: int = 2, weight: float = 3.0) -> pathlib.Path:
    """Write a graph whose last logit is ``weight * mean(R)``, so scores are predictable."""
    w = np.zeros((3, out_dim), dtype=np.float32)
    w[0, out_dim - 1] = weight
    graph = (
        _len_field(1, _node("GlobalAveragePool", ("input",), ("pooled",), "gap"))
        + _len_field(1, _node("Flatten", ("pooled",), ("flat",), "flat"))
        + _len_field(1, _node("Gemm", ("flat", "W", "B"), ("logits",), "fc"))
        + _len_field(2, "plumerank_test_tile_classifier")
        + _len_field(5, _initializer("W", w))
        + _len_field(5, _initializer("B", np.zeros(out_dim, dtype=np.float32)))
        + _len_field(11, _value_info("input", (1, 3, TILE_SIZE, TILE_SIZE)))
        + _len_field(12, _value_info("logits", (1, out_dim)))
    )
    model = (
        _int_field(1, 7)  # ir_version
        + _len_field(2, "plumerank-tests")
        + _len_field(7, graph)
        + _len_field(8, _len_field(1, "") + _int_field(2, 13))  # opset ai.onnx v13
    )
    path.write_bytes(model)
    return path


@pytest.fixture(scope="module")
def onnx_2class(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return _minimal_onnx(tmp_path_factory.mktemp("onnx") / "two.onnx", out_dim=2)


@pytest.fixture(scope="module")
def onnx_1logit(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return _minimal_onnx(tmp_path_factory.mktemp("onnx") / "one.onnx", out_dim=1)


@pytest.fixture(scope="module")
def onnx_3head(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return _minimal_onnx(tmp_path_factory.mktemp("onnx") / "three.onnx", out_dim=3)


@pytest.fixture(scope="module")
def onnx_nan(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A model whose weights have gone NaN -- what a diverged training run exports."""
    return _minimal_onnx(tmp_path_factory.mktemp("onnx") / "nan.onnx", weight=float("nan"))


def _expected(mean_red_0_255: float) -> float:
    """What the hand-built net must return for a tile of that mean red value."""
    return float(1.0 / (1.0 + np.exp(-3.0 * mean_red_0_255 / 255.0)))


# ---------------------------------------------------------------------------------
# Grid geometry -- B1's published invocation count is len(xs) * len(ys)
# ---------------------------------------------------------------------------------


def test_grid_origins_reach_the_far_edge() -> None:
    xs = grid_origins(WORK_WIDTH)
    ys = grid_origins(WORK_HEIGHT)
    assert xs[0] == 0 and ys[0] == 0
    assert xs[-1] == WORK_WIDTH - TILE_SIZE
    assert ys[-1] == WORK_HEIGHT - TILE_SIZE


def test_grid_origins_are_strictly_ascending() -> None:
    origins = grid_origins(WORK_HEIGHT)
    assert all(b > a for a, b in zip(origins, origins[1:]))


def test_grid_origins_leave_no_unscanned_band() -> None:
    # Every column of the frame must be covered by at least one tile, or B1 is a strawman
    # that can miss a plume simply because of where it appeared.
    covered = np.zeros(WORK_WIDTH, dtype=bool)
    for ox in grid_origins(WORK_WIDTH):
        covered[ox : ox + TILE_SIZE] = True
    assert covered.all()


def test_grid_origins_step_by_the_stride_except_at_the_edge() -> None:
    origins = grid_origins(WORK_WIDTH)
    assert all(b - a == B1_GRID_STRIDE for a, b in zip(origins, origins[1:-1]))


def test_grid_origins_rejects_an_axis_shorter_than_a_tile() -> None:
    with pytest.raises(ValueError):
        grid_origins(TILE_SIZE - 1)


def test_grid_origins_rejects_a_nonpositive_stride() -> None:
    with pytest.raises(ValueError):
        grid_origins(WORK_WIDTH, stride=0)


def test_grid_origins_rejects_a_stride_wider_than_a_tile() -> None:
    # The far-edge flush origin only closes the band at the END of the axis. A stride
    # wider than a tile opens bands BETWEEN tiles, which the coverage test above would
    # still pass on the default stride while B1 quietly stopped seeing part of the frame.
    with pytest.raises(ValueError, match="gaps"):
        grid_origins(WORK_WIDTH, stride=TILE_SIZE + 1)


# ---------------------------------------------------------------------------------
# should_escalate
# ---------------------------------------------------------------------------------


def test_should_escalate_needs_both_conditions() -> None:
    assert should_escalate(1, ESCALATION_HAZARD_MIN)
    assert not should_escalate(ESCALATION_TOP_K + 1, 1.0)
    assert not should_escalate(1, ESCALATION_HAZARD_MIN - 1e-9)


def test_should_escalate_is_inclusive_at_both_boundaries() -> None:
    # Both constants are floors, not strict thresholds; a rule that differed from its
    # written form at exactly the boundary is the kind of thing nobody finds later.
    assert should_escalate(ESCALATION_TOP_K, ESCALATION_HAZARD_MIN)


# ---------------------------------------------------------------------------------
# The 99/1 claim, on a simulated camera-night
# ---------------------------------------------------------------------------------


def _hazard(s1: float, s2: float, s3: float, s4: float) -> float:
    logit = FUSION_BIAS + FUSION_W_S1 * s1 + FUSION_W_S2 * s2 + FUSION_W_S3 * s3 + FUSION_W_S4 * s4
    return float(1.0 / (1.0 + np.exp(-logit)))


def _simulated_night(n_cameras: int = 40) -> tuple[int, int, list[int]]:
    """One 40-camera night with one ignition -- the corpus's actual shape.

    Quiet cameras draw their statistics from [0, 0.25], which the fusion weights cannot
    lift past ESCALATION_HAZARD_MIN however they land, so the escalations that happen are
    the ones the ignition causes. Returns (escalations, camera-frames, escalated frames on
    the fire camera).
    """
    rng = np.random.default_rng(SEED)
    escalations = 0
    fire_frames: list[int] = []
    for frame in range(FRAMES_PER_WINDOW):
        hazards = []
        for cam in range(n_cameras):
            s = rng.uniform(0.0, 0.25, 4)
            if cam == 0 and frame >= IGNITION_FRAME_INDEX:
                ramp = min(1.0, (frame - IGNITION_FRAME_INDEX) / 10.0)
                s = np.minimum(1.0, s + 0.9 * ramp)
            hazards.append(_hazard(*s))
        order = sorted(range(n_cameras), key=lambda c: hazards[c], reverse=True)
        for rank, cam in enumerate(order, start=1):
            if should_escalate(rank, hazards[cam]):
                escalations += 1
                if cam == 0:
                    fire_frames.append(frame)
    return escalations, n_cameras * FRAMES_PER_WINDOW, fire_frames


def test_escalation_fraction_under_2pc() -> None:
    escalations, camera_frames, _ = _simulated_night()
    assert escalations / camera_frames <= ESCALATION_FRACTION_MAX


def test_escalation_is_not_vacuous() -> None:
    # A rule that never fires would trivially pass the fraction test and would also mean
    # the learned component never runs -- the failure the fraction test cannot see.
    escalations, _, fire_frames = _simulated_night()
    assert escalations > 0
    assert min(fire_frames) >= IGNITION_FRAME_INDEX
    assert len(fire_frames) > FRAMES_PER_WINDOW // 4


def test_escalation_fraction_has_a_structural_ceiling() -> None:
    # Worst case, every camera pinned at hazard 1.0: the rank condition alone caps the
    # fraction at ESCALATION_TOP_K / n_cameras. That ceiling (5% on a 40-camera wall) is
    # ABOVE ESCALATION_FRACTION_MAX, which is the honest statement: the ~1% is a measured
    # property of the hazard floor and of one ignition per night, not a guarantee of the
    # rule. bench.py publishes the realised number every run for exactly this reason.
    n_cameras = 40
    fired = sum(
        should_escalate(rank, 1.0) for rank in range(1, n_cameras + 1)
    )
    assert fired / n_cameras == ESCALATION_TOP_K / n_cameras


# ---------------------------------------------------------------------------------
# TileClassifier -- against a real cv2.dnn net on ENGINE_NEW
# ---------------------------------------------------------------------------------


def test_missing_model_is_a_hard_stop(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        TileClassifier(tmp_path / "absent.onnx")


def test_loads_under_engine_new(onnx_2class: pathlib.Path) -> None:
    clf = TileClassifier(onnx_2class)
    assert clf.engine == cv2.dnn.ENGINE_NEW
    assert clf.invocations == 0


def test_score_tile_is_exactly_one_forward_pass(onnx_2class: pathlib.Path) -> None:
    clf = TileClassifier(onnx_2class)
    frame = np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    clf.score_tile(frame, (100, 100, 20, 20))
    assert clf.invocations == 1
    clf.score_tile(frame, (100, 100, 20, 20))
    assert clf.invocations == 2


def test_score_tile_matches_the_analytic_answer(onnx_2class: pathlib.Path) -> None:
    clf = TileClassifier(onnx_2class)
    frame = np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    frame[:, :, 2] = 200  # BGR red channel; swapRB makes it channel 0 of the blob
    assert clf.score_tile(frame, (200, 100, 10, 10)) == pytest.approx(_expected(200.0), abs=1e-5)


def test_score_tile_slides_the_window_inside_the_frame(onnx_2class: pathlib.Path) -> None:
    # A component in the top-left corner would centre a tile at negative coordinates. The
    # tile slides rather than pads, so the score must equal the corner tile's own score.
    clf = TileClassifier(onnx_2class)
    rng = np.random.default_rng(SEED)
    frame = rng.integers(0, 256, (WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    corner_mean = float(frame[:TILE_SIZE, :TILE_SIZE, 2].mean())
    assert clf.score_tile(frame, (0, 0, 4, 4)) == pytest.approx(_expected(corner_mean), abs=1e-4)


def test_degenerate_bbox_raises(onnx_2class: pathlib.Path) -> None:
    clf = TileClassifier(onnx_2class)
    frame = np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        clf.score_tile(frame, (10, 10, 0, 5))
    assert clf.invocations == 0


def test_a_frame_smaller_than_a_tile_raises(onnx_2class: pathlib.Path) -> None:
    clf = TileClassifier(onnx_2class)
    small = np.zeros((TILE_SIZE - 1, TILE_SIZE - 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        clf.score_tile(small, (0, 0, 4, 4))
    with pytest.raises(ValueError):
        clf.score_frame_grid(small)


def test_score_frame_grid_counts_every_tile(onnx_2class: pathlib.Path) -> None:
    clf = TileClassifier(onnx_2class)
    frame = np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    clf.score_frame_grid(frame)
    expected = len(grid_origins(WORK_WIDTH)) * len(grid_origins(WORK_HEIGHT))
    assert clf.invocations == expected


def test_score_frame_grid_is_the_max_over_the_grid(onnx_2class: pathlib.Path) -> None:
    clf = TileClassifier(onnx_2class)
    rng = np.random.default_rng(SEED)
    frame = rng.integers(0, 256, (WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    best = max(
        _expected(float(frame[oy : oy + TILE_SIZE, ox : ox + TILE_SIZE, 2].mean()))
        for oy in grid_origins(WORK_HEIGHT)
        for ox in grid_origins(WORK_WIDTH)
    )
    assert clf.score_frame_grid(frame) == pytest.approx(best, abs=1e-4)


def test_both_uses_share_one_invocation_counter(onnx_2class: pathlib.Path) -> None:
    # One counter, not two: the 99/1 arithmetic must not be quotable against whichever
    # denominator flatters it.
    clf = TileClassifier(onnx_2class)
    frame = np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    clf.score_tile(frame, (100, 100, 20, 20))
    grid = clf.invocations
    clf.score_frame_grid(frame)
    assert clf.invocations == grid + len(grid_origins(WORK_WIDTH)) * len(grid_origins(WORK_HEIGHT))


def test_scores_are_probabilities(onnx_2class: pathlib.Path) -> None:
    clf = TileClassifier(onnx_2class)
    rng = np.random.default_rng(SEED)
    frame = rng.integers(0, 256, (WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    assert 0.0 <= clf.score_tile(frame, (240, 130, 30, 30)) <= 1.0
    assert 0.0 <= clf.score_frame_grid(frame) <= 1.0


def test_a_single_logit_head_agrees_with_the_two_class_head(
    onnx_1logit: pathlib.Path, onnx_2class: pathlib.Path
) -> None:
    # sigmoid(x) == softmax([0, x])[1]. Supporting both heads is only defensible if they
    # give the same answer for the same model.
    frame = np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    frame[:, :, 2] = 137
    one = TileClassifier(onnx_1logit).score_tile(frame, (200, 100, 10, 10))
    two = TileClassifier(onnx_2class).score_tile(frame, (200, 100, 10, 10))
    assert one == pytest.approx(two, abs=1e-6)


def test_an_unrecognised_head_raises_rather_than_guessing(onnx_3head: pathlib.Path) -> None:
    clf = TileClassifier(onnx_3head)
    frame = np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        clf.score_tile(frame, (100, 100, 20, 20))


# ---------------------------------------------------------------------------------
# The silent-wrongness cases: a number that looks like an answer but is not one
# ---------------------------------------------------------------------------------


def test_a_nan_model_raises_instead_of_scoring_a_tile(onnx_nan: pathlib.Path) -> None:
    clf = TileClassifier(onnx_nan)
    frame = np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    frame[:, :, 2] = 200
    with pytest.raises(ValueError, match="non-finite"):
        clf.score_tile(frame, (200, 100, 10, 10))


def test_a_nan_model_does_not_report_a_quiet_frame_to_b1(onnx_nan: pathlib.Path) -> None:
    # The expensive one. max() over the grid takes the LARGER of 0.0 and NaN as 0.0, so
    # before the guard a diverged model reported "no smoke anywhere" -- forever, on every
    # frame, with B1's invocation count still ticking up to prove it was working.
    clf = TileClassifier(onnx_nan)
    frame = np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.uint8)
    frame[:, :, 2] = 200
    with pytest.raises(ValueError, match="non-finite"):
        clf.score_frame_grid(frame)


@pytest.mark.parametrize(
    "out",
    [
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([[0.0, np.nan]], dtype=np.float32),
        np.array([[-np.inf, -np.inf]], dtype=np.float32),
    ],
)
def test_a_non_finite_output_never_becomes_a_probability(out: np.ndarray) -> None:
    # The two-logit softmax is the subtle half: shifting by max() turns [-inf, -inf] into
    # nan/nan, which is a division that produces a number-shaped nothing.
    with pytest.raises(ValueError, match="non-finite"):
        _probability(out)


def test_a_float_frame_is_refused_rather_than_scaled_twice(onnx_2class: pathlib.Path) -> None:
    # A caller handing over a normalised float frame gets it multiplied by 1/255 again and
    # reads back a perfectly well-formed probability for an almost-black tile.
    clf = TileClassifier(onnx_2class)
    frame = np.zeros((WORK_HEIGHT, WORK_WIDTH, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        clf.score_tile(frame, (100, 100, 20, 20))
    with pytest.raises(ValueError, match="uint8"):
        clf.score_frame_grid(frame)
    assert clf.invocations == 0


def test_a_single_channel_frame_is_refused(onnx_2class: pathlib.Path) -> None:
    # A greyscale frame slices to a (96, 96) tile whose .shape[0]/.shape[1] pass the tile
    # check, so without this guard the only thing standing between a grey frame and a
    # score is whatever the engine happens to do with a one-channel blob.
    clf = TileClassifier(onnx_2class)
    grey = np.zeros((WORK_HEIGHT, WORK_WIDTH), dtype=np.uint8)
    with pytest.raises(ValueError, match="BGR"):
        clf.score_tile(grey, (100, 100, 20, 20))
    assert clf.invocations == 0


def test_a_frame_exactly_one_tile_wide_costs_exactly_one_invocation(
    onnx_2class: pathlib.Path,
) -> None:
    # The grid geometry's lower boundary: B1's published cost is len(xs)*len(ys), and an
    # off-by-one in grid_origins would make that 2 for a frame that holds one tile.
    clf = TileClassifier(onnx_2class)
    frame = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    clf.score_frame_grid(frame)
    assert clf.invocations == 1


def test_the_preprocessing_convention_is_inside_the_freeze() -> None:
    # Flipping TILE_SWAP_RB changes every score in the project and raises nothing. If it
    # ever moves back out of constants.py, results/constants.sha256 stops covering the one
    # set of numbers that defines what the shipped model actually computes.
    import plumerank.constants as k

    assert (k.TILE_PIXEL_SCALE, k.TILE_MEAN_BGR, k.TILE_SWAP_RB) == (
        TILE_PIXEL_SCALE,
        TILE_MEAN_BGR,
        TILE_SWAP_RB,
    )
    assert TILE_PIXEL_SCALE == 1.0 / 255.0 and TILE_MEAN_BGR == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------------
# scripts/train_tile_classifier.py -- the half of it that needs no torch
# ---------------------------------------------------------------------------------
#
# The script is imported here rather than in its own test file because it is the other
# end of this module: it defines the tiles, and escalate.py consumes them. The tests below
# never import torch (it is not in the runtime lockfile and is not installed), which is
# itself the point -- the module has to stay importable on the server, and one of these
# asserts exactly that.

import sys  # noqa: E402

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import train_tile_classifier as train  # noqa: E402


def test_the_training_script_imports_without_torch() -> None:
    # The server installs no torch (requirements.lock), and check_claims.py has to be able
    # to read this file there. A module-level `import torch` would break both.
    assert not hasattr(train, "torch")


def test_the_training_numbers_come_from_the_frozen_file() -> None:
    # POSITIVE_OVERLAP_MIN decides what the model is taught smoke looks like, and the
    # opset and the acceptance tolerance decide whether the export is allowed to ship. A
    # number that made the model but sits outside constants.sha256 is a number outside the
    # freeze, whatever the module it happens to live in.
    assert train.POSITIVE_OVERLAP_MIN is POSITIVE_OVERLAP_MIN
    assert train.ONNX_OPSET is ONNX_OPSET
    assert train.ONNX_ACCEPT_TOLERANCE is ONNX_ACCEPT_TOLERANCE


def test_a_tile_is_positive_at_the_overlap_rule_and_not_below_it() -> None:
    """The overlap is a fraction of the BOX, not of the tile.

    A box 40 px tall straddling the tile's top edge has 30% of ITSELF inside once 12 px
    are in, and that is a positive: the tile contains most of a small plume. The same
    geometry read tile-relative would need the box to fill 2,765 px of the 96x96 tile.
    """
    tall = 40.0
    inside_at_30pc = tall * POSITIVE_OVERLAP_MIN
    # box hangs above the tile, dipping `inside` pixels into it
    just_over = (0.0, -(tall - inside_at_30pc) - 1e-4, float(TILE_SIZE), inside_at_30pc + 1e-4)
    just_under = (0.0, -(tall - inside_at_30pc) + 1e-2, float(TILE_SIZE), inside_at_30pc - 1e-2)
    assert train.tile_label((0, 0), [just_over]) == "pos"
    assert train.tile_label((0, 0), [just_under]) == "skip"


def test_a_small_box_wholly_inside_a_tile_is_positive() -> None:
    """The regression guard for the defect that made the trained model useless.

    tile_label measured overlap against the TILE's area, so a box had to fill 30% of a
    96x96 tile -- 2,765 px of smoke -- to count. Measured over Pyro-SDIS at this working
    resolution the median smoke box is 15.6 x 11.3 px, about 2% of a tile, and only 2.4%
    of all boxes could EVER reach 30% of one. That reading left 628 usable images of
    29,537 and trained a classifier to val precision 0.024 against a 1.8% base rate -- a
    model that had learned nothing, in the role of baseline B1, which would have made the
    headline comparison a strawman.

    A median-sized plume box sitting entirely inside the tile is 100% overlapped and must
    be a positive.
    """
    median_box = (40.0, 40.0, 55.6, 51.3)          # 15.6 x 11.3 px, the measured median
    assert train.tile_label((0, 0), [median_box]) == "pos"


def test_a_negative_tile_needs_zero_overlap() -> None:
    assert train.tile_label((0, 0), [(500.0, 500.0, 520.0, 520.0)]) == "neg"


def test_an_ambiguous_tile_is_discarded_rather_than_reassigned() -> None:
    # A tile CLIPPING the edge of a plume teaches the net that the smoke boundary is
    # arbitrary -- which is the exact discrimination the escalation stage is asked to make.
    # Box-relative: a 100 px box with only 10 px inside the tile is 10% overlapped, below
    # the rule and above zero, so it is discarded rather than assigned to either class.
    clipped = (-90.0, 0.0, 10.0, 100.0)
    assert train.tile_label((0, 0), [clipped]) == "skip"


def test_an_image_with_no_label_file_is_all_negative(tmp_path: pathlib.Path) -> None:
    # Pyro-SDIS omits the label file for a frame with no smoke, so a missing file is data,
    # not an error -- but a MALFORMED one is an error, which the next test asserts.
    assert train.parse_yolo_labels(tmp_path / "absent.txt", WORK_WIDTH, WORK_HEIGHT) == []


def test_yolo_labels_are_parsed_in_pixels(tmp_path: pathlib.Path) -> None:
    label = tmp_path / "frame.txt"
    label.write_text("0 0.5 0.5 0.25 0.5\n")
    (x0, y0, x1, y1) = train.parse_yolo_labels(label, WORK_WIDTH, WORK_HEIGHT)[0]
    assert (x0, x1) == pytest.approx((WORK_WIDTH * 0.375, WORK_WIDTH * 0.625))
    assert (y0, y1) == pytest.approx((WORK_HEIGHT * 0.25, WORK_HEIGHT * 0.75))


def test_a_malformed_label_line_raises(tmp_path: pathlib.Path) -> None:
    # A silently dropped box becomes a mislabelled negative tile, which looks like noise.
    label = tmp_path / "frame.txt"
    label.write_text("0 0.5 0.5\n")
    with pytest.raises(ValueError):
        train.parse_yolo_labels(label, WORK_WIDTH, WORK_HEIGHT)


def test_training_preprocessing_is_identical_to_blobFromImage() -> None:
    # The single highest-value assertion in this file. If these two paths drift, the
    # exported model scores tiles it was never trained on and nothing else in the suite
    # would notice -- the ONNX would still load and still return plausible probabilities.
    tiles, _, _ = train.build_synthetic_corpus(4, np.random.default_rng(SEED))
    for tile in tiles:
        blob = cv2.dnn.blobFromImage(
            tile,
            scalefactor=TILE_PIXEL_SCALE,
            size=(TILE_SIZE, TILE_SIZE),
            mean=TILE_MEAN_BGR,
            swapRB=TILE_SWAP_RB,
            crop=False,
        )
        assert np.abs(train.preprocess(tile[None]) - blob).max() == 0.0


def test_the_synthetic_corpus_is_balanced_and_tile_shaped() -> None:
    tiles, labels, sources = train.build_synthetic_corpus(16, np.random.default_rng(SEED))
    assert tiles.shape == (16, TILE_SIZE, TILE_SIZE, 3)
    assert int((labels == 1).sum()) == int((labels == 0).sum()) == 8
    assert all(s.startswith("synthetic://throwaway/") for s in sources)


def test_the_synthetic_model_cannot_overwrite_the_real_one() -> None:
    # A throwaway plumbing model landing on models/tile_classifier.onnx would be the
    # quietest possible way to ship an unevaluated model.
    assert train.SYNTHETIC_MODEL_OUT != train.DEFAULT_MODEL_OUT


def test_a_figlib_path_in_the_corpus_stops_the_run() -> None:
    with pytest.raises(SystemExit):
        train.assert_no_figlib(["data/corpus/figlib/20200611_skyline/1591..jpg"])
    with pytest.raises(SystemExit):
        train.assert_no_figlib(["/tmp/hpwren/frame.jpg"])


def test_the_manifest_records_the_sources_and_carries_no_figlib_path(
    tmp_path: pathlib.Path,
) -> None:
    import json

    sources = ["data/pyrosdis/images/train/a.jpg", "data/pyrosdis/images/train/b.jpg"]
    out = tmp_path / "tile_manifest.json"
    train.write_manifest(
        out,
        sources,
        {"source_images": 2},
        tmp_path / "model.onnx",
        corpus="pyro-sdis",
        throwaway=False,
        val={"precision": 0.9, "recall": 0.8, "n_tiles": 10},
    )
    manifest = json.loads(out.read_text())
    assert manifest["sources"] == sources
    assert manifest["corpus"] == "pyro-sdis" and manifest["throwaway"] is False
    assert manifest["positive_overlap_min"] == POSITIVE_OVERLAP_MIN
    blob = json.dumps(manifest).lower()
    assert not any(marker in blob for marker in train.FIGLIB_MARKERS)


def test_a_missing_pyrosdis_corpus_names_the_fetch_command(tmp_path: pathlib.Path) -> None:
    # The default path is the real one, so its absence must point at the fix rather than
    # tempting the operator into --synthetic.
    with pytest.raises(FileNotFoundError, match="fetch.py --corpus pyrosdis"):
        train.build_pyrosdis_corpus(tmp_path, "train", np.random.default_rng(SEED))


def _write_frame(root: pathlib.Path, name: str, with_label: bool) -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(root / "images" / "train" / f"{name}.jpg"),
        np.full((WORK_HEIGHT, WORK_WIDTH, 3), 120, dtype=np.uint8),
    )
    if with_label:
        (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "train" / f"{name}.txt").write_text("0 0.5 0.5 0.4 0.6\n")


def test_a_missing_label_directory_stops_the_run(tmp_path: pathlib.Path) -> None:
    # The nastiest failure in this script. parse_yolo_labels reads a MISSING label file as
    # "this frame has no smoke" -- Pyro-SDIS's own convention -- so a truncated fetch that
    # dropped labels/ trains an all-negative corpus. The loss falls, the ONNX exports, the
    # acceptance gate passes, and the model answers "no smoke" to every tile forever.
    _write_frame(tmp_path, "a", with_label=False)
    with pytest.raises(FileNotFoundError, match="labels"):
        train.build_pyrosdis_corpus(tmp_path, "train", np.random.default_rng(SEED))


def test_a_corpus_with_no_positive_tiles_stops_the_run(tmp_path: pathlib.Path) -> None:
    _write_frame(tmp_path, "a", with_label=True)
    (tmp_path / "labels" / "train" / "a.txt").write_text("")  # boxes parsed away to none
    with pytest.raises(RuntimeError, match="not one positive"):
        train.build_pyrosdis_corpus(tmp_path, "train", np.random.default_rng(SEED))


def test_a_real_corpus_yields_both_classes(tmp_path: pathlib.Path) -> None:
    # The counterpart to the two guards above: they must not fire on a healthy corpus.
    _write_frame(tmp_path, "a", with_label=True)
    tiles, labels, sources = train.build_pyrosdis_corpus(
        tmp_path, "train", np.random.default_rng(SEED)
    )
    assert tiles.shape[1:] == (TILE_SIZE, TILE_SIZE, 3)
    assert 1 in labels and 0 in labels
    assert len(sources) == 1


@pytest.mark.parametrize(
    "pred, truth, precision, recall",
    [
        ([1, 1, 0, 0], [1, 0, 1, 0], 0.5, 0.5),
        ([0, 0, 0, 0], [1, 0, 1, 0], None, 0.0),
        ([1, 1, 1, 1], [0, 0, 0, 0], 0.0, None),
    ],
)
def test_unmeasurable_precision_or_recall_is_none_not_zero(
    pred: list[int], truth: list[int], precision: float | None, recall: float | None
) -> None:
    # 0.0 for "the model predicted nothing" is a measurement-shaped value for a quantity
    # nobody measured, and val precision/recall is published in the manifest and REPORT.md.
    got = train.precision_recall(np.asarray(pred), np.asarray(truth))
    assert got["precision"] == precision and got["recall"] == recall
    assert got["n_tiles"] == 4


def test_a_tile_wholly_inside_a_large_plume_is_positive() -> None:
    """The other half of the overlap rule, and the reason it is measured both ways.

    A near plume is bigger than a 96x96 tile. A tile sitting wholly inside a 192 x 162 px
    box covers 29.6% of that box -- under the rule -- while being nothing but smoke. Judged
    box-relative alone it would be discarded; judged tile-relative it is 100% covered.
    """
    big = (100.0, 50.0, 292.0, 212.0)            # 192 x 162 px, a near plume
    assert train.tile_label((140, 80), [big]) == "pos"

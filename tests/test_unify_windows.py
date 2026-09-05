"""bench.unify_windows -- the fix for the defect that kept DEV and HOLD from ever running.

``data/manifest_dev.json`` declares three window shapes: (80 frames, t0 39), (80, t0 40) and
(81, t0 40). ``replay_arm`` needs one shape, so both fire arms exited on the manifest before
reading a pixel, and every number the project had published came from CONFUSE or micro. These
tests pin the two properties that make the trim a fix rather than a truncation: one common
length, and one common ignition index.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench  # noqa: E402


def W(seq, n, t0):
    return bench.Window(
        sequence_id=seq,
        camera_id=seq.split("_")[-1],
        t0_index=t0,
        frame_paths=tuple(pathlib.Path(f"{seq}/{i}.jpg") for i in range(n)),
    )


DEV_SHAPES = [(80, 39), (80, 40), (81, 40)]


def test_real_dev_shapes_reconcile_to_one_length_and_one_ignition():
    """The exact three shapes DEV declares. 79 frames with t0 at 39 is the largest window
    all three can supply, and the arithmetic is worth pinning: 80-40+39 = 79 is the binding
    constraint, from the (80, t0 40) rows."""
    arm = [W(f"s{i}", n, t0) for i, (n, t0) in enumerate(DEV_SHAPES)]
    (out,) = bench.unify_windows(arm)
    assert {len(w) for w in out} == {79}
    assert {w.t0_index for w in out} == {39}


def test_every_window_keeps_its_own_ignition_frame():
    """The point of anchoring rather than truncating: after the trim, index 39 of every
    window must still be the frame that WAS that window's t0 -- not merely some frame at a
    common offset. A trim that satisfies the length check while sliding the ignition clock
    would corrupt every time-to-detection the project publishes."""
    arm = [W(f"s{i}", n, t0) for i, (n, t0) in enumerate(DEV_SHAPES)]
    before = {w.sequence_id: w.frame_paths[w.t0_index] for w in arm}
    (out,) = bench.unify_windows(arm)
    for w in out:
        assert w.frame_paths[w.t0_index] == before[w.sequence_id]


def test_calibration_is_trimmed_to_the_same_night_as_the_measured_arm():
    """The budget is solved on CONFUSE and spent on DEV. If a CONFUSE camera-night were 81
    frames and a DEV one 79, the threshold would be calibrated against 2.5% more exposure
    than it is spent against, and PlumeRank would take a small unearned advantage on every
    fire arm."""
    arm = [W(f"d{i}", n, t0) for i, (n, t0) in enumerate(DEV_SHAPES)]
    calib = [W(f"c{i}", 81, None) for i in range(4)]
    out_arm, out_calib = bench.unify_windows(arm, calib)
    assert {len(w) for w in out_arm} == {len(w) for w in out_calib} == {79}


def test_non_fire_windows_are_centre_trimmed():
    """No ignition to anchor to, so take the middle: preferring the start of a night to its
    end would be an unmotivated choice that quietly favours whatever happens early."""
    arm = [W("f", 80, 40), W("g", 80, 40)]   # both the same shape -> common length 80
    calib = [W("c", 85, None)]
    out_arm, out_calib = bench.unify_windows(arm, calib)
    assert {len(w) for w in out_arm} == {80}
    w = out_calib[0]
    assert len(w) == 80
    assert w.frame_paths[0] == pathlib.Path("c/2.jpg")   # (85 - 80) // 2 == 2
    assert w.frame_paths[-1] == pathlib.Path("c/81.jpg")  # symmetric: 2 dropped, 3 dropped


def test_uniform_input_is_left_alone():
    """micro is uniform 31 frames with t0 at 8, and `make verify` asserts its metrics
    byte-for-byte against a committed file. If unification is not a no-op here, that
    contract breaks and the judge-facing three-minute command starts failing."""
    arm = [W(f"m{i}", 31, 8 if i < 3 else None) for i in range(6)]
    (out,) = bench.unify_windows(arm)
    assert all(len(w) == 31 for w in out)
    assert [w.frame_paths for w in out] == [w.frame_paths for w in arm]
    assert {w.t0_index for w in out if w.t0_index is not None} == {8}


def test_confuse_alone_is_unchanged():
    """The published CONFUSE numbers were computed at 81 frames per window. Replayed as its
    own arm it must stay 81, or those results silently stop being comparable."""
    arm = [W(f"c{i}", 81, None) for i in range(33)]
    (out,) = bench.unify_windows(arm)
    assert all(len(w) == 81 for w in out)


def test_refuses_rather_than_returns_a_degenerate_window():
    arm = [W("a", 3, 0), W("b", 3, 2)]
    with pytest.raises(SystemExit):
        bench.unify_windows(arm)


def test_empty_input_is_tolerated():
    assert bench.unify_windows([]) == ([],)

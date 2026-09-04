"""Tests for the script that produces the number the submission is scored on.

Everything here runs against a **synthetic arm built in a tmp dir** -- a handful of
16-frame windows, half of them an anchored growing wedge and half a translating blob --
so the whole file runs offline in seconds with no corpus, no network and no credentials.
The synthetic arm is not a claim about accuracy and nothing here asserts that PlumeRank
beats B1 on it; a fixture that decided the headline would be a fixture that could be
tuned until it did. What is asserted is everything a judge can check *mechanically*: the
two labelled blocks and their exact labels, the non-zero exit on a verify mismatch, the
HOLD refusal, one log line per HOLD read, that ``--sensitivity`` re-scores rather than
replays, and that the ablation CSV is written per cell and resumes.

The ONNX is built by hand as protobuf bytes, for the reason ``test_escalate.py`` gives:
there is no torch in the runtime lockfile and no committed model, and mocking ``cv2.dnn``
would hide the one failure that matters -- whether the engine loads the graph at all.
B1 is the same model object as the escalation path, exactly as in production.
"""

from __future__ import annotations

import csv
import json
import pathlib
import shutil
import sys
from typing import Any, Callable

import cv2
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench  # noqa: E402
from plumerank.constants import WORK_HEIGHT, WORK_WIDTH  # noqa: E402

N_FRAMES = 16
T0 = 6
SHA = "0123456789abcdef0123456789abcdef01234567"


# ---------------------------------------------------------------------------------
# A minimal ONNX file, written as protobuf bytes (see the module docstring)
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
    return _len_field(1, name) + _len_field(
        2, _len_field(1, _int_field(1, 1) + _len_field(2, shape))
    )


def _initializer(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return (
        b"".join(_int_field(1, d) for d in arr.shape)
        + _int_field(2, 1)
        + _len_field(8, name)
        + _len_field(9, arr.tobytes())
    )


def _node(op: str, ins: tuple[str, ...], outs: tuple[str, ...], name: str) -> bytes:
    return (
        b"".join(_len_field(1, i) for i in ins)
        + b"".join(_len_field(2, o) for o in outs)
        + _len_field(3, name)
        + _len_field(4, op)
    )


def _write_onnx(path: pathlib.Path) -> pathlib.Path:
    """GlobalAveragePool -> Flatten -> Gemm, whose last logit is ``3 * mean(R)/255``."""
    w = np.zeros((3, 2), dtype=np.float32)
    w[0, 1] = 3.0
    graph = (
        _len_field(1, _node("GlobalAveragePool", ("input",), ("pooled",), "gap"))
        + _len_field(1, _node("Flatten", ("pooled",), ("flat",), "flat"))
        + _len_field(1, _node("Gemm", ("flat", "W", "B"), ("logits",), "fc"))
        + _len_field(2, "plumerank_bench_test_tile_classifier")
        + _len_field(5, _initializer("W", w))
        + _len_field(5, _initializer("B", np.zeros(2, dtype=np.float32)))
        + _len_field(11, _value_info("input", (1, 3, 96, 96)))
        + _len_field(12, _value_info("logits", (1, 2)))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        _int_field(1, 7)
        + _len_field(2, "plumerank-tests")
        + _len_field(7, graph)
        + _len_field(8, _len_field(1, "") + _int_field(2, 13))
    )
    return path


# ---------------------------------------------------------------------------------
# The synthetic corpus
# ---------------------------------------------------------------------------------


def _background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(
        rng.integers(30, 90, (WORK_HEIGHT, WORK_WIDTH), dtype=np.uint8), (9, 9), 0
    )


def _fire_window(seed: int) -> list[np.ndarray]:
    """An anchored plume: the foot stays on one row while the top climbs -- S1's claim."""
    base = _background(seed)
    rng = np.random.default_rng(seed + 1)
    frames = []
    for i in range(N_FRAMES):
        f = base.copy()
        if i >= T0:
            g = i - T0
            cv2.ellipse(f, (240, 200), (10 + 2 * g, 10 + 7 * g), 0, 180, 360, 205, -1)
            f = cv2.GaussianBlur(f, (11, 11), 0)  # a soft edge, which is S3's claim
        noisy = np.clip(f.astype(np.int16) + rng.normal(0, 2, f.shape), 0, 255)
        frames.append(cv2.cvtColor(noisy.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return frames


def _cloud_window(seed: int) -> list[np.ndarray]:
    """A hard-edged blob that translates: the false-alarm shape the budget is spent on."""
    base = _background(seed)
    rng = np.random.default_rng(seed + 2)
    frames = []
    for i in range(N_FRAMES):
        f = base.copy()
        cv2.circle(f, ((60 + 12 * i) % (WORK_WIDTH - 60) + 30, 90), 22, 190, -1)
        noisy = np.clip(f.astype(np.int16) + rng.normal(0, 2, f.shape), 0, 255)
        frames.append(cv2.cvtColor(noisy.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return frames


def _write_window(dest: pathlib.Path, sequence_id: str, frames: list[np.ndarray]) -> list[str]:
    folder = dest / sequence_id
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for i, frame in enumerate(frames):
        # FIgLib's own naming -- epoch underscore signed offset -- so the fixture exercises
        # the same manifest shape data/manifest_dev.json has rather than a tidier one.
        name = f"{1700000000 + i * 60}_{i - T0:+06d}.jpg"
        cv2.imwrite(str(folder / name), frame)
        names.append(name)
    return names


def _arm(dest: pathlib.Path, prefix: str, n: int, fire: bool) -> list[dict[str, Any]]:
    out = []
    for k in range(n):
        sequence_id = f"{prefix}{k}"
        maker = _fire_window if fire else _cloud_window
        names = _write_window(dest, sequence_id, maker(1000 + 37 * k + len(prefix)))
        out.append(
            {
                "sequence_id": sequence_id,
                "camera_id": f"cam-{prefix}{k}",
                "t0_index": T0 if fire else None,
                "n_frames": N_FRAMES,
                "frames": names,
            }
        )
    return out


def _manifest(path: pathlib.Path, windows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"arm": path.stem, "n_windows": len(windows), "windows": windows}),
        encoding="utf-8",
    )


@pytest.fixture(scope="session")
def template(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """One synthetic tree, built once. Tests that mutate it get a copy."""
    root = tmp_path_factory.mktemp("plumerank-tree")
    corpus = root / "data" / "corpus" / "figlib"
    micro = root / "data" / "micro"
    (root / "results").mkdir(parents=True, exist_ok=True)
    _write_onnx(root / "models" / "tile_classifier.onnx")

    _manifest(root / "data" / "manifest_dev.json", _arm(corpus, "dev", 2, True))
    _manifest(root / "data" / "manifest_hold.json", _arm(corpus, "hold", 2, True))
    _manifest(root / "data" / "manifest_confuse.json", _arm(corpus, "conf", 2, False))
    _manifest(
        micro / "manifest_micro.json",
        _arm(micro, "microfire", 2, True) + _arm(micro, "microcloud", 2, False),
    )
    return root


@pytest.fixture
def tree(template: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "tree"
    shutil.copytree(template, root)
    return root


@pytest.fixture
def commit(monkeypatch: pytest.MonkeyPatch) -> str:
    """A fixed HEAD sha. bench refuses to read HOLD outside a checkout, and a tmp dir is
    not one -- patching the lookup is how the refusal stays testable without a git write."""
    monkeypatch.setattr(bench, "git_commit", lambda root: SHA)
    return SHA


def _freeze(root: pathlib.Path) -> None:
    from plumerank import constants as K

    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "constants.sha256").write_text(K.freeze_hash() + "\n", encoding="utf-8")


def _run(root: pathlib.Path, *argv: str) -> int:
    return bench.main([*argv, "--root", str(root)])


def _counting(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace replay_arm with a wrapper that counts pixel-reading replays."""
    calls = [0]
    real: Callable[..., Any] = bench.replay_arm

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(bench, "replay_arm", counted)
    return calls


def _bench_json(root: pathlib.Path, arm: str = "dev") -> dict[str, Any]:
    """One arm's published payload, from wherever that arm is allowed to write.

    Addressed through ``bench.results_filename`` rather than by a literal, because which
    arm owns ``results/bench_<date>.json`` is itself a claim this file has to hold the
    script to -- see test_only_the_hold_arm_may_write_the_published_headline_name.
    """
    name = bench.results_filename(arm, bench._today())
    return json.loads((root / "results" / name).read_text())


# ---------------------------------------------------------------------------------
# The two labelled blocks -- the most attackable claim in the project
# ---------------------------------------------------------------------------------


def test_micro_prints_exactly_two_labelled_blocks(
    tree: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tree, "--arm", "micro") == 0
    out = capsys.readouterr().out
    assert out.count(bench.BLOCK_1) == 1, "block 1 must appear exactly once"
    assert out.count(bench.BLOCK_2) == 1, "block 2 must appear exactly once"
    assert out.index(bench.BLOCK_1) < out.index(bench.BLOCK_2), "reproduced first, quoted second"
    assert bench.REPRODUCE_COMMAND in out


def test_micro_writes_the_expectation_when_it_is_absent_and_says_so(
    tree: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing expectation is not a pass and must not read like one."""
    assert not (tree / "results" / "verify_micro.json").exists()
    assert _run(tree, "--arm", "micro") == 0
    out = capsys.readouterr().out
    assert "WRITTEN, NOT VERIFIED" in out
    assert "Nothing verified it" in out
    assert (tree / "results" / "verify_micro.json").is_file()


def test_micro_matches_byte_for_byte_on_a_second_run(
    tree: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Determinism and the assertion in one: the second run is the check the first wrote."""
    assert _run(tree, "--arm", "micro") == 0
    capsys.readouterr()
    assert _run(tree, "--arm", "micro") == 0
    assert "MATCH (byte-for-byte)" in capsys.readouterr().out


def test_a_verify_micro_mismatch_exits_non_zero(
    tree: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tree, "--arm", "micro") == 0
    capsys.readouterr()
    expected = tree / "results" / "verify_micro.json"
    blob = json.loads(expected.read_text())
    blob["escalation_fraction"] = 0.99
    expected.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert _run(tree, "--arm", "micro") == 1
    captured = capsys.readouterr()
    assert "MISMATCH" in captured.out
    assert "escalation_fraction" in captured.out, "the differing line must be named"


def test_micro_never_says_the_short_command_reproduces_the_headline(
    tree: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The R8 trap, asserted rather than promised.

    A judge who runs the 3-minute command, prints a p50 that is not the README's p50 and
    concludes the headline does not reproduce is the worst outcome available here, so no
    line of this output may put the short command next to a claim about the HOLD number.
    """
    _freeze(tree)
    assert _run(tree, "--arm", "micro") == 0
    out = capsys.readouterr().out
    published = out[out.index(bench.BLOCK_2) :]
    assert "not recomputed here" in published.lower()
    assert "make fetch && make bench" in published
    for line in out.splitlines():
        low = line.lower()
        if "make verify" in low or "verify_micro" in low:
            assert "reproduc" not in low or "NOT VERIFIED" in line or "REPRODUCED HERE" in line


def test_micro_does_not_overwrite_the_published_headline_file(
    tree: pathlib.Path, commit: str
) -> None:
    """``make verify`` must be safe to run on a box that holds the published result."""
    _freeze(tree)
    assert _run(tree, "--arm", "hold") == 0
    published = tree / "results" / f"bench_{bench._today()}.json"
    before = published.read_bytes()
    assert _run(tree, "--arm", "micro") == 0
    assert published.read_bytes() == before


def test_the_published_block_is_quoted_not_recomputed(
    tree: pathlib.Path, commit: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _freeze(tree)
    assert _run(tree, "--arm", "hold") == 0
    stored = _bench_json(tree, "hold")["headline_block"]
    capsys.readouterr()
    assert _run(tree, "--arm", "micro") == 0
    published = capsys.readouterr().out.split(bench.BLOCK_2)[1]
    for line in stored.splitlines():
        assert line in published, f"block 2 must quote the committed file verbatim: {line!r}"


# ---------------------------------------------------------------------------------
# HOLD discipline
# ---------------------------------------------------------------------------------


def test_hold_refuses_to_run_without_the_frozen_constants(
    tree: pathlib.Path, commit: str
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(tree, "--arm", "hold")
    assert "constants.sha256" in str(excinfo.value)
    assert not (tree / "results" / "hold_access.log").exists(), "a refused run reads nothing"


def test_hold_refuses_when_constants_moved_since_the_freeze(
    tree: pathlib.Path, commit: str
) -> None:
    (tree / "results").mkdir(parents=True, exist_ok=True)
    (tree / "results" / "constants.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        _run(tree, "--arm", "hold")
    assert "has changed since the freeze" in str(excinfo.value)
    assert not (tree / "results" / "hold_access.log").exists()


def test_each_hold_read_appends_exactly_one_line(tree: pathlib.Path, commit: str) -> None:
    _freeze(tree)
    log = tree / "results" / "hold_access.log"
    for expected in (1, 2):
        assert _run(tree, "--arm", "hold") == 0
        lines = log.read_text(encoding="utf-8").splitlines()
        assert len(lines) == expected, "one line per read of HOLD pixels, not one per window"
        stamp, sha, rest = lines[-1].split(" ", 2)
        assert stamp.startswith("20") and "T" in stamp, "the timestamp must be ISO 8601"
        assert sha == SHA
        assert "arm=hold" in rest


def test_arms_that_are_not_hold_never_touch_the_log(tree: pathlib.Path, commit: str) -> None:
    _freeze(tree)
    assert _run(tree, "--arm", "dev") == 0
    assert _run(tree, "--arm", "confuse") == 0
    assert not (tree / "results" / "hold_access.log").exists()


# ---------------------------------------------------------------------------------
# --sensitivity re-scores, it does not replay
# ---------------------------------------------------------------------------------


def test_sensitivity_does_not_re_run_the_pipeline(
    tree: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eleven offsets must cost zero extra pixel reads.

    The offset moves the ground-truth clock, not the detector, so re-running the pipeline
    for it would not merely be slow -- it would allow the sensitivity rows to come from a
    different replay than the headline and nothing in the output would say so.
    """
    calls = _counting(monkeypatch)
    assert _run(tree, "--arm", "dev") == 0
    plain = calls[0]

    calls[0] = 0
    assert _run(tree, "--arm", "dev", "--sensitivity") == 0
    assert calls[0] == plain == 2, "one replay of the arm, one of the confusion set, and no more"


def test_sensitivity_writes_eleven_rows(tree: pathlib.Path) -> None:
    assert _run(tree, "--arm", "dev", "--sensitivity") == 0
    path = tree / "results" / f"sensitivity_dev_{bench._today()}.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["offset_minutes"]) for r in rows] == list(range(-5, 6))
    assert _bench_json(tree)["sensitivity"]["rows"][0]["offset_minutes"] == -5


def test_the_offset_shifts_time_to_detection_by_exactly_the_offset(tree: pathlib.Path) -> None:
    """The sensitivity strip is only meaningful if the clock is what moved.

    A one-minute later ignition clock must make a detection read one minute earlier, for
    every system that detected at all. If the detector output moved too, the strip would
    be measuring the pipeline's noise instead of the claim's robustness.
    """
    assert _run(tree, "--arm", "dev", "--sensitivity") == 0
    rows = {r["offset_minutes"]: r for r in _bench_json(tree)["sensitivity"]["rows"]}
    compared = 0
    for system in ("plumerank", "b1_cnn_grid", "b2_frame_diff"):
        base = rows[0][f"{system}_p50_minutes"]
        for offset in (-3, -1, 1, 3):
            shifted = rows[offset][f"{system}_p50_minutes"]
            # A missed percentile is None on both sides and carries no clock to shift.
            if base is not None and shifted is not None:
                assert shifted == pytest.approx(base - offset)
                compared += 1
    # Without this the test passes on a corpus where every system missed everything, which
    # is the one corpus on which it proves nothing at all.
    assert compared >= 8, f"only {compared} offsets were actually compared"


# ---------------------------------------------------------------------------------
# The ablation grid: written per cell, resumable
# ---------------------------------------------------------------------------------


def _ablation_kwargs(tree: pathlib.Path, replay: Callable[..., Any]) -> dict[str, Any]:
    return {
        "budget": 2.0,
        "onnx": tree / "models" / "tile_classifier.onnx",
        "excluded": (),
        "replay": replay,
        "seed": 20261026,
    }


def test_ablation_writes_each_cell_as_it_finishes(
    tree: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted run must leave the finished cells on disk, not in a lost buffer."""
    csv_path = tree / "results" / "ablation.csv"
    real = bench.replay_arm
    calls = [0]

    def failing(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] > 4:  # two cells are two replays each: fail at the start of the third
            raise KeyboardInterrupt("simulated SSH drop")
        return real(*args, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        bench.run_ablation(tree, csv_path, **_ablation_kwargs(tree, failing))

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2, "each finished cell is flushed before the next one starts"
    assert (rows[0]["bg"], rows[0]["s4"], rows[0]["escalation"]) == ("mog2", "True", "True")


def test_ablation_resumes_from_the_csv_it_already_wrote(
    tree: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eleven cells already on disk means one cell of work, and no rewritten rows."""
    csv_path = tree / "results" / "ablation.csv"
    cells = bench.ablation_cells()
    assert len(cells) == 12, "three background models x S4 on/off x escalation on/off"

    for bg, s4, esc in cells[1:]:
        bench._append_csv_row(
            csv_path,
            bench.ABLATION_COLUMNS,
            {c: "" for c in bench.ABLATION_COLUMNS}
            | {"bg": bg, "s4": str(s4), "escalation": str(esc), "p50_minutes": "99.000"},
        )
    before = csv_path.read_text(encoding="utf-8")

    calls = _counting(monkeypatch)
    rows = bench.run_ablation(tree, csv_path, **_ablation_kwargs(tree, bench.replay_arm))

    assert calls[0] == 2, "only the one missing cell may be replayed"
    after = csv_path.read_text(encoding="utf-8")
    assert after.startswith(before), "resumed rows must be left byte-identical"
    assert len(after.splitlines()) == 13, "header plus twelve cells"
    assert len(rows) == 12
    assert {(r["bg"], r["s4"], r["escalation"]) for r in rows} == {
        (bg, str(s4), str(esc)) for bg, s4, esc in cells
    }


# ---------------------------------------------------------------------------------
# The published payload
# ---------------------------------------------------------------------------------


def test_the_json_carries_model_invocations_per_camera_night_for_both_arms(
    tree: pathlib.Path,
) -> None:
    """What makes the 99/1 cost claim measured rather than argued."""
    assert _run(tree, "--arm", "dev") == 0
    payload = _bench_json(tree)
    calls = payload["model_invocations_per_camera_night"]
    assert set(calls) == {"plumerank", "b1_cnn_grid"}
    assert calls["b1_cnn_grid"] > 0, "B1 runs the model on every frame"
    assert calls["plumerank"] < calls["b1_cnn_grid"]
    assert payload["escalation"]["frames_total"] > 0
    assert 0.0 <= payload["escalation"]["fraction"] <= 1.0


def test_the_payload_carries_every_section_the_contract_names(tree: pathlib.Path) -> None:
    assert _run(tree, "--arm", "dev", "--sensitivity") == 0
    payload = _bench_json(tree)
    for key in (
        "headline",
        "baselines",
        "sensitivity",
        "operating_curve",
        "ablation",
        "escalation",
        "model_invocations_per_camera_night",
        "exclusions",
        "run",
        "headline_block",
    ):
        assert key in payload, f"missing published section {key!r}"
    for key in ("commit", "instance", "wall_clock_seconds", "date"):
        assert key in payload["run"]
    assert {r["system"] for r in payload["baselines"]} == set(bench.SYSTEMS)


def test_the_printed_block_is_the_block_stored_in_the_json(
    tree: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One string, printed and stored, so /api/benchmark cannot serve a fourth version."""
    assert _run(tree, "--arm", "dev") == 0
    printed = capsys.readouterr().out
    stored = _bench_json(tree)["headline_block"]
    assert stored in printed


def test_all_three_systems_are_calibrated_to_the_same_budget_on_the_same_set(
    tree: pathlib.Path,
) -> None:
    """The methodological contribution, asserted.

    Three different thresholds are expected -- the systems produce hazards on different
    scales -- but every one of them has to have been solved to the SAME target on the
    SAME confusion set, and the realised rate has to be measured rather than assumed.
    """
    assert _run(tree, "--arm", "dev", "--budget", "2.0") == 0
    payload = _bench_json(tree)
    assert payload["budget_alerts_per_camera_night"] == 2.0
    for row in payload["baselines"]:
        assert row["calibration_realised_rate_uncapped"] <= 2.0 + 1e-4, row["system"]
    assert payload["n_calibration_windows"] > 0


def test_the_published_rate_is_measured_not_taken_from_the_solver(
    tree: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A threshold whose realised rate is not the target must not be published as it.

    ``solve_threshold`` returns its high bound when the target is unreachable, so a script
    that printed the target back would publish a fabricated operating point off a bound.
    Forcing the bound here must stop the run, not decorate it.
    """
    monkeypatch.setattr(bench.rk, "solve_threshold", lambda *a, **k: 0.0)
    with pytest.raises(SystemExit) as excinfo:
        _run(tree, "--arm", "dev", "--budget", "0.001")
    assert "fabrication" in str(excinfo.value)


def test_the_operating_curve_covers_the_requested_budget(tree: pathlib.Path) -> None:
    assert _run(tree, "--arm", "dev", "--budget", "3.0") == 0
    budgets = [r["budget_alerts_per_camera_night"] for r in _bench_json(tree)["operating_curve"]]
    assert 3.0 in budgets, "the published operating point must be a point on the curve"
    assert budgets == sorted(set(budgets))


def test_every_background_model_runs(tree: pathlib.Path) -> None:
    """LSBP and GSOC are measured alternatives, not decoration -- so they must run here."""
    for bg in bench.BACKGROUNDS:
        assert _run(tree, "--arm", "dev", "--bg", bg) == 0
        assert _bench_json(tree)["background"] == bg


def test_a_missing_arm_fails_loudly(tree: pathlib.Path) -> None:
    (tree / "data" / "manifest_dev.json").unlink()
    with pytest.raises(SystemExit) as excinfo:
        _run(tree, "--arm", "dev")
    assert "manifest_dev.json" in str(excinfo.value)


def test_a_missing_frame_fails_before_the_replay_spends_the_run(tree: pathlib.Path) -> None:
    victim = sorted((tree / "data" / "corpus" / "figlib" / "dev0").glob("*.jpg"))[-1]
    victim.unlink()
    with pytest.raises(SystemExit) as excinfo:
        _run(tree, "--arm", "dev")
    assert "make fetch" in str(excinfo.value)


def test_no_flag_changes_what_the_measurement_is(tree: pathlib.Path) -> None:
    """R10, checked in the source rather than in a paragraph about the source."""
    source = (ROOT / "scripts" / "bench.py").read_text(encoding="utf-8")
    for forbidden in ("MOCK", "OFFLINE", "--dry-run"):
        assert forbidden not in source, f"{forbidden} must not exist anywhere in the bench path"


# ---------------------------------------------------------------------------------
# Parity with the pipeline that is actually deployed
# ---------------------------------------------------------------------------------
# The headline is a claim about the system the wall runs. Every line below asserts that
# bench.py's replay is api/main.py's ``Replay._tick`` and not a second implementation that
# drifted into being a better detector -- which is the one way this project can publish a
# number that is honestly measured and still not true of anything it ships.


class _FixedSegmenter:
    """A segmenter that returns a chosen component list, so the choice is the test."""

    def __init__(self, components: list[Any]) -> None:
        self._components = components

    def apply(self, grey: np.ndarray) -> list[Any]:
        return list(self._components)


def test_the_camera_is_scored_on_its_largest_component_and_measured_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``api/main.py:_best_track`` scores the largest component; so must this.

    Taking the best-scoring of up to MAX_COMPONENTS_PER_FRAME instead would benchmark a
    strictly stronger detector than the deployed one -- every minute of the resulting
    advantage would be unearned -- and would pay for up to eight ``measure`` calls per
    camera-minute where the wall pays for one. The stub scores the SMALL component higher,
    so a max-over-components implementation fails on both counts.
    """
    from plumerank.statistic import Tracker
    from plumerank.types import Component, Stats

    big = Component(x=10, y=10, w=40, h=40, area=1600, cx=30.0, cy=30.0)
    small = Component(x=200, y=100, w=10, h=10, area=100, cx=205.0, cy=105.0)
    measured: list[Component] = []

    def stub_measure(track: Any, grey: Any, prev_grey: Any = None) -> Stats:
        component = track.last
        measured.append(component)
        strong = 0.9 if component.area < big.area else 0.05
        return Stats(s1=strong, s2=strong, s3=strong, s4=0.0, raw_s3=0.0)

    monkeypatch.setattr(bench, "measure", stub_measure)
    lane = bench._Lane(
        window=bench.Window(sequence_id="w", camera_id="c", t0_index=None, frame_paths=()),
        segmenter=_FixedSegmenter([big, small]),
        tracker=Tracker(),
        hazards={},
    )
    grey = np.zeros((WORK_HEIGHT, WORK_WIDTH), dtype=np.uint8)

    _, _, component = bench._classical_tick(lane, grey, 0, True)

    assert component is big, "the largest component is the one the wall shows and scores"
    assert measured == [big], "one measure() per camera per minute, on the largest component"


def test_each_frame_is_registered_against_the_raw_previous_frame(
    tree: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``api/main.py`` keeps ``prev_grey`` raw and registers frame t onto it.

    Registering onto the previous *stabilised* frame chains the warps, so the replay
    converges on a common reference the deployed loop never builds -- a different
    background model, different components, a different p50, and nothing in the output
    saying the benchmark and the wall ran different pipelines.
    """
    windows = bench.load_arm(tree, "dev")
    raw = [
        [
            cv2.cvtColor(bench.read_working_frame(path), cv2.COLOR_BGR2GRAY)
            for path in window.frame_paths
        ]
        for window in windows
    ]
    seen: list[np.ndarray] = []
    real = bench.stabilise

    def spy(prev: np.ndarray, cur: np.ndarray) -> Any:
        seen.append(np.array(prev, copy=True))
        return real(prev, cur)

    monkeypatch.setattr(bench, "stabilise", spy)
    bench.replay_arm(
        windows,
        arm="dev",
        background="mog2",
        s4_enabled=True,
        escalation_enabled=False,
        classifier=None,
        systems=("plumerank",),
    )

    assert len(seen) == (N_FRAMES - 1) * len(windows), "every frame but the first registers"
    k = 0
    for t in range(1, N_FRAMES):
        for w in range(len(windows)):
            assert np.array_equal(seen[k], raw[w][t - 1]), f"frame {t} of window {w}"
            k += 1


# ---------------------------------------------------------------------------------
# Which arm owns the published filename
# ---------------------------------------------------------------------------------


def test_only_the_hold_arm_may_write_the_published_headline_name(
    tree: pathlib.Path, commit: str
) -> None:
    """``results/bench_<date>.json`` is an address, not a convention.

    ``api/main.py:_benchmark_file`` and ``check_submission_readiness.py`` both take the
    lexically newest ``results/bench_*.json`` with no arm filter. A DEV payload landing
    there is an in-sample number served by ``/api/benchmark``, rendered on ``/results`` and
    compared by the readiness check as the held-out headline -- produced by a command
    anyone might reasonably run on the same afternoon as ``make bench``.
    """
    _freeze(tree)
    assert _run(tree, "--arm", "hold") == 0
    published = tree / "results" / f"bench_{bench._today()}.json"
    before = published.read_bytes()

    assert _run(tree, "--arm", "dev") == 0
    assert _run(tree, "--arm", "confuse") == 0

    assert published.read_bytes() == before, "no other arm may touch the published file"
    assert [p.name for p in sorted((tree / "results").glob("bench_*.json"))] == [published.name]
    assert json.loads(published.read_text(encoding="utf-8"))["arm"] == "hold"


def test_the_confuse_arm_reads_its_pixels_once_and_publishes_one_rate(
    tree: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On ``--arm confuse`` the calibration set IS the arm, so the two rates are one rate.

    Replaying the same JPEGs twice costs twice the CPU and hands the second pass a
    different point in OpenCV's RNG stream, so the file would carry two slightly different
    false-alarm rates for one corpus with nothing to explain the gap.
    """
    calls = _counting(monkeypatch)
    assert _run(tree, "--arm", "confuse") == 0
    assert calls[0] == 1, "one replay, because the arm and the confusion set are one set"

    payload = _bench_json(tree, "confuse")
    for row in payload["baselines"]:
        assert row["calibration_realised_rate_uncapped"] == row["arm_realised_rate_uncapped"], (
            row["system"]
        )


def test_every_ablation_cell_is_seeded_before_it_runs(
    tree: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming is the default, so a cell must be a function of its own configuration.

    Seeding once per process makes the RNG state a cell starts from depend on how many
    cells preceded it: the grid an SSH drop produced is then not the grid a clean run
    produces, and neither is necessarily the published one.

    The assertion is on the seeding rather than on two grids agreeing, and deliberately:
    the classical stage is deterministic on this fixture today, so a value comparison
    would pass whether or not the seeding survived -- it would be a test with no failure
    mode. This one fails the moment the reseed is removed, which is the regression.
    Resumed cells are not reseeded because they are not recomputed.
    """
    two = [("mog2", True, True), ("mog2", True, False)]
    monkeypatch.setattr(bench, "ablation_cells", lambda: list(two))
    seeds: list[int] = []
    monkeypatch.setattr(bench, "_seed_everything", lambda seed: seeds.append(seed))

    csv_path = tree / "results" / "ablation_seeded.csv"
    bench.run_ablation(tree, csv_path, **_ablation_kwargs(tree, bench.replay_arm))
    assert seeds == [20261026, 20261026], "one reseed per computed cell"

    seeds.clear()
    bench.run_ablation(tree, csv_path, **_ablation_kwargs(tree, bench.replay_arm))
    assert seeds == [], "a fully resumed grid recomputes nothing and reseeds nothing"


# ---------------------------------------------------------------------------------
# What block 1 is allowed to look like
# ---------------------------------------------------------------------------------


def test_the_micro_run_never_prints_a_second_headline_shaped_block(
    tree: pathlib.Path, commit: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run_arm`` builds a headline block for every arm, micro included.

    Printing the micro one would put ``PlumeRank -- median time-to-detection @ 2.00
    alerts/camera-night`` on screen off a four-window corpus, in the exact shape of the
    published HOLD block -- two headline-shaped blocks, one of which is not the headline.
    """
    _freeze(tree)
    assert _run(tree, "--arm", "hold") == 0
    capsys.readouterr()
    assert _run(tree, "--arm", "micro") == 0

    out = capsys.readouterr().out
    marker = "PlumeRank -- median time-to-detection"
    assert out.count(marker) == 1, "exactly one headline-shaped block, and it is the quoted one"
    assert out.index(marker) > out.index(bench.BLOCK_2)


def test_the_micro_escalation_line_names_the_roster_it_was_measured_on(
    tree: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ESCALATION_TOP_K caps escalations per TICK, so the fraction scales with the roster.

    Two of four micro windows is 50%; two of sixty HOLD windows is 3.3%. A bare percentage
    under the word REPRODUCED, an order of magnitude above the project's ~1% claim, reads
    as the submission contradicting itself -- so the line has to say what it is a fraction
    of.
    """
    assert _run(tree, "--arm", "micro") == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "touched by ONNX" in ln)
    assert "4-window roster" in line, line
    assert f"top-{bench.K.ESCALATION_TOP_K}" in line, line


# ---------------------------------------------------------------------------------
# The excluded sequences, published in the shape /results can read
# ---------------------------------------------------------------------------------


def test_the_excluded_count_beside_an_arm_is_that_arms_own_exclusions(
    tree: pathlib.Path,
) -> None:
    """``data/excluded.json`` spans all four arms; the arm line describes one arm.

    Printing the whole file's length beside one arm's denominator hands a reader an
    addition that does not come out -- "48 sequences, 20 excluded" over a manifest of 60 --
    and the count is one of the three numbers complexity.md section 8a says may move and
    must therefore be machine-derived every time it is printed.
    """
    (tree / "data" / "excluded.json").write_text(
        json.dumps(
            {
                "dev0": "first frame is a lens flare",
                "hold1": "sequence starts after ignition",
                "nosucharm9": "named by no manifest on this box",
            }
        ),
        encoding="utf-8",
    )
    assert _run(tree, "--arm", "dev") == 0

    payload = _bench_json(tree)
    assert payload["exclusions"]["count"] == 1
    assert payload["exclusions"]["sequence_ids"] == ["dev0"]
    assert sorted(payload["exclusions"]["all_arms_sequence_ids"]) == [
        "dev0",
        "hold1",
        "nosucharm9",
    ]
    assert payload["n_windows"] == 1, "the dropped window is out of the denominator too"
    arm_line = next(
        line for line in payload["headline_block"].splitlines() if line.strip().startswith("arm")
    )
    assert "1 sequences, 1 excluded" in arm_line, arm_line


def test_the_exclusion_reason_survives_into_the_published_file(tree: pathlib.Path) -> None:
    """``/results`` renders records, not ids -- and "why" is the question a count invites.

    ``web/lib/read.ts:readExclusions`` reads ``exclusions.items`` as objects carrying
    ``sequence_id`` and ``reason`` and drops every entry whose ``sequence_id`` is empty, so
    a list of bare strings renders as an empty panel next to a denominator that moved.
    """
    (tree / "data" / "excluded.json").write_text(
        json.dumps([{"sequence_id": "dev1", "reason": "smoke already visible at t0"}]),
        encoding="utf-8",
    )
    assert _run(tree, "--arm", "dev") == 0

    items = _bench_json(tree)["exclusions"]["items"]
    assert items == [{"sequence_id": "dev1", "reason": "smoke already visible at t0"}]

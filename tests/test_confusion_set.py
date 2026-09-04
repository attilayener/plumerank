"""scripts/build_confusion_set.py, driven against a synthetic archive in a temporary tree.

No socket is opened here either: ``probe_exists`` and ``fetch_with_retry`` are the script's
only two ways out, and both are replaced. What is NOT synthetic is the camera roster and
the clock -- those come from the committed ``data/manifest_hold.json``, because the
invariants worth asserting are exactly the ones that relate CONFUSE to HOLD:

* every CONFUSE camera appears in HOLD;
* every CONFUSE window sits within +/-45 minutes of its camera's paired HOLD ignition,
  recomputed from the frame epochs rather than read back out of the field the script wrote;
* every window is 81 frames, or there is no manifest at all.

A confusion set that fails any of those is a straw man, and a straw man calibrates the
threshold every headline in the project is quoted at. Nothing here checks that the images
are pretty; everything here checks that the arm is the same experiment as HOLD.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys
import zlib
from datetime import date, datetime, timezone

import cv2
import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_confusion_set as bcs  # noqa: E402

from plumerank.constants import (  # noqa: E402
    FRAMES_PER_WINDOW,
    IGNITION_FRAME_INDEX,
    WORK_HEIGHT,
    WORK_WIDTH,
)

_DATA = _ROOT / "data"

#: The day the synthetic archive holds frames for. Far from every FIgLib fire date in the
#: committed manifests, so the non-fire exclusion never removes it.
_SWEEP_DAY = date(2026, 7, 26)

#: Seconds past the minute this camera writes at. Non-zero on purpose: a window's predicted
#: epochs come from a wall clock that knows nothing about the camera's phase, so a phase of
#: zero would let a broken phase search pass.
_CAMERA_PHASE_SECONDS = 17


def _seed(label: str) -> int:
    """A stable seed for one class's scenes.

    NOT ``hash(label)``: str hashing is salted per interpreter unless PYTHONHASHSEED is
    pinned, so that spelling regenerates every scene in this file from a different seed on
    every run. A suite that quietly draws new pixels each time cannot tell a real
    regression from a bad draw, in a repository whose headline claim is that a number comes
    back the same twice.
    """
    return zlib.crc32(label.encode("ascii"))


# ---------------------------------------------------------------------------------
# Synthetic scenes -- one generator per confuser class
# ---------------------------------------------------------------------------------

_SRC_H, _SRC_W = 360, 640


def _terrain(rng: np.random.Generator) -> np.ndarray:
    """A clear day: smooth sky over blocky ground. The null case, and no class at all."""
    img = np.zeros((_SRC_H, _SRC_W, 3), np.uint8)
    horizon = _SRC_H // 2
    for row in range(horizon):
        img[row, :, :] = 205 - int(30 * row / horizon)
    blocks = rng.integers(40, 190, (_SRC_H - horizon, _SRC_W // 8), dtype=np.uint8)
    img[horizon:, :, :] = np.repeat(blocks, 8, axis=1)[:, :_SRC_W, None]
    return img


def _grain(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sensor noise. Without it two identical frames have nothing that moved, and the
    hardness statistic would be measuring an empty difference image."""
    noise = rng.integers(-2, 3, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _scene(label: str, k: int, rng: np.random.Generator) -> np.ndarray:
    if label == "fog_marine":
        # A marine layer that has swallowed the scene and is still rising: neutral, almost
        # edgeless, and the ground is exactly as bright as the sky.
        img = _terrain(rng)
        bank = np.zeros((_SRC_H, _SRC_W), np.float32)
        bank[max(0, 46 - 7 * k) :] = 1.0
        bank = cv2.GaussianBlur(bank, (0, 0), 18)[:, :, None]
        return _grain((img * (1 - bank) + 200 * bank).astype(np.uint8), rng)
    if label == "low_cloud":
        img = np.full((_SRC_H, _SRC_W, 3), 90, np.uint8)
        img[: _SRC_H // 3 + 20] = 210  # the deck is ABOVE; the ground stays dark
        return _grain(img, rng)
    if label == "dust":
        img = np.zeros((_SRC_H, _SRC_W, 3), np.uint8)
        img[:, :, 0], img[:, :, 1], img[:, :, 2] = 60, 110, 175  # BGR: brown, not grey
        return _grain(img, rng)
    if label == "exposure_hunting":
        return _grain(
            np.full((_SRC_H, _SRC_W, 3), 120 if k % 2 == 0 else 150, np.uint8), rng
        )
    if label == "cloud_shadow":
        img = _terrain(rng)
        shadow = np.zeros((_SRC_H, _SRC_W), np.float32)
        cv2.circle(shadow, (int(70 + k * 75), _SRC_H - 80), 130, 1.0, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 20)[:, :, None]
        return _grain((img * (1 - 0.6 * shadow)).astype(np.uint8), rng)
    if label == "lens_artefact":
        img = _terrain(rng)
        for i in range(14):  # a web across the glass: fixed structure in the empty sky
            cv2.line(img, (0, 6 * i + 4), (_SRC_W, 6 * i + 30), (245, 245, 245), 1)
        cv2.circle(img, (_SRC_W // 3, _SRC_H // 6), 26, (250, 250, 250), -1)
        return _grain(img, rng)
    return _grain(_terrain(rng), rng)


def _encode(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def _class_frames() -> dict[str, list[bytes]]:
    """One JPEG per probe slot per class, shared by every camera of that class.

    Shared rather than generated per camera because the classifier reads pixels, not
    identities, and 45 x 7 fresh scenes would make this file the slowest in the suite.
    """
    out: dict[str, list[bytes]] = {}
    for label in list(bcs.CLASS_QUOTAS) + ["clear"]:
        rng = np.random.default_rng(_seed(label))
        out[label] = [_encode(_scene(label, k, rng)) for k in range(bcs.PROBE_SLOTS)]
    return out


# ---------------------------------------------------------------------------------
# The synthetic archive
# ---------------------------------------------------------------------------------


class _Archive:
    """``{url: bytes}`` plus a membership set, standing in for a bucket with no index."""

    def __init__(self) -> None:
        self.images: dict[str, bytes] = {}
        self.heads = 0

    def head(self, url: str, timeout: float = 0.0) -> bool:
        self.heads += 1
        return url in self.images

    def get(self, url: str, attempts: int = 3) -> bytes:
        if url not in self.images:
            raise bcs.FetchError(f"synthetic archive has no {url}")
        return self.images[url]


def _camera_classes(hold: dict) -> dict[str, str]:
    """One class per HOLD camera, in quota order. 45 cameras, 45 windows, one each."""
    cameras = sorted({w["camera_id"] for w in hold["windows"]})
    assert len(cameras) == sum(bcs.CLASS_QUOTAS.values())
    plan: list[str] = []
    for label, quota in bcs.CLASS_QUOTAS.items():
        plan.extend([label] * quota)
    return dict(zip(cameras, plan))


def _populate(
    archive: _Archive,
    hold: dict,
    classes: dict[str, str],
    slots: list[int] | None = None,
) -> None:
    """Lay one 81-frame window per camera into the archive at the right hour."""
    frames = _class_frames()
    paired = bcs.paired_hold_windows(hold)
    indices = list(range(FRAMES_PER_WINDOW)) if slots is None else slots
    for camera_id, window in paired.items():
        minutes = bcs.local_minutes_of_day(bcs.hold_t0_epoch(window))
        centre = bcs.candidate_centre_epoch(_SWEEP_DAY, minutes, 0) + _CAMERA_PHASE_SECONDS
        label = classes[camera_id]
        for i in indices:
            epoch = centre + (i - IGNITION_FRAME_INDEX) * 60
            archive.images[bcs.frame_url(camera_id, epoch)] = frames[label][
                i * bcs.PROBE_SLOTS // FRAMES_PER_WINDOW
            ]


@pytest.fixture()
def tree(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """A data root with the real DEV and HOLD manifests, and an empty archive."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    for name in ("manifest_dev.json", "manifest_hold.json"):
        shutil.copy(_DATA / name, data_root / name)

    archive = _Archive()
    monkeypatch.setattr(bcs, "probe_exists", archive.head)
    monkeypatch.setattr(bcs, "fetch_with_retry", archive.get)
    # One day, one candidate per camera: the sweep's breadth is what makes the real run
    # take an hour, and it is not what these tests are about.
    monkeypatch.setattr(bcs, "CANDIDATE_DAYS", 1)
    monkeypatch.setattr(bcs, "CANDIDATE_DAY_STRIDE", 1)
    monkeypatch.setattr(bcs, "MAX_CANDIDATES_PER_CAMERA", 1)

    hold = json.loads((data_root / "manifest_hold.json").read_text())
    return data_root, archive, hold


def _run(data_root: pathlib.Path) -> int:
    return bcs.main(
        [
            "--data-root",
            str(data_root),
            "--end-date",
            _SWEEP_DAY.isoformat(),
        ]
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    """One real selection run, shared by every manifest assertion below.

    Module-scoped because the run costs a second and the ten tests that read its output all
    read the SAME output -- rebuilding per test would not test anything extra and would put
    this file alone over a third of the 90 s `make test` budget.
    """
    data_root = tmp_path_factory.mktemp("confuse") / "data"
    data_root.mkdir()
    for name in ("manifest_dev.json", "manifest_hold.json"):
        shutil.copy(_DATA / name, data_root / name)
    hold = json.loads((data_root / "manifest_hold.json").read_text())

    archive = _Archive()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bcs, "probe_exists", archive.head)
        patch.setattr(bcs, "fetch_with_retry", archive.get)
        patch.setattr(bcs, "CANDIDATE_DAYS", 1)
        patch.setattr(bcs, "CANDIDATE_DAY_STRIDE", 1)
        patch.setattr(bcs, "MAX_CANDIDATES_PER_CAMERA", 1)
        _populate(archive, hold, _camera_classes(hold))
        assert _run(data_root) == 0

    return json.loads((data_root / "manifest_confuse.json").read_text()), hold


# ---------------------------------------------------------------------------------
# The URL shape, checked against 60 real working examples
# ---------------------------------------------------------------------------------

_MTA_URL = re.compile(
    r"^https://cdn\.hpwren\.ucsd\.edu/MTA/([^/]+)/large/(\d{8})/Q(\d)/(\d+)\.jpg$"
)


def test_the_quarter_directory_is_derived_correctly_from_the_epoch() -> None:
    """The one piece of archive structure that cannot be guessed, checked against fact.

    The bucket serves no index, so the date and quarter in a path can only be validated
    against URLs known to work. ``spike/urls_fog.txt`` holds 60 of them. Both are UTC and
    the quarter is numbered from 1 -- get either wrong and every enumerated URL 403s while
    looking perfectly reasonable.
    """
    lines = [
        line.strip()
        for line in (_ROOT / "spike" / "urls_fog.txt").read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 60
    for line in lines:
        match = _MTA_URL.match(line)
        assert match, line
        camera, day, quarter, epoch = match.groups()
        assert bcs.frame_url(camera, int(epoch)) == line
        assert bcs.quarter_directory(int(epoch)) == f"{day}/Q{quarter}"


def test_the_quarter_boundary_is_local_and_one_based() -> None:
    """The archive buckets by SAN DIEGO LOCAL date and a LOCAL three-hour block numbered
    from 1, so Q runs 1..8 -- not by a six-hour UTC block, which is what this test
    asserted until it was checked against the live bucket.

    Decisive evidence, measured rather than reasoned: epoch 1785092431 on hp-s-mobo-c
    returns HTTP 200 under .../20260726/Q5/ and HTTP 403 under .../20260726/Q4/. The
    local-DATE half holds too: an epoch at local 2026-07-26 19:00 (UTC 2026-07-27 02:00)
    is served under 20260726/Q7, while both 20260727/Q1 and 20260727/Q7 return 403.

    Why the fixture could not catch this: every one of the 60 URLs in spike/urls_fog.txt
    falls in local hours 6, 7 or 8 -- the single band where the UTC rule and the local
    rule agree on Q3. A round-trip test over that file therefore passed under both, which
    is a good reminder that a fixture drawn from one narrow slice proves one narrow slice.
    """
    boundary = int(datetime(2026, 7, 26, 9, 0, 0, tzinfo=bcs.PACIFIC).timestamp())
    assert bcs.quarter_directory(boundary) == "20260726/Q4"
    assert bcs.quarter_directory(boundary - 1) == "20260726/Q3"

    # Local evening stays on the local date rather than rolling into the next UTC day.
    evening = int(datetime(2026, 7, 26, 19, 0, 0, tzinfo=bcs.PACIFIC).timestamp())
    assert bcs.quarter_directory(evening) == "20260726/Q7"


# ---------------------------------------------------------------------------------
# Clock arithmetic
# ---------------------------------------------------------------------------------


def test_times_of_day_are_compared_circularly() -> None:
    # 23:50 and 00:10 are twenty minutes apart. The linear answer is 1420 and would reject
    # every legal pre-dawn pairing without saying why.
    assert bcs.minutes_apart(23 * 60 + 50, 10) == 20
    assert bcs.minutes_apart(10, 23 * 60 + 50) == 20
    assert bcs.minutes_apart(600, 645) == 45


def test_a_frame_epoch_is_read_the_same_way_in_both_archives() -> None:
    assert bcs.frame_epoch("1719093958_+00060.jpg") == 1719093958
    assert bcs.frame_epoch("1719091498_-02400.jpg") == 1719091498
    assert bcs.frame_epoch("large/20260726/Q3/1785071971.jpg") == 1785071971


def test_a_candidate_centre_lands_on_the_local_time_of_day_it_was_asked_for() -> None:
    # Built through a localised datetime rather than by adding seconds to UTC midnight, so
    # that a summer day is not an hour off -- which for a marine-layer camera is the
    # difference between fog and no fog.
    epoch = bcs.candidate_centre_epoch(date(2026, 7, 26), 6 * 60 + 20, 0)
    assert bcs.local_minutes_of_day(epoch) == 6 * 60 + 20
    shifted = bcs.candidate_centre_epoch(date(2026, 7, 26), 6 * 60 + 20, -30)
    assert bcs.minutes_apart(bcs.local_minutes_of_day(shifted), 6 * 60 + 20) == 30


# ---------------------------------------------------------------------------------
# Walking an archive that will not list itself
# ---------------------------------------------------------------------------------


def test_the_phase_search_finds_a_camera_that_does_not_write_on_the_minute(tree) -> None:
    data_root, archive, hold = tree
    _populate(archive, hold, _camera_classes(hold))
    camera = sorted({w["camera_id"] for w in hold["windows"]})[0]
    paired = bcs.paired_hold_windows(hold)[camera]
    centre = bcs.candidate_centre_epoch(
        _SWEEP_DAY, bcs.local_minutes_of_day(bcs.hold_t0_epoch(paired)), 0
    )

    found = bcs.enumerate_window(camera, centre, list(range(FRAMES_PER_WINDOW)))
    assert len(found) == FRAMES_PER_WINDOW
    assert found[0] == centre - IGNITION_FRAME_INDEX * 60 + _CAMERA_PHASE_SECONDS


def test_the_phase_is_paid_for_once_and_then_reused(tree) -> None:
    data_root, archive, hold = tree
    _populate(archive, hold, _camera_classes(hold))
    camera = sorted({w["camera_id"] for w in hold["windows"]})[0]
    paired = bcs.paired_hold_windows(hold)[camera]
    centre = bcs.candidate_centre_epoch(
        _SWEEP_DAY, bcs.local_minutes_of_day(bcs.hold_t0_epoch(paired)), 0
    )

    archive.heads = 0
    bcs.enumerate_window(camera, centre, list(range(FRAMES_PER_WINDOW)))
    # One full-minute search plus one HEAD for each of the other 80 slots. Searching the
    # whole minute 81 times would be ~5,000 requests per window at HPWREN's expense.
    assert archive.heads < FRAMES_PER_WINDOW + 2 * bcs.PHASE_SEARCH_SECONDS + 2


def test_a_slot_the_archive_does_not_hold_is_simply_absent(tree) -> None:
    data_root, archive, hold = tree
    _populate(archive, hold, _camera_classes(hold), slots=[0, 40, 80])
    camera = sorted({w["camera_id"] for w in hold["windows"]})[0]
    paired = bcs.paired_hold_windows(hold)[camera]
    centre = bcs.candidate_centre_epoch(
        _SWEEP_DAY, bcs.local_minutes_of_day(bcs.hold_t0_epoch(paired)), 0
    )
    found = bcs.enumerate_window(camera, centre, [0, 1, 40, 80])
    assert sorted(found) == [0, 40, 80]


# ---------------------------------------------------------------------------------
# The measurement that assigns a class
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("label", sorted(bcs.CLASS_QUOTAS))
def test_each_synthetic_confuser_lands_in_its_own_class(label: str) -> None:
    rng = np.random.default_rng(_seed(label))
    frames = [
        cv2.resize(
            _scene(label, k, rng), (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA
        )
        for k in range(bcs.PROBE_SLOTS)
    ]
    assert bcs.measure(frames).label == label


def test_an_ordinary_clear_day_is_not_any_class() -> None:
    # A false-alarm denominator made of clear afternoons would be trivially easy and the
    # solved threshold would be meaningless, so the classifier has to be able to say no.
    rng = np.random.default_rng(7)
    frames = [
        cv2.resize(
            _scene("clear", k, rng), (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA
        )
        for k in range(bcs.PROBE_SLOTS)
    ]
    assert bcs.measure(frames).label is None


def test_fog_and_low_cloud_are_separated_only_by_where_the_layer_sits() -> None:
    # The two classes share every other feature; the sky-over-ground ratio is the whole
    # discriminator, and it is worth one test on its own because a sign error there would
    # swap 18 windows with 4 and nothing else would notice.
    fog = {
        "sky_persistent_structure": 0.0,
        "luma_reversal_max": 0.1,
        "warmth_r_minus_b": 0.0,
        "saturation_mean": 2.0,
        "edge_energy": 4.0,
        "sky_ground_ratio": 1.0,
        "shadow_motion": 0.0,
    }
    assert bcs.classify(fog) == "fog_marine"
    assert bcs.classify({**fog, "sky_ground_ratio": 2.3}) == "low_cloud"


def test_a_fog_bank_under_a_rising_sun_is_fog_and_not_a_broken_camera() -> None:
    """The dawn case, which a max-step exposure test labels ``exposure_hunting``.

    Probe frames are ~13 minutes apart, so the sun moves whole-frame luma by more than
    EXPOSURE_REVERSAL_MIN across a window while the marine layer sits under it. Because
    exposure_hunting is evaluated SECOND in _CLASS_ORDER, a magnitude-only test does not
    merely mislabel one window -- it takes the hardest fog candidates out of the pool
    before the 18 fog slots are ever offered them, and the manifest then publishes
    "auto-exposure hunting" as the reason. Only the instrument reverses direction.
    """
    rng = np.random.default_rng(_seed("fog_marine"))
    ramping = [
        cv2.resize(
            np.clip(_scene("fog_marine", k, rng).astype(np.int16) + 9 * k, 0, 255).astype(
                np.uint8
            ),
            (WORK_WIDTH, WORK_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        for k in range(bcs.PROBE_SLOTS)
    ]
    measurement = bcs.measure(ramping)
    # The monotone climb is real and large -- this is not a scene that stayed still.
    assert max(
        abs(float(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).mean())
            - float(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).mean()))
        for a, b in zip(ramping, ramping[1:])
    ) >= bcs.EXPOSURE_REVERSAL_MIN
    assert measurement.features["luma_reversal_max"] < bcs.EXPOSURE_REVERSAL_MIN
    assert measurement.label == "fog_marine"


def test_a_reversal_is_only_as_big_as_the_part_that_came_back() -> None:
    # A one-way step of 40 followed by a 1-unit correction is not a hunting shutter, and
    # scoring it 40 would readmit every monotone scene through the back door.
    flat = np.full((WORK_HEIGHT, WORK_WIDTH, 3), 100, np.uint8)
    steps = [100, 140, 139, 139, 139, 139, 139]
    frames = [np.full_like(flat, v) for v in steps]
    assert bcs.measure(frames).features["luma_reversal_max"] == pytest.approx(1.0, abs=0.5)


def test_hardness_is_the_projects_own_boundary_statistic() -> None:
    """The selection criterion is raw S3 and not a bespoke score invented here.

    Selecting the confusion set with anything else would make "these are the hardest
    non-fires for PlumeRank" an assertion. Selecting it with the statistic that is supposed
    to reject fog makes the arm adversarial to this project by construction.
    """
    rng = np.random.default_rng(3)
    frames = [
        cv2.resize(
            _scene("fog_marine", k, rng),
            (WORK_WIDTH, WORK_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        for k in range(bcs.PROBE_SLOTS)
    ]
    value = bcs.hardness(frames)
    assert value > 0.0 and np.isfinite(value)

    grey = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    component = bcs._largest_moving_component(
        cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), grey
    )
    assert component is not None
    from plumerank.statistic import s3_diffuse_boundary

    assert value == pytest.approx(s3_diffuse_boundary(grey, component)[1])


def test_a_still_window_is_the_softest_possible_confuser() -> None:
    # Nothing moved, so there is no component to place a boundary band on. 0.0 and not an
    # exception: a window that cannot fool anything belongs at the bottom of the ranking,
    # not out of it.
    flat = [np.full((WORK_HEIGHT, WORK_WIDTH, 3), 128, np.uint8)] * 3
    assert bcs.hardness(flat) == 0.0


# ---------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------


def _scored(camera: str, label: str, hardness: float) -> bcs._Scored:
    candidate = bcs.Candidate(
        camera_id=camera,
        centre_epoch=1785071971,
        offset_minutes=0,
        paired_hold_sequence_id="held",
        paired_hold_t0_epoch=1785071971,
    )
    measurement = bcs.Measurement(features={}, label=label, hardness=hardness)
    return bcs._Scored(candidate, measurement)


def test_within_a_class_the_hardest_candidate_wins() -> None:
    pool = {
        f"cam{i}": [_scored(f"cam{i}", "fog_marine", float(i))] for i in range(5)
    }
    chosen = bcs._select(pool, {"fog_marine": 2})
    assert {s.candidate.camera_id for s in chosen.values()} == {"cam4", "cam3"}


def test_the_scarcest_class_is_filled_first() -> None:
    """A camera that is the only supplier of a rare class must not be spent on a common one.

    ``cam0`` is the hardest fog in the pool AND the only lens artefact. Filling fog first
    would take it and then fail to fill lens at all -- with 18 fog slots and 3 lens slots
    in the real quotas, that is not a corner case, it is the normal case.
    """
    pool = {
        "cam0": [
            _scored("cam0", "fog_marine", 9.0),
            _scored("cam0", "lens_artefact", 1.0),
        ],
        "cam1": [_scored("cam1", "fog_marine", 5.0)],
    }
    chosen = bcs._select(pool, {"fog_marine": 1, "lens_artefact": 1})
    by_camera = {s.candidate.camera_id: s.measurement.label for s in chosen.values()}
    assert by_camera == {"cam0": "lens_artefact", "cam1": "fog_marine"}


def test_a_class_that_cannot_be_filled_is_a_failure_not_a_short_arm() -> None:
    pool = {"cam0": [_scored("cam0", "dust", 1.0)]}
    with pytest.raises(bcs.FetchError, match="short"):
        bcs._select(pool, {"dust": 3})


# ---------------------------------------------------------------------------------
# The manifest invariants -- the ones a hostile judge runs first
# ---------------------------------------------------------------------------------


def test_every_confuse_camera_appears_in_hold(built) -> None:
    manifest, hold = built
    hold_cameras = {w["camera_id"] for w in hold["windows"]}
    assert {w["camera_id"] for w in manifest["windows"]} <= hold_cameras


def test_every_window_is_within_45_minutes_of_its_paired_hold_ignition(built) -> None:
    """Recomputed from the frame epochs, not read back out of the field the script wrote.

    A test that trusted ``local_minutes_delta`` would pass on a script that computed it
    wrong and wrote it faithfully -- which is the only way this invariant realistically
    breaks.
    """
    manifest, hold = built
    by_id = {w["sequence_id"]: w for w in hold["windows"]}
    for window in manifest["windows"]:
        paired = by_id[window["paired_hold_sequence_id"]]
        assert paired["camera_id"] == window["camera_id"]
        hold_minutes = bcs.local_minutes_of_day(
            bcs.frame_epoch(paired["frames"][paired["t0_index"]])
        )
        centre = bcs.frame_epoch(window["frames"][window["centre_index"]])
        assert bcs.minutes_apart(bcs.local_minutes_of_day(centre), hold_minutes) <= 45


def test_every_window_is_exactly_81_frames(built) -> None:
    manifest, _ = built
    for window in manifest["windows"]:
        assert window["n_frames"] == len(window["frames"]) == FRAMES_PER_WINDOW
        # 60 s apart and strictly increasing: a window assembled out of order would score
        # a plume growing backwards.
        epochs = [bcs.frame_epoch(f) for f in window["frames"]]
        assert epochs == sorted(epochs)
        assert all(50 <= b - a <= 70 for a, b in zip(epochs, epochs[1:]))


def test_the_class_histogram_is_exactly_the_quota(built) -> None:
    manifest, _ = built
    histogram: dict[str, int] = {}
    for window in manifest["windows"]:
        histogram[window["class"]] = histogram.get(window["class"], 0) + 1
    assert histogram == bcs.CLASS_QUOTAS
    assert manifest["n_windows"] == 45 == sum(bcs.CLASS_QUOTAS.values())


def test_no_window_carries_an_ignition_clock(built) -> None:
    # A CONFUSE window is the false-alarm denominator: nothing ignites, so types.Sequence
    # must read it as is_fire False. A stray t0_index would make it a fire sequence with no
    # fire in it and quietly poison the detection metric.
    manifest, _ = built
    assert all(w["t0_index"] is None for w in manifest["windows"])


def test_one_window_per_camera(built) -> None:
    manifest, _ = built
    cameras = [w["camera_id"] for w in manifest["windows"]]
    assert len(cameras) == len(set(cameras))


def test_sequence_ids_are_disjoint_from_dev_and_hold(built) -> None:
    manifest, hold = built
    dev = json.loads((_DATA / "manifest_dev.json").read_text())
    other = {w["sequence_id"] for w in hold["windows"]} | {
        w["sequence_id"] for w in dev["windows"]
    }
    assert not ({w["sequence_id"] for w in manifest["windows"]} & other)


def test_every_window_records_the_measurement_that_selected_it(built) -> None:
    manifest, _ = built
    for window in manifest["windows"]:
        criterion = window["criterion"]
        assert criterion["rule"]
        assert isinstance(criterion["hardness_raw_s3"], (int, float))
        # All eight features, every time -- the selection has to be re-derivable from the
        # manifest alone, or "the criterion is computed" is just a sentence.
        assert set(criterion["features"]) == {
            "luma_mean",
            "edge_energy",
            "sky_ground_ratio",
            "warmth_r_minus_b",
            "saturation_mean",
            "luma_reversal_max",
            "shadow_motion",
            "sky_persistent_structure",
        }
    assert set(manifest["criterion_thresholds"]) >= {"HAZE_EDGE_ENERGY_MAX"}
    assert manifest["search_budget"]["end_date"] == _SWEEP_DAY.isoformat()


def test_the_manifest_carries_the_hpwren_credit_and_the_fetch_contract(built) -> None:
    manifest, _ = built
    assert manifest["credit"] == "Imagery: HPWREN, http://hpwren.ucsd.edu/"
    assert "%2B" in manifest["url_note"]
    assert "receipt pins WHAT" in manifest["sha256_note"]


def test_fetch_can_plan_the_confuse_manifest_without_a_branch(built, tmp_path) -> None:
    """The other half of the seam: fetch.py must consume this manifest as written.

    ``url_prefix`` and ``local_root`` are the only two keys CONFUSE adds, and they exist
    precisely so that fetch.py has one engine rather than one branch per archive.
    """
    manifest, _ = built
    import fetch

    planned = fetch.plan_manifest(manifest, tmp_path / "manifest_confuse.json")
    assert len(planned) == 45 * FRAMES_PER_WINDOW
    assert all(p.url.startswith(bcs.MTA_BASE_URL) for p in planned)
    assert all(p.relative_path.startswith("corpus/mta/") for p in planned)
    assert all(_MTA_URL.match(p.url) for p in planned)


# ---------------------------------------------------------------------------------
# Degrading honestly
# ---------------------------------------------------------------------------------


def test_an_unreachable_archive_writes_no_manifest(tree, monkeypatch) -> None:
    data_root, _, _ = tree

    def down(url: str, timeout: float = 0.0) -> bool:
        raise bcs.FetchError("HEAD https://cdn.hpwren.ucsd.edu/... failed: [Errno 8]")

    monkeypatch.setattr(bcs, "probe_exists", down)
    assert _run(data_root) == 1
    assert not (data_root / "manifest_confuse.json").exists()


def test_an_empty_archive_fails_rather_than_writing_a_short_arm(tree) -> None:
    data_root, _, _ = tree
    # Nothing populated. Every class comes up empty, and the failure has to be a non-zero
    # exit rather than a manifest with zero windows that solve_threshold would happily
    # bisect on.
    assert _run(data_root) == 1
    assert not (data_root / "manifest_confuse.json").exists()


def test_a_window_that_cannot_reach_81_frames_is_not_shortened(tree) -> None:
    data_root, archive, hold = tree
    # The screening probes exist, so selection succeeds; the full enumeration then finds
    # holes. A shortened window would be a shorter camera-night than every other window in
    # the evaluation and would silently change the alert rate's denominator.
    _populate(
        archive,
        hold,
        _camera_classes(hold),
        slots=[round(i * (FRAMES_PER_WINDOW - 1) / (bcs.PROBE_SLOTS - 1)) for i in range(bcs.PROBE_SLOTS)],
    )
    assert _run(data_root) == 1
    assert not (data_root / "manifest_confuse.json").exists()


def test_a_missing_hold_manifest_is_a_hard_stop(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    assert bcs.main(["--data-root", str(data_root)]) == 2
    assert not (data_root / "manifest_confuse.json").exists()


def test_the_manifest_is_written_once_at_the_end(tree, monkeypatch) -> None:
    """No partial manifest, ever -- checked by failing after selection has succeeded.

    A half-built CONFUSE arm loads, solves a threshold and prints a number, and nothing
    downstream can tell. That makes "write last, or not at all" a correctness property
    rather than tidiness.
    """
    data_root, archive, hold = tree
    _populate(archive, hold, _camera_classes(hold))
    out = data_root / "manifest_confuse.json"

    real = bcs.enumerate_window
    calls: list[int] = []

    def fail_late(*args, **kwargs):
        calls.append(1)
        if len(calls) > 45:  # the screening pass is done; the enumeration pass has begun
            return {}
        return real(*args, **kwargs)

    monkeypatch.setattr(bcs, "enumerate_window", fail_late)
    assert _run(data_root) == 1
    assert not out.exists()

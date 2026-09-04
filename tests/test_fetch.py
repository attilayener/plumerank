"""scripts/fetch.py, driven against a stubbed archive in a temporary tree.

Not one of these tests opens a socket. ``fetch.http_get`` is the only place in the script
that does, so replacing it replaces the network -- and the fact that ONE monkeypatch is
enough is itself worth asserting, because a second socket would be a second thing that can
reach out of ``make test``.

The two properties that matter most here are the ones a corpus loses silently:

* **The receipt round-trip.** A first run records a hash per file; a later run re-hashes
  and disagrees loudly. Half the tests below are about the ways that disagreement can be
  laundered into a pass, and none of them may work.
* **Fetching is not reading.** Writing HOLD bytes to disk must never touch
  ``results/hold_access.log``. That log is what ``test_hold_untouched_until_freeze``
  builds its argument on, and a log that fires on ``make fetch`` would make the argument
  worthless.
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fetch  # noqa: E402

from plumerank.constants import FRAMES_PER_WINDOW, WORK_HEIGHT, WORK_WIDTH  # noqa: E402

_DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------------
# A stubbed archive
# ---------------------------------------------------------------------------------


def _jpeg(seed: int, width: int = 320, height: int = 240) -> bytes:
    """A real JPEG, because the micro build decodes what the fetch wrote.

    Deterministic per seed so a "the server changed its mind" test can serve different
    bytes on purpose rather than by accident.
    """
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


class _Archive:
    """A dict of url -> bytes, plus a call log, standing in for the CDN."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.gets: list[str] = []

    def __call__(self, url: str, timeout: float = 0.0) -> bytes:
        self.gets.append(url)
        if url not in self.files:
            raise fetch.FetchError(f"stub archive has no {url}")
        return self.files[url]


def _manifest(arm: str, n_windows: int = 2, n_frames: int = 3) -> dict:
    """A miniature manifest in the committed schema, with FIgLib's '+' frame names."""
    windows = []
    for w in range(n_windows):
        frames = [
            f"{1719091498 + 60 * f}_{'+' if f else '-'}{f * 60:05d}.jpg"
            for f in range(n_frames)
        ]
        windows.append(
            {
                "sequence_id": f"2024062{w}_TestFire_cam{w}-n-mobo-c",
                "camera_id": f"cam{w}-n-mobo-c",
                "date": f"2024062{w}",
                "t0_index": 0,
                "t0_offset_seconds": 0,
                "n_frames": n_frames,
                "frames": frames,
            }
        )
    return {
        "arm": arm,
        "n_windows": n_windows,
        "generated": "2026-09-04",
        "source": "HPWREN FIgLib",
        "credit": "Imagery: HPWREN, http://hpwren.ucsd.edu/",
        "licence_note": "Evaluation only.",
        "base_url": "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/",
        "url_note": "Frame URL = base_url + sequence_id + '/' + frame.",
        "catalogue": "test",
        "selection": ["test"],
        "sha256_note": "test",
        "windows": windows,
    }


@pytest.fixture()
def tree(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> tuple[pathlib.Path, _Archive]:
    """A data root holding DEV and HOLD manifests, and the archive that serves them."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    files: dict[str, bytes] = {}
    seed = 0
    for arm in ("dev", "hold"):
        manifest = _manifest(arm, n_windows=5, n_frames=4)
        (data_root / f"manifest_{arm}.json").write_text(json.dumps(manifest))
        for window in manifest["windows"]:
            prefix = manifest["base_url"] + window["sequence_id"] + "/"
            for frame in window["frames"]:
                seed += 1
                files[fetch.frame_url(prefix, frame)] = _jpeg(seed)
    # CONFUSE is written by the other script and is legitimately absent here; the fetch
    # group must not silently succeed on it, so the tests below request the manifests they
    # mean rather than the whole group.
    archive = _Archive(files)
    monkeypatch.setattr(fetch, "http_get", archive)
    monkeypatch.setattr(fetch, "RETRY_BACKOFF_SECONDS", 0.0)
    return data_root, archive


def _run(data_root: pathlib.Path, manifests: tuple[str, ...] = ("manifest_dev.json",)) -> int:
    """main(), restricted to the manifests a test actually staged."""
    original = dict(fetch._MANIFESTS_BY_CORPUS)
    fetch._MANIFESTS_BY_CORPUS["figlib"] = manifests
    try:
        return fetch.main(["--corpus", "figlib", "--data-root", str(data_root)])
    finally:
        fetch._MANIFESTS_BY_CORPUS.clear()
        fetch._MANIFESTS_BY_CORPUS.update(original)


# ---------------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------------


def test_a_plus_in_a_frame_name_is_percent_encoded() -> None:
    # Every post-ignition FIgLib frame carries one, so getting this wrong would mean the
    # 40 frames that matter most are the 40 that fail to download.
    url = fetch.frame_url("https://x/seq/", "1719093958_+00060.jpg")
    assert url == "https://x/seq/1719093958_%2B00060.jpg"
    assert "+" not in url


def test_a_frame_name_may_carry_its_own_directory() -> None:
    # An MTA confusion window spans two quarter directories, so its frame names contain
    # slashes that must survive encoding as separators.
    url = fetch.frame_url("https://x/cam/", "large/20260726/Q3/1785071971.jpg")
    assert url == "https://x/cam/large/20260726/Q3/1785071971.jpg"


def test_every_committed_dev_frame_encodes_to_a_plausible_url() -> None:
    manifest = json.loads((_DATA / "manifest_dev.json").read_text())
    planned = fetch.plan_manifest(manifest, _DATA / "manifest_dev.json")
    assert len(planned) == sum(w["n_frames"] for w in manifest["windows"])
    assert all(p.url.startswith(manifest["base_url"]) for p in planned)
    assert all("+" not in p.url for p in planned)
    assert sum("%2B" in p.url for p in planned) > 0


def test_a_window_whose_declared_length_disagrees_with_its_list_is_rejected() -> None:
    manifest = _manifest("dev", n_windows=1, n_frames=3)
    manifest["windows"][0]["n_frames"] = 81
    with pytest.raises(fetch.FetchError, match="disagree"):
        fetch.plan_manifest(manifest, pathlib.Path("x.json"))


# ---------------------------------------------------------------------------------
# The receipt round-trip
# ---------------------------------------------------------------------------------


def test_a_first_fetch_writes_a_hash_and_a_size_for_every_file(tree) -> None:
    data_root, archive = tree
    assert _run(data_root) == 0
    receipt = json.loads((data_root / fetch.RECEIPT_NAME).read_text())
    assert receipt["n_files"] == 20 == len(archive.gets)
    for entry in receipt["files"].values():
        assert len(entry["sha256"]) == 64 and entry["bytes"] > 0


def test_a_second_run_downloads_nothing_and_still_verifies_everything(tree) -> None:
    data_root, archive = tree
    assert _run(data_root) == 0
    archive.gets.clear()
    assert _run(data_root) == 0
    # Zero requests is the resumability claim. It is also the politeness claim: `make
    # fetch` is documented as re-runnable, and a re-run that re-downloads the whole corpus
    # would be a re-run nobody is allowed to do.
    assert archive.gets == []


def test_one_deleted_frame_is_the_only_one_re_fetched(tree) -> None:
    data_root, archive = tree
    assert _run(data_root) == 0
    receipt = json.loads((data_root / fetch.RECEIPT_NAME).read_text())
    victim = sorted(receipt["files"])[7]
    (data_root / victim).unlink()
    archive.gets.clear()
    assert _run(data_root) == 0
    assert len(archive.gets) == 1


def test_a_byte_flipped_on_disk_fails_and_leaves_the_receipt_alone(tree) -> None:
    data_root, archive = tree
    assert _run(data_root) == 0
    receipt_path = data_root / fetch.RECEIPT_NAME
    before = receipt_path.read_text()
    victim = data_root / sorted(json.loads(before)["files"])[0]
    victim.write_bytes(victim.read_bytes() + b"\x00")

    assert _run(data_root) == 1
    # Rewriting the receipt here is the one failure that would make every later run pass:
    # the corrupted file would become the recorded truth and nothing would ever say so.
    assert receipt_path.read_text() == before


def test_an_archive_that_serves_different_bytes_fails_and_writes_nothing(tree) -> None:
    data_root, archive = tree
    assert _run(data_root) == 0
    receipt = json.loads((data_root / fetch.RECEIPT_NAME).read_text())
    victim_relative = sorted(receipt["files"])[3]
    victim = data_root / victim_relative
    victim.unlink()
    sequence_id, frame = victim_relative.split("/")[-2:]
    manifest = json.loads((data_root / "manifest_dev.json").read_text())
    url = fetch.frame_url(manifest["base_url"] + sequence_id + "/", frame)
    archive.files[url] = _jpeg(9999)

    assert _run(data_root) == 1
    # The bad bytes must not reach the disk either. Writing them would mean the NEXT run
    # compares them against the receipt, fails again, and looks like disk corruption
    # rather than an archive that changed.
    assert not victim.exists()


def test_a_truncated_file_from_before_the_receipt_is_re_fetched(tree) -> None:
    data_root, archive = tree
    manifest = json.loads((data_root / "manifest_dev.json").read_text())
    window = manifest["windows"][0]
    partial = (
        data_root
        / "corpus/figlib"
        / window["sequence_id"]
        / window["frames"][0]
    )
    partial.parent.mkdir(parents=True, exist_ok=True)
    # A JPEG cut off mid-transfer: right prefix, no end-of-image marker. On the first
    # fetch there is no recorded hash to catch it, so the only remaining signal is the
    # file's own structure.
    partial.write_bytes(_jpeg(1)[:200])

    assert _run(data_root) == 0
    assert partial.read_bytes().endswith(b"\xff\xd9")


def test_a_missing_manifest_is_a_hard_stop_not_an_empty_fetch(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    assert fetch.main(["--corpus", "pyrosdis", "--data-root", str(data_root)]) == 2
    assert not (data_root / fetch.RECEIPT_NAME).exists()


def test_a_missing_manifest_stops_the_group_before_a_byte_is_requested(tree) -> None:
    """`--corpus all` must not spend the FIgLib download and then fail on pyrosdis.

    Lazily, that run pulled every DEV and HOLD frame, hit the absent manifest, exited
    non-zero WITHOUT writing the receipt, and did the same on the next attempt: a corpus
    that downloads forever and is never recorded. The check has to happen before the pool
    opens, so this asserts the request count and not just the exit code.
    """
    data_root, archive = tree
    original = dict(fetch._MANIFESTS_BY_CORPUS)
    fetch._MANIFESTS_BY_CORPUS["figlib"] = ("manifest_dev.json", "manifest_absent.json")
    try:
        assert fetch.main(["--corpus", "figlib", "--data-root", str(data_root)]) == 2
    finally:
        fetch._MANIFESTS_BY_CORPUS.clear()
        fetch._MANIFESTS_BY_CORPUS.update(original)
    assert archive.gets == []
    assert not (data_root / fetch.RECEIPT_NAME).exists()


def test_an_unreachable_archive_fails_loudly_instead_of_a_traceback(tree, monkeypatch) -> None:
    # A FetchError raised inside the worker pool used to escape main() as a traceback, so
    # the one failure an operator is most likely to hit was the one with no "FAILED" line.
    data_root, _ = tree
    monkeypatch.setattr(fetch, "RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(
        fetch,
        "http_get",
        lambda url, timeout=0.0: (_ for _ in ()).throw(TimeoutError("network down")),
    )
    assert _run(data_root) == 2
    assert not (data_root / fetch.RECEIPT_NAME).exists()


def test_a_window_with_no_frame_list_is_a_fetch_error_not_a_key_error() -> None:
    manifest = _manifest("dev", n_windows=1, n_frames=2)
    del manifest["windows"][0]["frames"]
    with pytest.raises(fetch.FetchError, match="has no 'frames'"):
        fetch.plan_manifest(manifest, pathlib.Path("x.json"))


# ---------------------------------------------------------------------------------
# Fetching is not reading
# ---------------------------------------------------------------------------------


def test_fetching_hold_never_appends_to_the_hold_access_log(tree, tmp_path) -> None:
    data_root, _ = tree
    results = tmp_path / "results"
    results.mkdir()
    log = results / "hold_access.log"

    assert _run(data_root, ("manifest_hold.json",)) == 0

    # HOLD bytes are on disk...
    hold = json.loads((data_root / "manifest_hold.json").read_text())
    first = hold["windows"][0]
    assert (
        data_root / "corpus/figlib" / first["sequence_id"] / first["frames"][0]
    ).is_file()
    # ...and nothing recorded a read, because writing bytes is not looking at them.
    assert not log.exists()


def test_the_script_cannot_touch_the_hold_access_log_at_all() -> None:
    # Mechanical, and stronger than the behavioural test above: the behavioural one only
    # covers the paths a test happens to walk, and this one covers the file.
    source = (_SCRIPTS / "fetch.py").read_text()
    assert "hold_access" in source, "the rule should be stated in the docstring"
    assert source.count("hold_access") == 1, "stated once, never used as a path"


# ---------------------------------------------------------------------------------
# Politeness
# ---------------------------------------------------------------------------------


def test_concurrency_is_capped_however_it_is_asked_for(tree) -> None:
    data_root, _ = tree
    assert fetch.MAX_CONCURRENCY <= 8
    receipt = fetch.read_receipt(data_root / fetch.RECEIPT_NAME)
    planned = fetch.plan_manifest(
        json.loads((data_root / "manifest_dev.json").read_text()),
        data_root / "manifest_dev.json",
    )
    # 64 is what an impatient operator types. The pool must not honour it.
    result = fetch.fetch_planned(planned, data_root, receipt, concurrency=64)
    assert result.downloaded == len(planned)


def test_the_user_agent_names_the_project_and_a_way_back() -> None:
    assert "PlumeRank" in fetch.USER_AGENT
    assert "repository" in fetch.USER_AGENT
    assert fetch.USER_AGENT.isascii()


def test_a_transient_failure_is_retried_and_a_permanent_one_is_not(monkeypatch) -> None:
    monkeypatch.setattr(fetch, "RETRY_BACKOFF_SECONDS", 0.0)
    calls: list[str] = []

    def flaky(url: str, timeout: float = 0.0) -> bytes:
        calls.append(url)
        if len(calls) < 3:
            raise TimeoutError("connection reset")
        return b"ok"

    monkeypatch.setattr(fetch, "http_get", flaky)
    assert fetch.fetch_with_retry("https://x/a.jpg") == b"ok"
    assert len(calls) == 3

    import urllib.error

    def gone(url: str, timeout: float = 0.0) -> bytes:
        calls.append(url)
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(fetch, "http_get", gone)
    calls.clear()
    with pytest.raises(fetch.FetchError, match="403"):
        fetch.fetch_with_retry("https://x/b.jpg")
    # Asking a CDN three times for a key it has already said does not exist is the exact
    # behaviour that gets a research group's bucket rate-limited.
    assert len(calls) == 1


def test_transport_failures_eventually_give_up_loudly(monkeypatch) -> None:
    monkeypatch.setattr(fetch, "RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(
        fetch, "http_get", lambda url, timeout=0.0: (_ for _ in ()).throw(TimeoutError("x"))
    )
    with pytest.raises(fetch.FetchError, match="after 3 attempts"):
        fetch.fetch_with_retry("https://x/c.jpg")


# ---------------------------------------------------------------------------------
# The micro corpus
# ---------------------------------------------------------------------------------


def test_micro_frames_are_written_at_working_resolution(tree) -> None:
    data_root, _ = tree
    assert _run(data_root) == 0
    micro = json.loads((data_root / "micro" / "manifest_micro.json").read_text())
    assert micro["arm"] == "micro"
    assert micro["n_windows"] == fetch.MICRO_SEQUENCES

    for window in micro["windows"]:
        for frame in window["frames"]:
            image = cv2.imread(str(data_root / "micro" / window["sequence_id"] / frame))
            assert image is not None
            assert image.shape[:2] == (WORK_HEIGHT, WORK_WIDTH)


def test_micro_is_cut_from_five_distinct_cameras(tree) -> None:
    data_root, _ = tree
    assert _run(data_root) == 0
    micro = json.loads((data_root / "micro" / "manifest_micro.json").read_text())
    cameras = {w["camera_id"] for w in micro["windows"]}
    # Five windows off one camera would exercise a roster of one, and the ranking is the
    # thing `make verify` is supposed to recompute.
    assert len(cameras) == fetch.MICRO_SEQUENCES


def test_the_micro_manifest_uses_the_committed_schema(tree) -> None:
    data_root, _ = tree
    assert _run(data_root) == 0
    micro = json.loads((data_root / "micro" / "manifest_micro.json").read_text())
    committed = json.loads((_DATA / "manifest_dev.json").read_text())
    # Same keys, so anything that reads a manifest reads this one too without a branch.
    assert set(committed).issubset(set(micro))
    window_keys = set(committed["windows"][0])
    assert all(window_keys.issubset(set(w)) for w in micro["windows"])


def test_provenance_carries_the_credit_reference_verbatim(tree) -> None:
    data_root, _ = tree
    assert _run(data_root) == 0
    text = (data_root / "micro" / "PROVENANCE.md").read_text()
    dev = json.loads((data_root / "manifest_dev.json").read_text())
    assert dev["credit"] in text
    assert dev["licence_note"] in text
    # The three-minute command must never be described as reproducing the headline; the
    # only copy of the pixels that ships is also the easiest place to imply it does.
    assert "make fetch && make bench" in text


def test_the_real_micro_selection_is_about_four_hundred_frames() -> None:
    # Pure arithmetic on the committed manifest -- no network, no pixels. It is what makes
    # `make verify` a three-minute command rather than a 95-minute one.
    #
    # 402 and not 405: 21 of the 30 DEV windows are 80 frames because the archive is
    # missing one minute in each, and the manifest says 80 rather than padding to 81. The
    # test asserts the real number so that a future manifest change that quietly drops a
    # window shows up here instead of in a benchmark denominator.
    dev = json.loads((_DATA / "manifest_dev.json").read_text())
    chosen = fetch.micro_sequences(dev)
    assert len(chosen) == 5
    assert len({w["camera_id"] for w in chosen}) == 5
    total = sum(len(w["frames"]) for w in chosen)
    assert total == 402
    assert all(len(w["frames"]) in (FRAMES_PER_WINDOW - 1, FRAMES_PER_WINDOW) for w in chosen)


def test_micro_refuses_to_build_from_frames_that_are_not_there(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(fetch.FetchError, match="missing or unreadable"):
        fetch.build_micro(_manifest("dev", n_windows=5, n_frames=2), data_root)

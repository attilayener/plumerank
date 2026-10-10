"""The HTTP surface, asserted without a network and without a deployed box.

What these tests are for
------------------------
Two of them exist because a claim in the submission depends on them. ``/build-info`` is
what turns "we integrated OpenCV 5" from an assertion into something a judge checks with
one ``curl``, so the test asserts the two things that ``check_submission_readiness.py``
greps for: a version starting ``5.`` and the verbatim build information. And the route
inventory is asserted as an exact set, because a fifth undocumented route is a surface
nobody wrote down, nobody proxied in the Caddyfile and nobody hardened -- on a box that is
public and unattended for fourteen days.

The rest assert the *degraded* path, which is the one a judge is most likely to meet: a
clean clone has no model, no corpus and no results file. Every one of those must produce a
designed state rather than a stack trace, and none of them may produce a fabricated row.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import threading
import time
from typing import Any, Callable, Iterator

import numpy as np
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from api import main
from api.main import app
from plumerank.types import Component, Sequence, Stats, Track, WatchEntry

#: The four documented routes, written out rather than imported from the app.
#: A test that read them from the thing it is testing would agree with any fifth route the
#: app grew, which is the one failure this test exists to catch.
DOCUMENTED_ROUTES: frozenset[str] = frozenset(
    {"/build-info", "/health", "/api/watchlist", "/api/benchmark"}
)


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """A client whose context manager runs the app's lifespan, so shutdown really happens."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def reprobe() -> Iterator[None]:
    """Force one fresh probe, then put the process-wide singleton back exactly as it was.

    ``REPLAY`` is one object for the life of the process by design. A test that makes it
    re-probe against a fixture directory would otherwise leave that verdict -- and that
    roster -- behind for whatever runs next, and the failure would land in the other test.
    """
    replay = main.REPLAY
    names = (
        "_probed",
        "_unavailable",
        "_cameras",
        "_threshold",
        "_threshold_source",
        "_classifier",
        "_credit",
        "_frame_count",
        "_t0_index",
    )
    saved = {name: getattr(replay, name) for name in names}
    replay._probed = False
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(replay, name, value)


# ---------------------------------------------------------------------------------
# /build-info
# ---------------------------------------------------------------------------------


def test_build_info_publishes_a_verifiable_opencv_5(client: TestClient) -> None:
    response = client.get("/build-info")
    assert response.status_code == 200

    body = response.json()
    assert body["opencv"]["version"].startswith("5."), body["opencv"]["version"]
    # The verbatim build log is the half that cannot be faked by a version string: it names
    # the modules compiled in and the toolchain that produced them.
    assert "General configuration" in body["opencv"]["build_information"]
    assert body["opencv"]["modules"]["dnn"]["ok"] is True
    assert body["opencv"]["modules"]["bgsegm"]["ok"] is True


def test_build_info_uses_the_key_spellings_the_deploy_docs_grep_for(client: TestClient) -> None:
    """``launch.md`` section 7 and ``bootstrap.sh`` read these three by name, at the top level.

    ``amiId`` or ``git_sha`` would break a written verification step silently -- the curl
    would still return 200 and the field the reader was told to compare would simply not be
    there.
    """
    body = client.get("/build-info").json()
    for key in ("instance_id", "ami_id", "commit"):
        assert key in body, f"{key} missing from /build-info"


def test_build_info_answers_with_no_model_no_results_and_no_corpus(
    client: TestClient, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route ships before the pipeline does, so an empty box must not 500 it."""
    monkeypatch.setattr(main, "ONNX_PATH", tmp_path / "nope.onnx")
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path / "no-results")
    monkeypatch.setattr(main, "CORPUS_DIR", tmp_path / "no-corpus")

    response = client.get("/build-info")
    assert response.status_code == 200

    body = response.json()
    assert body["onnx_sha256"] is None
    assert body["dnn"]["loaded"] is False
    # An absent model is reported as absent rather than as a quiet fallback to
    # ENGINE_CLASSIC, which covers a fraction of the ONNX spec and would be a different
    # system described by the same JSON.
    assert body["dnn"]["engine_selected"] is None
    assert body["dnn"]["engine_requested"] == "ENGINE_NEW"
    assert body["constants"]["sha256"]


# ---------------------------------------------------------------------------------
# The route inventory
# ---------------------------------------------------------------------------------


def test_exactly_the_four_documented_routes_exist() -> None:
    """Every route on the app, not only the ones that happen to be ``APIRoute``.

    ``check_claims.py`` introspects ``app.routes`` and exits non-zero on a documented route
    it cannot find; this is the other direction -- an *undocumented* one. Filtering to
    ``APIRoute`` would let a ``Mount`` (a static-files mount, an OpenAPI schema route) past
    the assertion, and a mount is exactly the shape the fifth surface would take.
    """
    paths = {route.path for route in app.routes}
    assert paths == DOCUMENTED_ROUTES, paths
    assert all(isinstance(route, APIRoute) for route in app.routes), [
        type(route).__name__ for route in app.routes
    ]


def test_there_is_no_upload_route() -> None:
    """``POST /api/sequence`` was cut, and the cut is a security property, not a schedule one.

    Asserted separately from the inventory above so that a future re-add fails with the
    reason attached rather than as an anonymous set mismatch.
    """
    for route in app.routes:
        if isinstance(route, APIRoute):
            assert "POST" not in route.methods, f"{route.path} accepts POST"


def test_cors_allows_a_cross_origin_get_but_not_other_methods(client: TestClient) -> None:
    """README.md documents pointing a local `npm run dev` at the live box -- a genuinely
    cross-origin GET from a browser. Without CORS headers that fails silently in-browser
    (a browser check caught it: 'blocked by CORS policy', zero console errors on a server-rendered
    curl check). Scoped to GET only -- there is nothing here for a wider policy to protect."""
    resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.headers["access-control-allow-origin"] == "*"

    preflight = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    allowed = preflight.headers.get("access-control-allow-methods", "")
    assert "POST" not in allowed


# ---------------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------------


def test_health_is_well_formed_before_any_replay_has_run(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["uptime_seconds"] >= 0.0
    # Nothing is replaying yet: "idle" on a box that can run, "unavailable" on one that
    # cannot -- never "running".
    expected = "idle" if body["live_pipeline"]["available"] else "unavailable"
    assert body["replay"]["state"] == expected
    assert body["replay"]["viewers"] == 0
    assert body["replay"]["last_tick"] is None
    assert body["replay"]["ticks"] == 0

    live = body["live_pipeline"]
    assert isinstance(live["available"], bool)
    # The degraded state is designed, so it is asserted as a pair: unavailable means an
    # empty roster AND a stated reason. A wall that reported rows it could not compute, or
    # a silence with no explanation, are the two failures this covers.
    if live["available"]:
        assert live["cameras"] > 0 and live["threshold"] is not None
    else:
        assert live["cameras"] == 0
        assert live["detail"].startswith("live pipeline unavailable -- ")


# ---------------------------------------------------------------------------------
# /api/benchmark
# ---------------------------------------------------------------------------------


def test_benchmark_404s_cleanly_when_no_results_file_exists(
    client: TestClient, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)

    response = client.get("/api/benchmark")
    assert response.status_code == 404
    assert "bench" in response.json()["detail"]


def test_benchmark_serves_the_committed_bytes_unchanged(
    client: TestClient, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-for-byte, not field-for-field.

    The endpoint, the committed file and the printed block have to agree about the
    headline; two of the three agree forever only if they are literally the same bytes. A
    re-serialised object would round a float differently the day the writer changed and
    nothing would fail.
    """
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    raw = b'{\n  "headline": {"p50_minutes": 7.0},\n  "calibration": {"threshold": 0.6125}\n}\n'
    (tmp_path / "bench_2026-10-08.json").write_bytes(raw)

    response = client.get("/api/benchmark")
    assert response.status_code == 200
    assert response.content == raw
    assert response.headers["x-plumerank-results-file"].endswith("bench_2026-10-08.json")


def test_benchmark_serves_the_newest_dated_file(
    client: TestClient, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    (tmp_path / "bench_2026-10-08.json").write_bytes(b'{"run": "old"}')
    (tmp_path / "bench_2026-10-19.json").write_bytes(b'{"run": "new"}')

    assert client.get("/api/benchmark").json() == {"run": "new"}


def test_benchmark_fails_loudly_on_a_corrupt_results_file(
    client: TestClient, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Present-and-wrong is a different fact from absent, and gets a different status."""
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    (tmp_path / "bench_2026-10-08.json").write_bytes(b"{not json")

    response = client.get("/api/benchmark")
    assert response.status_code == 500
    assert "not valid JSON" in response.json()["detail"]


# ---------------------------------------------------------------------------------
# The threshold is read, never invented
# ---------------------------------------------------------------------------------


def test_threshold_comes_from_the_committed_benchmark_and_has_no_default(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wall running on a made-up threshold is the most expensive wrong answer here.

    It would look exactly like one calibrated to 2.0 alerts per camera-night while being
    calibrated to nothing, so the absence of a solved threshold must degrade the pipeline
    rather than fall back.
    """
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    assert main._calibrated_threshold() == (None, None)

    (tmp_path / "bench_2026-10-08.json").write_text(json.dumps(_held_out(0.61)), encoding="utf-8")
    value, source = main._calibrated_threshold()
    assert value == pytest.approx(0.61)
    assert source is not None and source.endswith(":calibration.threshold")


def _held_out(threshold: float, **over: Any) -> dict[str, Any]:
    """The smallest payload _heldout_rejection accepts: HOLD, solved on CONFUSE, these constants."""
    payload: dict[str, Any] = {
        "arm": "hold",
        "calibration": {"threshold": threshold, "realised_on": "confuse"},
        "run": {"constants_sha256": main.freeze_hash()},
    }
    payload.update(over)
    return payload


def test_an_ablation_name_never_calibrates_the_wall(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bench_<date>_s4off.json sorts after bench_<date>.json; only the exact name is read."""
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    (tmp_path / "bench_2026-10-10.json").write_text(json.dumps(_held_out(0.61)), encoding="utf-8")
    (tmp_path / "bench_2026-10-10_s4off.json").write_text(json.dumps(_held_out(0.99)), encoding="utf-8")
    assert main._benchmark_file() == tmp_path / "bench_2026-10-10.json"
    assert main._calibrated_threshold()[0] == pytest.approx(0.61)


@pytest.mark.parametrize(
    ("over", "why"),
    [
        ({"arm": "dev"}, "an in-sample arm under the held-out name"),
        ({"calibration": {"threshold": 0.61, "realised_on": "dev"}}, "not solved on CONFUSE"),
        ({"run": {"constants_sha256": "0" * 64}}, "solved under other constants"),
    ],
)
def test_a_held_out_file_that_cannot_calibrate_this_wall_is_refused_and_named(
    client: TestClient,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    reprobe: None,
    over: dict[str, Any],
    why: str,
) -> None:
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    # Past the corpus check on a clean clone (data/corpus is gitignored), as in the test above.
    monkeypatch.setattr(main, "_wall_sequences", lambda: ([object()], None))
    (tmp_path / "bench_2026-10-08.json").write_text(
        json.dumps(_held_out(0.61, **over)), encoding="utf-8"
    )
    assert main._calibrated_threshold()[0] is None, why
    live = client.get("/health").json()["live_pipeline"]
    assert live["available"] is False
    # Found-and-refused is not "has not been published": that sends an operator to re-run HOLD.
    assert "is committed but cannot calibrate" in live["detail"], live["detail"]
    assert "bench_2026-10-08.json" in live["detail"], live["detail"]


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ('{"calibration": {"threshold": null}}', "a JSON null"),
        ('{"calibration": {"threshold": {"dev": 0.61}}}', "an object"),
        ('{"threshold": NaN}', "json.loads accepts a bare NaN and float() takes it"),
        ('{"threshold": Infinity}', "and an Infinity"),
    ],
)
def test_an_unusable_threshold_degrades_the_wall_instead_of_500ing_health(
    client: TestClient,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    reprobe: None,
    body: str,
    why: str,
) -> None:
    """``/health`` is polled daily for a fortnight and its job is to REPORT a broken box.

    Every value here reached ``float()`` unguarded before. The first two raised out of
    ``probe`` and turned ``/health`` into a 500 -- the route explaining the outage becoming
    part of it. The last two converted cleanly and installed a threshold no hazard can ever
    cross, so the wall would have run all night ranking cameras that could not alert while
    ``meta`` published a calibrated operating point beside them.
    """
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    # The probe checks the corpus before the threshold, and data/corpus is gitignored, so on
    # a clean clone it would stop at "run `make fetch`" and never reach the branch under test.
    # One stand-in sequence gets it past that check; the probe returns on the threshold
    # before it ever touches a sequence.
    monkeypatch.setattr(main, "_wall_sequences", lambda: ([object()], None))
    (tmp_path / "bench_2026-10-08.json").write_text(body, encoding="utf-8")

    assert main._calibrated_threshold()[0] is None, why

    response = client.get("/health")
    assert response.status_code == 200, why
    live = response.json()["live_pipeline"]
    assert live["available"] is False
    assert live["threshold"] is None
    # The reason names the file it found rather than telling an operator to spend ninety-five
    # minutes running `make bench` against a results file that is already sitting there.
    assert "is not a usable number" in live["detail"], live["detail"]
    assert "bench_2026-10-08.json" in live["detail"], live["detail"]


def test_probe_reports_an_unforeseen_failure_rather_than_raising(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, reprobe: None
) -> None:
    """The catch-all in ``probe`` is a route contract, so it is asserted like one."""

    def boom() -> tuple[list[Any], str | None]:
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(main, "_wall_sequences", boom)

    response = client.get("/health")
    assert response.status_code == 200
    live = response.json()["live_pipeline"]
    assert live["available"] is False
    assert live["cameras"] == 0
    assert "the disk went away" in live["detail"], live["detail"]


# ---------------------------------------------------------------------------------
# /api/watchlist -- SSE, driven at the ASGI level
# ---------------------------------------------------------------------------------
#
# Not through TestClient, and the reason is a property of the tools rather than a
# preference: starlette's TestClient (testclient.py: `portal.call(self.app, ...)`) and
# httpx's ASGITransport both run the application to COMPLETION and buffer the whole body
# before they hand back a response. An endless stream never completes, so a read of the
# first event under either of them deadlocks instead of failing. Driving the ASGI callable
# directly is the only in-process way to read one event off a live stream -- and it is
# also the only way to deliver a real `http.disconnect`, which is precisely the behaviour
# the viewer count depends on.


class _Viewer:
    """One SSE client, speaking ASGI: open, read events, disconnect."""

    def __init__(self, path: str = "/api/watchlist") -> None:
        self.scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"testserver"), (b"accept", b"text/event-stream")],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        }
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self._chunks: asyncio.Queue[bytes] = asyncio.Queue()
        self._disconnect = asyncio.Event()
        self._request_sent = False
        self._task: asyncio.Task[None] | None = None

    async def _receive(self) -> dict[str, Any]:
        if not self._request_sent:
            self._request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await self._disconnect.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = {k.decode(): v.decode() for k, v in message["headers"]}
        elif message["type"] == "http.response.body" and message.get("body"):
            self._chunks.put_nowait(message["body"])

    async def __aenter__(self) -> "_Viewer":
        self._task = asyncio.create_task(app(self.scope, self._receive, self._send))
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._disconnect.set()
        if self._task is not None:
            await asyncio.wait_for(self._task, timeout=5.0)

    async def next_event(self, timeout: float = 10.0) -> tuple[str, dict[str, Any]]:
        """The next real event, skipping the keep-alive comment frames."""
        deadline = time.monotonic() + timeout
        while True:
            chunk = await asyncio.wait_for(
                self._chunks.get(), timeout=max(0.1, deadline - time.monotonic())
            )
            text = chunk.decode()
            if text.startswith(":"):
                continue
            name = text.split("event: ", 1)[1].split("\n", 1)[0]
            data = text.split("data: ", 1)[1].split("\n", 1)[0]
            return name, json.loads(data)


async def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


async def _drain_replay() -> None:
    """Let the loop finish inside the test's own event loop.

    A replay task left pending when ``asyncio.run`` tears the loop down would be destroyed
    mid-await and would leave the process-wide singleton holding a task bound to a dead
    loop -- which the next test would inherit.
    """
    await _wait_for(lambda: main.REPLAY._task is None, timeout=5.0)
    await main.REPLAY.stop()


@pytest.fixture()
def quick_idle_stop() -> Iterator[None]:
    """Shorten the loop's idle grace period for the duration of one test.

    The same loop, ticking the same way -- only the wait after the last viewer leaves is
    shortened, because the shipped 120 s is a CPU-credit decision and not a behaviour this
    suite can afford to sit through. Nothing about the pipeline is replaced or stubbed.
    """
    original = main.REPLAY.idle_stop_seconds
    main.REPLAY.idle_stop_seconds = 0.05
    try:
        yield
    finally:
        main.REPLAY.idle_stop_seconds = original


def test_the_stream_sends_meta_first(quick_idle_stop: None) -> None:
    async def run() -> tuple[str, dict[str, Any], dict[str, str], int | None]:
        async with _Viewer() as viewer:
            name, payload = await viewer.next_event()
            headers, status = viewer.headers, viewer.status
        await _drain_replay()
        return name, payload, headers, status

    name, meta, headers, status = asyncio.run(run())

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert name == "meta"
    assert meta["replay_id"]
    assert meta["evidence_base"] == "/evidence"
    assert meta["threshold_hash"] == main.freeze_hash()
    # The key names web/lib/types.ts reads. Asserted by name because a rename here is
    # invisible from this side: the frontend would fall back to its designed "the stream
    # did not say" state over a stream that did say, and nothing would raise.
    for key in (
        "cameras",
        "credit",
        "frame_count",
        "t0_index",
        "seconds_per_frame",
        "warmup_frames",
        "top_k",
        "budget_target",
        "budget_per_camera",
        "drift_tolerance",
        "threshold",
        "threshold_hash",
    ):
        assert key in meta, f"{key} missing from the meta event"
    assert meta["budget_target"] == 2.0
    assert meta["seconds_per_frame"] == 60.0
    assert all(isinstance(c, str) for c in meta["cameras"])
    # Same pairing as /health: either the roster and the threshold are both real, or the
    # roster is empty and the degraded flag says why. Never rows without a threshold.
    if meta["live_pipeline"]["available"]:
        assert meta["cameras"] and meta["threshold"] is not None
    else:
        assert meta["cameras"] == []
        assert meta["threshold"] is None
        assert meta["live_pipeline"]["detail"].startswith("live pipeline unavailable -- ")


def test_the_wall_refuses_a_hold_sequence_that_is_on_disk_and_in_a_wall_manifest(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HOLD is read once, by bench.py, after the freeze -- with no exemption for the wall.

    Asserted against a manifest that *does* name a real HOLD id, with its frames *present*
    on disk, because the interesting failure is the one where nothing else would stop it.
    Reading the roster off the shipped manifests instead would pass on this box for the
    wrong reason -- the wall is degraded here, so the roster is empty and any assertion
    about what is not in it holds trivially.
    """
    hold = main._load_manifest(main.DATA_DIR / main.HOLD_MANIFEST)
    assert hold is not None, "data/manifest_hold.json is committed and must be readable"
    held_id = str(hold["windows"][0]["sequence_id"])

    corpus = tmp_path / "corpus" / "figlib"
    for sequence_id in (held_id, "open-sequence"):
        (corpus / sequence_id).mkdir(parents=True)
        (corpus / sequence_id / "0001.jpg").write_bytes(b"not a real jpeg")

    def window(sequence_id: str) -> dict[str, Any]:
        return {
            "sequence_id": sequence_id,
            "camera_id": f"cam-{sequence_id}",
            "t0_index": 40,
            "frames": ["0001.jpg"],
        }

    (tmp_path / main.HOLD_MANIFEST).write_text(
        json.dumps({"arm": "hold", "windows": [window(held_id)]}), encoding="utf-8"
    )
    (tmp_path / "manifest_dev.json").write_text(
        json.dumps(
            {
                "arm": "dev",
                "credit": "Imagery: HPWREN",
                "windows": [window(held_id), window("open-sequence")],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "CORPUS_DIR", corpus)

    sequences, credit = main._wall_sequences()

    assert [s.sequence_id for s in sequences] == ["open-sequence"], (
        f"{held_id} is a HOLD sequence and reached the wall"
    )
    # The non-held window still loads, so the test fails for the HOLD rule rather than
    # because the fixture happened to be unreadable.
    assert credit == "Imagery: HPWREN"


def test_the_loop_starts_on_connect_and_is_torn_down_on_disconnect(
    quick_idle_stop: None,
) -> None:
    """The whole burstable-CPU argument, asserted.

    A viewer that vanishes must decrement the count. If it does not, the loop keeps a
    phantom watcher and runs for the entire unattended judging window -- which is the exact
    failure the on-demand design exists to prevent.
    """

    async def run() -> tuple[dict[str, Any], bool, bool]:
        assert main.REPLAY._task is None, "no loop may run before anyone connects"
        async with _Viewer() as viewer:
            await viewer.next_event()
            attached = main.REPLAY.status()
        gone = await _wait_for(lambda: main.REPLAY.status()["viewers"] == 0)
        stopped = await _wait_for(lambda: main.REPLAY._task is None)
        await _drain_replay()
        return attached, gone, stopped

    attached, gone, stopped = asyncio.run(run())
    # "running" on a box that can replay; on one that cannot the loop still holds the
    # stream open, and the published state says why it computes nothing.
    assert attached["state"] in ("running", "unavailable")
    assert attached["viewers"] == 1
    assert gone, "a disconnected viewer was still counted"
    assert stopped, "the loop outlived its last viewer"


def test_two_viewers_share_one_loop(quick_idle_stop: None) -> None:
    """One replay, process-wide: the second viewer must not start a second wall.

    Two loops would mean two judges seeing different rankings off the same box, two sets of
    background models inside 2 GiB, and the CPU-credit budget spent twice.
    """

    async def run() -> tuple[int, str, str]:
        async with _Viewer() as first:
            _, meta_one = await first.next_event()
            async with _Viewer() as second:
                _, meta_two = await second.next_event()
                viewers = main.REPLAY.status()["viewers"]
        await _drain_replay()
        return viewers, meta_one["replay_id"], meta_two["replay_id"]

    viewers, first_id, second_id = asyncio.run(run())
    assert viewers == 2
    assert first_id == second_id


def test_the_sse_encoder_refuses_a_non_finite_number() -> None:
    """A bare ``NaN`` is not JSON: the browser's parse throws and the wall stops silently.

    Raising here is the loud version of that failure, one stage earlier and with a name.
    """
    with pytest.raises(ValueError):
        main._sse("tick", {"hazard": float("nan")})


# ---------------------------------------------------------------------------------
# The loop's own arithmetic, without waiting for a replay
# ---------------------------------------------------------------------------------


def test_an_idle_loop_stops_without_computing_anything() -> None:
    """The idle branch must not tick, because a tick is the thing that costs CPU credits."""

    async def run() -> int:
        replay = main.Replay()
        replay.idle_stop_seconds = 0.05
        replay.probe()
        started = time.monotonic()
        await replay._run()
        assert time.monotonic() - started < 2.0
        return replay.status()["ticks"]

    assert asyncio.run(run()) == 0


# ---------------------------------------------------------------------------------
# Evidence crops
# ---------------------------------------------------------------------------------


def _one_row() -> tuple[WatchEntry, dict[str, Any], dict[str, Any]]:
    """One watch-list row with a real component, a real track and a real frame."""
    component = Component(x=100, y=110, w=48, h=36, area=1_100, cx=124.0, cy=128.0)
    track = Track(track_id=1)
    track.add(0, component, window=10)
    stats = Stats(s1=0.62, s2=0.41, s3=0.55, s4=0.18, raw_s3=1.9)
    entry = WatchEntry(
        rank=1,
        camera_id="om-w-mobo-c",
        hazard=0.71,
        s1=stats.s1,
        s2=stats.s2,
        s3=stats.s3,
        s4=stats.s4,
        escalated=False,
        alerted=True,
        budget_spent=1,
    )
    frame = np.full((main.WORK_HEIGHT, main.WORK_WIDTH, 3), 90, dtype=np.uint8)
    return entry, {entry.camera_id: frame}, {entry.camera_id: (component, track, stats)}


def test_an_evidence_url_is_never_reused_across_a_replay_restart(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same path, new pixels: the ``?t=`` suffix is the only thing stopping a stale crop.

    The replay id is fixed for the life of the process and the frame index cycles, so the
    JPEG at a given path is overwritten roughly once a camera-night. When the loop stops for
    want of viewers and a returning judge starts it again, a cache-buster that had been
    reset to zero would hand that judge the browser's copy of the *previous* session's crop
    underneath this session's ranking -- the exact failure the suffix exists to prevent, and
    one that looks like a working wall.
    """
    monkeypatch.setattr(main, "EVIDENCE_DIR", tmp_path)
    entry, frames, picked = _one_row()
    replay = main.Replay()

    first = replay._write_evidence([entry], frames, picked, 7)[0]
    replay._reset_replay()
    second = replay._write_evidence([entry], frames, picked, 7)[0]

    assert first.evidence_url is not None and second.evidence_url is not None
    prefix = f"/evidence/{replay.replay_id}/7/{entry.camera_id}.jpg?t="
    assert first.evidence_url.startswith(prefix), first.evidence_url
    assert second.evidence_url.startswith(prefix), second.evidence_url
    assert first.evidence_url != second.evidence_url
    # The URL has to name a file that is really there: Caddy serves this path off disk and
    # a 404 under a ranked row is a row a dispatcher cannot overrule.
    assert (tmp_path / replay.replay_id / "7" / f"{entry.camera_id}.jpg").is_file()


def test_a_row_with_no_component_gets_no_evidence_url(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A quiet camera stays on the wall with a null thumbnail rather than a borrowed one."""
    monkeypatch.setattr(main, "EVIDENCE_DIR", tmp_path)
    entry, frames, _ = _one_row()
    quiet = main.Replay()._write_evidence([entry], frames, {}, 7)[0]

    assert quiet.evidence_url is None
    assert not list(tmp_path.rglob("*.jpg"))


# ---------------------------------------------------------------------------------
# One clock over windows recorded on different nights
# ---------------------------------------------------------------------------------


def _window(sequence_id: str, t0_index: int | None, n_frames: int) -> Sequence:
    return Sequence(
        sequence_id=sequence_id,
        camera_id=f"cam-{sequence_id}",
        arm="dev" if t0_index is not None else "confuse",
        frame_paths=[f"{sequence_id}/{i:04d}.jpg" for i in range(n_frames)],
        t0_index=t0_index,
    )


def test_every_camera_ignites_on_the_same_replay_index() -> None:
    """The wall renders ONE "T+n since ignition" for the whole roster, so it must be true of all of it.

    FIgLib's convention is 40 frames of lead, but 12 of the 30 committed DEV windows are a
    frame short of it and carry ``t0_index: 39``. Replaying each window from its own frame 0
    against one published ``t0_index`` therefore captions 40% of the roster a whole minute
    wrong -- on a wall whose headline number is measured in minutes.
    """
    sequences = [
        _window("late", 40, 81),
        _window("early", 39, 80),
        _window("late-short", 40, 80),
        _window("no-ignition", None, 90),
    ]

    cameras, shared, frame_count = main._align(sequences)

    # The smallest lead on the roster, because a window cannot be handed frames in front of
    # the ones it has.
    assert shared == 39
    for camera in cameras:
        if camera.sequence.t0_index is None:
            # Centre-trimmed, as bench.py's unify_windows cuts the calibration windows.
            assert camera.frame_offset == (90 - 79) // 2
            continue
        assert camera.sequence.t0_index - camera.frame_offset == shared, camera.camera_id
    # The shortest window AFTER its offset: 80 - 1 for "late-short" is what bounds the night.
    assert frame_count == 79
    for camera in cameras:
        assert frame_count - 1 + camera.frame_offset < len(camera.sequence.frame_paths)


def test_a_roster_that_never_ignites_publishes_no_ignition_clock() -> None:
    """A CONFUSE-only wall has no T+ to count, and must say so rather than pick one."""
    cameras, shared, frame_count = main._align(
        [_window("fog-a", None, 81), _window("fog-b", None, 80)]
    )

    assert shared is None
    assert [c.frame_offset for c in cameras] == [0, 0]
    assert frame_count == 80


# ---------------------------------------------------------------------------------
# Hardening -- audit 2026-10-04 P2
# ---------------------------------------------------------------------------------


def test_health_answers_head(client: TestClient) -> None:
    """Uptime monitors and `curl -I` send HEAD; a 405 there reads as an outage."""
    response = client.head("/health")
    assert response.status_code == 200


def test_the_wall_refuses_a_viewer_past_the_cap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "MAX_VIEWERS", 0)
    response = client.get("/api/watchlist")
    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    assert main.REPLAY.viewer_count == 0, "a refused viewer must never be attached"


def test_build_info_is_computed_once_per_model_and_freeze(
    client: TestClient, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"not a model")
    monkeypatch.setattr(main, "ONNX_PATH", model)
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    calls: list[int] = []
    real = main._dnn_engine

    def counting() -> dict[str, Any]:
        calls.append(1)
        return real()

    monkeypatch.setattr(main, "_dnn_engine", counting)
    first = client.get("/build-info").json()
    second = client.get("/build-info").json()
    assert first == second and len(calls) == 1
    # A new freeze on disk is a different body: the cache must notice.
    (tmp_path / "constants.sha256").write_text("deadbeef\n", encoding="ascii")
    third = client.get("/build-info").json()
    assert len(calls) == 2
    assert third["constants"]["committed_sha256"] == "deadbeef"


def test_the_wall_refuses_a_client_past_its_own_cap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One script must not be able to hold every slot of the global cap."""
    monkeypatch.setattr(main, "MAX_VIEWERS_PER_CLIENT", 0)
    response = client.get("/api/watchlist")
    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"
    assert main.REPLAY.viewer_count == 0


def test_the_client_address_behind_caddy_is_the_hop_caddy_appended() -> None:
    from starlette.requests import Request as StarletteRequest

    def request(peer: str, xff: str | None) -> StarletteRequest:
        headers = [(b"x-forwarded-for", xff.encode())] if xff else []
        return StarletteRequest(
            {"type": "http", "client": (peer, 1234), "headers": headers, "method": "GET", "path": "/"}
        )

    assert main.client_address(request("127.0.0.1", "6.6.6.6, 203.0.113.9")) == "203.0.113.9"
    assert main.client_address(request("198.51.100.4", "6.6.6.6")) == "198.51.100.4"
    assert main.client_address(request("127.0.0.1", None)) == "127.0.0.1"


def test_health_says_unavailable_rather_than_running_when_the_pipeline_is_down() -> None:
    replay = main.Replay()
    replay._probed = True
    replay._unavailable = "no calibrated threshold"
    assert replay.status()["state"] == "unavailable"


def test_keepalive_is_not_sent_on_every_poll() -> None:
    assert main.KEEPALIVE_SECONDS >= 10 * main.IDLE_POLL_SECONDS


def test_thumbnails_cover_the_top_k_plus_one_round_robin_camera(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Most tiles have no tracked component, so without these they show no image at all."""
    import numpy as np

    from plumerank.types import WatchEntry

    monkeypatch.setattr(main, "EVIDENCE_DIR", tmp_path)
    replay = main.Replay()
    keys = [f"cam{i}" for i in range(main.TOP_K + 3)]
    replay._cameras = [
        main._Camera(main.Sequence(sequence_id=k, camera_id=k, arm="dev", frame_paths=[], t0_index=None))
        for k in keys
    ]
    frames = {k: np.full((270, 480, 3), 80, np.uint8) for k in keys}
    entries = [
        WatchEntry(rank=i + 1, camera_id=k, hazard=0.5, s1=0, s2=0, s3=0, s4=0,
                   escalated=False, alerted=False, budget_spent=0)
        for i, k in enumerate(reversed(keys))
    ]
    replay._write_thumbnails(entries, frames)
    top = {e.camera_id for e in entries if e.rank <= main.TOP_K}
    assert top | {keys[0]} == set(replay._frame_urls)
    written = list((tmp_path / replay.replay_id / "frames").glob("*.jpg"))
    assert len(written) == main.TOP_K + 1
    assert all(u.startswith("/evidence/") and "?t=" in u for u in replay._frame_urls.values())


def test_the_first_tick_of_a_replay_gives_every_tile_a_picture(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_reset_replay empties _frame_urls; the first tick must not leave quiet tiles blank."""
    from plumerank.types import WatchEntry

    monkeypatch.setattr(main, "EVIDENCE_DIR", tmp_path)
    replay = main.Replay()
    keys = [f"cam{i}" for i in range(main.TOP_K + 3)]
    replay._cameras = [
        main._Camera(main.Sequence(sequence_id=k, camera_id=k, arm="dev", frame_paths=[], t0_index=None))
        for k in keys
    ]
    frames = {k: np.full((270, 480, 3), 80, np.uint8) for k in keys}
    entries = [
        WatchEntry(rank=1, camera_id=keys[0], hazard=0.5, s1=0, s2=0, s3=0, s4=0,
                   escalated=False, alerted=False, budget_spent=0)
    ]
    replay._write_thumbnails(entries, frames, fill_all=True)
    assert set(replay._frame_urls) == set(keys)
    written = list((tmp_path / replay.replay_id / "frames").glob("*.jpg"))
    assert len(written) == len(keys)


def test_repeated_cameras_get_url_safe_roster_keys() -> None:
    seqs = [
        main.Sequence(sequence_id=f"s{i}", camera_id="cp-s-mobo-c", arm="dev", frame_paths=["a"], t0_index=None)
        for i in range(2)
    ]
    cameras, _, _ = main._align(seqs)
    assert [c.camera_id for c in cameras] == ["cp-s-mobo-c~1", "cp-s-mobo-c~2"]


# ---------------------------------------------------------------------------------
# Round-2 audit: development fallback, roster, frame cache, caps, pruning, telemetry
# ---------------------------------------------------------------------------------


def _dev_payload(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "background": "mog2",
        "budget_alerts_per_camera_night": main.BUDGET_ALERTS_PER_CAMERA_NIGHT,
        "baseline": {"shipped": True},
        "calibration": {"realised_on": "confuse", "threshold": 0.84},
        "run": {"constants_sha256": main.freeze_hash()},
    }
    body.update(overrides)
    return body


def test_calibration_kind_names_the_run() -> None:
    assert main.calibration_kind(None) is None
    assert main.calibration_kind("results/bench_2026-10-20.json:calibration.threshold") == "held-out"
    assert main.calibration_kind("results/arm_dev_2026-10-04.json:calibration.threshold") == "development"


def test_the_development_fallback_uses_the_newest_valid_shipped_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    (tmp_path / "arm_dev_2026-10-01.json").write_text(json.dumps(_dev_payload()))
    # Newest by name, but solved under constants that are no longer running.
    (tmp_path / "arm_dev_2026-10-04.json").write_text(
        json.dumps(_dev_payload(run={"constants_sha256": "0" * 64}))
    )
    # An ablation cell never qualifies, whatever its date.
    (tmp_path / "arm_dev_s4off_2026-10-09.json").write_text(json.dumps(_dev_payload()))
    assert main._development_file() == tmp_path / "arm_dev_2026-10-01.json"
    assert "predates the running constants" in main._DEV_REJECTIONS[0]
    value, source = main._calibrated_threshold()
    assert value == pytest.approx(0.84)
    assert source is not None and "arm_dev_2026-10-01.json" in source


@pytest.mark.parametrize(
    "override",
    [
        {"run": {"constants_sha256": "0" * 64}},
        {"background": "gsoc"},
        {"budget_alerts_per_camera_night": 3.0},
        {"baseline": {"shipped": False}},
        {"calibration": {"realised_on": "dev", "threshold": 0.84}},
    ],
)
def test_an_unusable_development_run_takes_the_wall_down_with_its_reason(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, reprobe: None, override: dict[str, Any]
) -> None:
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    (tmp_path / "arm_dev_2026-10-04.json").write_text(json.dumps(_dev_payload(**override)))
    assert main._development_file() is None
    assert main._calibrated_threshold() == (None, None)
    monkeypatch.setattr(
        main, "_wall_sequences",
        lambda: ([main.Sequence(sequence_id="s", camera_id="c", arm="dev", frame_paths=["x"], t0_index=None)], None),
    )
    main.REPLAY.probe()
    assert not main.REPLAY.available
    assert "Newest development run: arm_dev_2026-10-04.json" in main.REPLAY.live_pipeline()["detail"]


def test_the_wall_reads_confuse_frames_from_their_local_root_and_drops_exclusions(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write(root: pathlib.Path, sequence_id: str) -> None:
        (root / sequence_id).mkdir(parents=True)
        (root / sequence_id / "f.jpg").write_bytes(b"x")

    figlib, mta = tmp_path / "corpus" / "figlib", tmp_path / "corpus" / "mta"
    write(figlib, "fire-a")
    write(figlib, "night-b")
    write(mta, "confuse-c")
    window = lambda sid: {"sequence_id": sid, "camera_id": f"cam-{sid}", "t0_index": None, "frames": ["f.jpg"]}  # noqa: E731
    (tmp_path / "manifest_dev.json").write_text(
        json.dumps({"arm": "dev", "windows": [window("fire-a"), window("night-b")]})
    )
    (tmp_path / "manifest_confuse.json").write_text(
        json.dumps({"arm": "confuse", "local_root": "corpus/mta", "windows": [window("confuse-c")]})
    )
    (tmp_path / "excluded.json").write_text(
        json.dumps({"excluded": [{"sequence_id": "night-b", "reason": "night frame"}]})
    )
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "CORPUS_DIR", figlib)
    sequences, _ = main._wall_sequences()
    assert [(s.sequence_id, s.arm) for s in sequences] == [("fire-a", "dev"), ("confuse-c", "confuse")]


def test_the_frame_cache_is_pixel_identical_and_is_read_back(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = sorted((main.REPO_ROOT / "data" / "micro").rglob("*.jpg"))[0]
    direct = main._decode_working_frame(str(frame))
    monkeypatch.setattr(main, "FRAME_CACHE_DIR", tmp_path / "cache")
    first = main._read_working_frame(str(frame))
    assert np.array_equal(first, direct) and first.dtype == direct.dtype
    assert len(list((tmp_path / "cache").rglob("*.npy"))) == 1

    def no_decode(_path: str) -> np.ndarray:
        raise AssertionError("a cached frame must not be decoded again")

    monkeypatch.setattr(main, "_decode_working_frame", no_decode)
    assert np.array_equal(main._read_working_frame(str(frame)), direct)


def test_a_zero_length_cache_entry_is_re_decoded_and_healed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """np.load raises EOFError, not OSError/ValueError, on an empty file -- which would fail
    that frame's tick on every lap instead of falling back to decoding."""
    frame = sorted((main.REPO_ROOT / "data" / "micro").rglob("*.jpg"))[0]
    monkeypatch.setattr(main, "FRAME_CACHE_DIR", tmp_path / "cache")
    cached = main._cache_path(str(frame))
    assert cached is not None
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"")
    out = main._read_working_frame(str(frame))
    assert np.array_equal(out, main._decode_working_frame(str(frame)))
    assert cached.stat().st_size > 0, "the empty entry is rewritten, not left to fail again"


def test_a_burst_of_connects_cannot_exceed_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Slots are reserved before the response exists, so two admits cannot share one."""
    from starlette.requests import Request as StarletteRequest

    monkeypatch.setattr(main, "MAX_VIEWERS", 1)
    req = StarletteRequest(
        {"type": "http", "client": ("198.51.100.7", 1), "headers": [], "method": "GET", "path": "/"}
    )

    async def run() -> None:
        first = await main.watchlist(req)
        with pytest.raises(main.HTTPException) as refused:
            await main.watchlist(req)
        assert refused.value.status_code == 503
        first._release()  # the response ended before its body was iterated
        assert main.REPLAY._reserved == 0
        assert "198.51.100.7" not in main._CLIENT_VIEWERS

    asyncio.run(run())


def test_startup_prunes_only_stale_replay_directories(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "EVIDENCE_DIR", tmp_path)
    for name in ("20261001T000000Z", "20261004T075517Z", "keep-me"):
        (tmp_path / name).mkdir()
    (tmp_path / ".gitkeep").write_text("")
    assert main.prune_evidence("20261004T075517Z") == ["20261001T000000Z"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [".gitkeep", "20261004T075517Z", "keep-me"]


def test_the_alert_rate_is_published_over_fire_free_windows_only() -> None:
    replay = main.Replay()
    seq = lambda sid, arm: main.Sequence(sequence_id=sid, camera_id=sid, arm=arm, frame_paths=["x"] * 81, t0_index=None)  # noqa: E731
    replay._cameras = [main._Camera(seq("fire", "dev")), main._Camera(seq("calm", "confuse"))]
    replay._confuse_keys = {"calm"}
    replay._frame_count = 81
    replay._alerts_uncapped = 40  # the fire window alerting after ignition, as designed
    replay._alerts_confuse = 2
    metrics = replay.night_metrics()
    assert metrics["RealisedAlertRate"] == pytest.approx(2.0)
    replay._confuse_keys = set()
    assert "RealisedAlertRate" not in replay.night_metrics()


def test_publishing_telemetry_never_blocks_the_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = main.Replay()
    replay._cameras = [main._Camera(main.Sequence(sequence_id="c", camera_id="c", arm="confuse", frame_paths=["x"], t0_index=None))]
    replay._confuse_keys = {"c"}
    replay._frame_count = 81
    released = __import__("threading").Event()

    class SlowSink:
        def put(self, *a: Any, **k: Any) -> bool:
            released.wait(5)
            return True

    replay._telemetry = SlowSink()  # type: ignore[assignment]
    started = time.monotonic()
    replay._publish_telemetry()
    assert time.monotonic() - started < 0.5, "a slow CloudWatch stalled the tick"
    released.set()


def test_the_measured_tick_period_is_published() -> None:
    replay = main.Replay()
    for t in (0.0, 2.5, 5.0, 7.5):
        replay._observe_tick_start(t)
    assert replay.tick_seconds_measured == pytest.approx(2.5)
    assert replay.status()["tick_seconds_measured"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------------
# Round-3 audit: ticks that raise, the failed state, the replay wrap, atomic images
# ---------------------------------------------------------------------------------


class _CountingTelemetry:
    """Stands in for Telemetry (which has __slots__) and counts heartbeats."""

    def __init__(self, beats: list[int]) -> None:
        self.beats = beats

    def heartbeat(self) -> bool:
        self.beats.append(1)
        return True


def _ready_replay(monkeypatch: pytest.MonkeyPatch) -> main.Replay:
    """A probed, available Replay with one fake viewer and instant backoff."""
    monkeypatch.setattr(main, "TICK_BACKOFF_BASE_SECONDS", 0.0)
    replay = main.Replay()
    replay._probed = True
    replay._unavailable = None
    replay._frame_count = 10
    replay.tick_seconds = 0.0
    replay._viewers.add(asyncio.Queue())
    return replay


def test_one_failing_tick_is_skipped_and_the_wall_keeps_going(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _ready_replay(monkeypatch)
    calls = {"n": 0}

    def tick() -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full, once")
        if calls["n"] >= 4:
            replay._viewers.clear()  # let the loop idle out
            replay.idle_stop_seconds = 0.0
        return {"frame_index": calls["n"]}

    monkeypatch.setattr(replay, "_tick", tick)
    asyncio.run(replay._run())
    assert replay.failed is None and replay.available
    assert replay.status()["consecutive_tick_failures"] == 0
    assert replay._frame_index == 1, "the failing frame is skipped, not retried forever"


def test_repeated_tick_failures_enter_a_failed_state_that_stops_the_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _ready_replay(monkeypatch)

    def tick() -> dict[str, Any]:
        raise ValueError("undecodable frame")

    monkeypatch.setattr(replay, "_tick", tick)
    beats: list[int] = []
    replay._telemetry = _CountingTelemetry(beats)  # type: ignore[assignment]
    asyncio.run(replay._run())
    assert replay.failed is not None and "undecodable frame" in replay.failed
    assert replay.status()["state"] == "failed"
    assert not replay.available
    assert replay.heartbeat() is False and beats == [], "a failed wall must go silent"


def test_a_wall_unavailable_by_design_still_heartbeats(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = main.Replay()
    replay._probed = True
    replay._unavailable = "no calibrated threshold in results/ yet"
    beats: list[int] = []
    replay._telemetry = _CountingTelemetry(beats)  # type: ignore[assignment]
    assert replay.heartbeat() is True and beats == [1]


def test_the_next_viewer_re_probes_a_failed_wall(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = main.Replay()
    replay._probed = True
    replay._unavailable = None
    replay._failed = "5 consecutive ticks failed"
    probes: list[int] = []

    def probe() -> None:
        probes.append(1)
        replay._probed = True
        replay._unavailable = "re-probed"

    monkeypatch.setattr(replay, "probe", probe)

    async def run() -> None:
        queue = await replay.attach()
        replay.detach(queue)
        await replay.stop()

    asyncio.run(run())
    assert probes == [1]
    assert replay.failed is None


def test_a_second_caller_waits_for_the_probe_already_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second judge connecting while the first judge's probe is still scanning must wait
    for that probe's verdict -- not return early and start the loop on a cleared roster."""
    entered, release = threading.Event(), threading.Event()
    replay = main.Replay()

    def slow_reason() -> str | None:
        entered.set()
        release.wait(5)
        return "stub verdict"

    monkeypatch.setattr(replay, "_probe_reason", slow_reason)

    async def run() -> None:
        first = asyncio.create_task(replay.ensure_probed())
        assert await asyncio.to_thread(entered.wait, 5)
        second = asyncio.create_task(replay.ensure_probed())
        await asyncio.sleep(0.05)
        assert not second.done(), "returned before the in-flight probe had a verdict"
        release.set()
        await asyncio.gather(first, second)
        assert replay._unavailable == "stub verdict"

    asyncio.run(run())


def test_health_reports_degraded_for_a_failed_wall(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main.REPLAY, "_failed", "5 consecutive ticks failed; last: OSError")
    monkeypatch.setattr(main.REPLAY, "_probed", True)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["replay"]["state"] == "failed"
    assert "consecutive ticks failed" in body["live_pipeline"]["detail"]


def test_every_camera_starts_fresh_at_the_replay_wrap() -> None:
    """bench.py replays each window once from fresh state; the wall must not carry a warm
    background model across the jump from the window's last frame back to its first."""
    replay = main.Replay()
    cams = [
        main._Camera(main.Sequence(sequence_id=f"s{i}", camera_id=f"c{i}", arm="dev",
                                   frame_paths=[], t0_index=None))
        for i in range(3)
    ]
    for cam in cams:
        cam.prev_grey = np.zeros((4, 4), np.uint8)
        cam.prev_work = np.zeros((4, 4, 3), np.uint8)
    replay._cameras = cams
    before = [id(c.tracker) for c in cams]
    replay._reset_camera_night()
    assert all(c.prev_grey is None and c.prev_work is None for c in cams)
    assert all(id(c.tracker) != b for c, b in zip(cams, before))


def test_the_tick_says_when_its_ranking_is_still_warming() -> None:
    """Every lap resets every segmenter, so ``warm`` must come from the server each tick."""
    from plumerank.constants import WARMUP_FRAMES

    replay = main.Replay()
    cams = [
        main._Camera(main.Sequence(sequence_id=f"s{i}", camera_id=f"c{i}", arm="dev",
                                   frame_paths=[], t0_index=None))
        for i in range(3)
    ]
    replay._cameras = cams
    assert replay._warm_state() == (False, 0)
    for cam in cams:
        cam.segmenter._frames_seen = WARMUP_FRAMES
    cams[0].segmenter._frames_seen = WARMUP_FRAMES - 1
    assert replay._warm_state() == (False, WARMUP_FRAMES - 1), "one cold model is a cold wall"
    cams[0].segmenter._frames_seen = WARMUP_FRAMES
    assert replay._warm_state() == (True, WARMUP_FRAMES)
    replay._reset_camera_night()
    assert replay._warm_state() == (False, 0), "the camera-night wrap re-warms every model"
    assert main.Replay()._warm_state() == (False, 0), "an empty roster is not warm"


def test_images_are_written_atomically(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caddy may be serving the previous file at the same path; no partial file may remain."""
    monkeypatch.setattr(main, "EVIDENCE_DIR", tmp_path)
    entry, frames, picked = _one_row()
    replay = main.Replay()
    replay._cameras = [main._Camera(main.Sequence(sequence_id="s", camera_id=entry.camera_id,
                                                  arm="dev", frame_paths=[], t0_index=None))]
    replay._write_evidence([entry], frames, picked, 3)
    replay._write_thumbnails([entry], frames)
    assert not list(tmp_path.rglob("*.partial.jpg"))
    assert (tmp_path / replay.replay_id / "3" / f"{entry.camera_id}.jpg").is_file()
    assert (tmp_path / replay.replay_id / "frames" / f"{entry.camera_id}.jpg").is_file()


def test_the_wall_cuts_confuse_windows_exactly_as_the_bench_calibrates_on_them() -> None:
    """The wall's fire-free rate reproduces the calibration only if it replays the same
    frames: bench.py centre-trims an 81-frame CONFUSE window to the 79-frame common length."""
    import importlib.util

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bench_for_align", root / "scripts" / "bench.py")
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["bench_for_align"] = bench  # dataclasses resolve their module by name
    spec.loader.exec_module(bench)

    shapes = [("f1", 39, 80), ("f2", 40, 81), ("f3", 40, 80), ("c1", None, 81), ("c2", None, 81)]
    seqs = [_window(sid, t0, n) for sid, t0, n in shapes]
    cameras, _, count = main._align(seqs)
    windows = [
        bench.Window(sid, sid, t0, tuple(pathlib.Path(f"{sid}/{i}.jpg") for i in range(n)))
        for sid, t0, n in shapes
    ]
    (unified,) = bench.unify_windows(windows)
    assert count == len(unified[0])
    for cam, win in zip(cameras, unified):
        wall = [
            int(pathlib.Path(p).stem)
            for p in cam.sequence.frame_paths[cam.frame_offset : cam.frame_offset + count]
        ]
        assert wall == [int(p.stem) for p in win.frame_paths], cam.camera_id

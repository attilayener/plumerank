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
import time
from typing import Any, Callable, Iterator

import numpy as np
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from api import main
from api.main import app
from plumerank.types import Component, Sequence, Stats, Track, WatchEntry

#: The four routes of build-contract.md W5, written out rather than imported from the app.
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
    (shipcheck caught it: 'blocked by CORS policy', zero console errors on a server-rendered
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
    assert body["replay"]["state"] == "idle"
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

    (tmp_path / "bench_2026-10-08.json").write_text(
        json.dumps({"calibration": {"threshold": 0.61}}), encoding="utf-8"
    )
    value, source = main._calibrated_threshold()
    assert value == pytest.approx(0.61)
    assert source is not None and source.endswith(":calibration.threshold")


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
    await _wait_for(lambda: main.REPLAY.status()["state"] == "idle", timeout=5.0)
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
        assert main.REPLAY.status()["state"] == "idle"
        async with _Viewer() as viewer:
            await viewer.next_event()
            attached = main.REPLAY.status()
        gone = await _wait_for(lambda: main.REPLAY.status()["viewers"] == 0)
        stopped = await _wait_for(lambda: main.REPLAY.status()["state"] == "idle")
        await _drain_replay()
        return attached, gone, stopped

    attached, gone, stopped = asyncio.run(run())
    assert attached["state"] == "running"
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
            assert camera.frame_offset == 0
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

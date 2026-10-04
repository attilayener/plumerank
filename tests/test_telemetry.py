"""Tests for the only boto3 in the project.

Nothing here touches AWS. ``_cloudwatch_client`` exists as a named module-level function
precisely so it can be replaced, and every test replaces it -- a telemetry test that
needed credentials would be a test that never runs in CI, which is the same as no test.

The property under test is unusual: the module's job is to *fail silently and keep going*.
So most of these assert that a broken sink returns False, leaves nothing raised, and stops
talking to the client -- because a credential-less laptop replaying 3,240 ticks must not
pay a network timeout on each one.
"""

from __future__ import annotations

from typing import Any

import pytest

from plumerank import telemetry as tel
from plumerank.constants import CLOUDWATCH_NAMESPACE


class StubCloudWatch:
    """Records calls instead of making them. Optionally fails, like a real one does."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raises = raises

    def put_metric_data(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubCloudWatch:
    client = StubCloudWatch()
    monkeypatch.setattr(tel, "_cloudwatch_client", lambda: client)
    return client


# ---------------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------------


def test_put_reaches_cloudwatch_in_the_documented_shape(stub: StubCloudWatch) -> None:
    assert tel.Telemetry().put("RealisedAlertRate", 1.87) is True
    (call,) = stub.calls
    assert call["Namespace"] == CLOUDWATCH_NAMESPACE
    assert call["MetricData"] == [
        {"MetricName": "RealisedAlertRate", "Value": 1.87, "Unit": "None"}
    ]


def test_heartbeat_is_a_counted_one(stub: StubCloudWatch) -> None:
    assert tel.Telemetry().heartbeat() is True
    datum = stub.calls[0]["MetricData"][0]
    assert datum == {"MetricName": "Heartbeat", "Value": 1.0, "Unit": "Count"}


def test_the_namespace_is_overridable_but_defaults_to_the_frozen_one(stub: StubCloudWatch) -> None:
    tel.Telemetry(namespace="PlumeRankStaging").put("EscalationFraction", 0.012)
    assert stub.calls[0]["Namespace"] == "PlumeRankStaging"


def test_all_three_metrics_are_accepted(stub: StubCloudWatch) -> None:
    t = tel.Telemetry()
    assert all(t.put(name, 1.0) for name in tel.METRICS)
    assert len(stub.calls) == 3


def test_the_published_surface_is_exactly_three_metrics() -> None:
    # A fourth metric is a dashboard nobody built and an alarm nobody watches.
    assert tel.METRICS == ("RealisedAlertRate", "EscalationFraction", "Heartbeat")


# ---------------------------------------------------------------------------------
# The laptop with no AWS account -- the path that must never crash
# ---------------------------------------------------------------------------------


def test_no_credentials_degrades_to_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> Any:
        raise RuntimeError("Unable to locate credentials")

    monkeypatch.setattr(tel, "_cloudwatch_client", boom)
    t = tel.Telemetry()
    assert t.live is False
    assert t.put("Heartbeat", 1.0) is False
    assert t.heartbeat() is False


def test_disabled_never_builds_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> Any:
        raise AssertionError("enabled=False must not reach the client factory at all")

    monkeypatch.setattr(tel, "_cloudwatch_client", boom)
    assert tel.Telemetry(enabled=False).put("Heartbeat", 1.0) is False


@pytest.mark.parametrize(
    "exc", [RuntimeError("NoCredentialsError"), OSError("dns"), ValueError("region")]
)
def test_a_failing_put_returns_false_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    client = StubCloudWatch(raises=exc)
    monkeypatch.setattr(tel, "_cloudwatch_client", lambda: client)
    assert tel.Telemetry().put("RealisedAlertRate", 2.0) is False


def test_the_first_failure_disables_the_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    # 3,240 ticks x one network timeout is a stalled replay wall, not a missing dashboard.
    # The silence alarm on Heartbeat is what catches the disabled sink.
    client = StubCloudWatch(raises=RuntimeError("NoCredentialsError"))
    monkeypatch.setattr(tel, "_cloudwatch_client", lambda: client)
    t = tel.Telemetry()
    assert t.put("Heartbeat", 1.0) is False
    assert t.live is False
    assert t.put("Heartbeat", 1.0) is False
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------------
# Caller mistakes are flagged, not published
# ---------------------------------------------------------------------------------


def test_an_unknown_metric_is_refused_without_a_call(stub: StubCloudWatch) -> None:
    assert tel.Telemetry().put("RealizedAlertRate", 2.0) is False
    assert stub.calls == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_is_refused_without_a_call(stub: StubCloudWatch, value: float) -> None:
    # inf reaches here from metrics.time_to_detection, where it means "missed". Publishing
    # it would put a datapoint on the dashboard that means the opposite of what it says.
    assert tel.Telemetry().put("RealisedAlertRate", value) is False
    assert stub.calls == []


def test_a_refused_metric_does_not_disable_the_sink(stub: StubCloudWatch) -> None:
    t = tel.Telemetry()
    t.put("NotAMetric", 1.0)
    assert t.live is True
    assert t.put("Heartbeat", 1.0) is True


# ---------------------------------------------------------------------------------


def test_boto3_is_not_imported_at_module_import_time() -> None:
    # The import lives inside _cloudwatch_client so that importing plumerank on a box
    # without boto3 -- or without credentials -- costs nothing and cannot fail.
    assert not hasattr(tel, "boto3")


# ---------------------------------------------------------------------------------
# Region and recovery -- audit 2026-10-04 #9: zero metrics ever landed on the live box
# ---------------------------------------------------------------------------------


def test_the_region_comes_from_the_environment_then_the_deploy_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    assert tel.region() == tel.DEFAULT_REGION == "us-west-2"
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    assert tel.region() == "eu-west-1"
    monkeypatch.setenv("AWS_REGION", "ap-southeast-3")
    assert tel.region() == "ap-southeast-3"


def test_the_client_is_built_with_an_explicit_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """boto3.client("cloudwatch") with no region raises NoRegionError on a bare box."""
    boto3 = pytest.importorskip("boto3")
    seen: dict[str, Any] = {}

    def fake_client(service: str, **kwargs: Any) -> StubCloudWatch:
        seen["service"] = service
        seen.update(kwargs)
        return StubCloudWatch()

    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr(boto3, "client", fake_client)
    tel._cloudwatch_client()
    assert seen["service"] == "cloudwatch" and seen["region_name"] == "us-west-2"


def test_the_client_makes_one_short_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow CloudWatch must cost seconds, never boto3's default retry ladder."""
    boto3 = pytest.importorskip("boto3")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(boto3, "client", lambda service, **kw: seen.update(kw) or StubCloudWatch())
    tel._cloudwatch_client()
    config = seen["config"]
    assert config.connect_timeout <= 2.0 and config.read_timeout <= 3.0
    assert config.retries["max_attempts"] == 1


def test_every_datapoint_carries_the_instance_dimension_when_set(
    stub: StubCloudWatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alarms watch this box's dimension, so no other publisher can keep them green."""
    monkeypatch.setenv(tel.INSTANCE_ENV, "i-0123456789abcdef0")
    assert tel.Telemetry().heartbeat() is True
    datum = stub.calls[-1]["MetricData"][0]
    assert datum["Dimensions"] == [{"Name": "InstanceId", "Value": "i-0123456789abcdef0"}]


def test_no_dimension_without_an_instance_id(
    stub: StubCloudWatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(tel.INSTANCE_ENV, raising=False)
    assert tel.Telemetry().heartbeat() is True
    assert "Dimensions" not in stub.calls[-1]["MetricData"][0]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_failure_pauses_the_sink_and_it_retries_after_the_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = StubCloudWatch(raises=RuntimeError("Throttling"))
    monkeypatch.setattr(tel, "_cloudwatch_client", lambda: client)
    clock = Clock()
    t = tel.Telemetry(clock=clock)
    assert t.put("Heartbeat", 1.0) is False
    assert t.live is False
    clock.now += tel.RETRY_AFTER_SECONDS - 1
    assert t.put("Heartbeat", 1.0) is False
    assert len(client.calls) == 1, "no retry inside the pause"
    client._raises = None
    clock.now += 2
    assert t.live is True
    assert t.put("Heartbeat", 1.0) is True
    assert len(client.calls) == 2


def test_a_client_that_could_not_be_built_is_rebuilt_after_the_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    good = StubCloudWatch()

    def factory() -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("NoRegionError")
        return good

    monkeypatch.setattr(tel, "_cloudwatch_client", factory)
    clock = Clock()
    t = tel.Telemetry(clock=clock)
    assert t.put("Heartbeat", 1.0) is False
    clock.now += tel.RETRY_AFTER_SECONDS
    assert t.put("Heartbeat", 1.0) is True
    assert len(attempts) == 2 and len(good.calls) == 1


def test_a_failure_is_logged_at_warning_once_per_pause(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = StubCloudWatch(raises=RuntimeError("NoCredentialsError"))
    monkeypatch.setattr(tel, "_cloudwatch_client", lambda: client)
    t = tel.Telemetry(clock=Clock())
    with caplog.at_level("WARNING", logger=tel.LOG.name):
        for _ in range(50):
            t.put("Heartbeat", 1.0)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "us-west-2" in warnings[0].getMessage() or "region" in warnings[0].getMessage()


# --- opt-in ---------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("", False), ("0", False), ("true", False)])
def test_publishing_is_opt_in_by_environment(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    """Only the box sets PLUMERANK_TELEMETRY=1; a laptop must never feed the silence alarm."""
    monkeypatch.setenv(tel.TELEMETRY_ENV, value)
    assert tel.enabled_by_environment() is expected


def test_the_api_builds_its_sink_from_the_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from api import main

    monkeypatch.delenv(tel.TELEMETRY_ENV, raising=False)
    assert main.Replay()._telemetry.live is False

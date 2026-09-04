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

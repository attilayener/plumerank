"""Coverage-driven tests for plumerank.telemetry, targeting uncovered paths.

This module targets line 176: the case where a pause expires, reconnection is
attempted, but the client factory still fails on the retry.
"""

from __future__ import annotations

from typing import Any

import pytest

from plumerank import telemetry as tel


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


class Clock:
    """Controllable clock for testing pause/retry logic."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_reconnection_after_pause_fails_returns_false_line_176(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pause expires and reconnection attempt also fails, put() returns False.

    This test targets line 176: the `return False` inside the pause-recovery block
    when _connect() fails after a pause has expired.

    Scenario:
    1. Initial put() fails to build a client (factory raises)
    2. Sink is paused with _paused_until set
    3. Time passes and pause expires
    4. put() is called again, live is now True
    5. We enter the pause-recovery block (self._paused_until is not None)
    6. Clear _paused_until and attempt _connect()
    7. _connect() fails again (factory still raises)
    8. Return False at line 176 (not reached by other tests)
    """
    attempts: list[int] = []

    def failing_factory() -> Any:
        attempts.append(1)
        raise RuntimeError("PersistentConnectionError")

    monkeypatch.setattr(tel, "_cloudwatch_client", failing_factory)
    clock = Clock()
    t = tel.Telemetry(clock=clock)

    # First put fails to build a client and pauses the sink
    result1 = t.put("Heartbeat", 1.0)
    assert result1 is False, "Initial put should fail"
    assert t.live is False, "Sink should be paused after initial failure"
    assert len(attempts) == 1, "Factory should be called once"
    assert t._client is None, "Client should not exist after failed _connect()"

    # Advance time past the pause window
    clock.now += tel.RETRY_AFTER_SECONDS

    # Verify pause has expired
    assert t.live is True, "Sink should be live after pause expires"

    # Second put attempts to reconnect but factory still fails
    # This execution path hits line 176: return False in the pause-recovery block
    result2 = t.put("Heartbeat", 1.0)
    assert result2 is False, "Reconnection attempt should fail and return False"
    assert len(attempts) == 2, "Factory should be called a second time for reconnection"
    assert t.live is False, "Sink should be paused again after reconnection failure"
    assert t._client is None, "Client should still not exist"

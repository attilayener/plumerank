"""Three CloudWatch metrics, and the only boto3 in the project.

What this exists for
--------------------
The alert budget is the operating parameter the whole system is calibrated to, so the
responsible thing is to put the *realised* rate on a dashboard with a drift alarm around
it, rather than to publish a calibration and hope it holds. Three metrics and no fourth:

* ``RealisedAlertRate``    -- alerts per camera-night, against the 2.0 target.
* ``EscalationFraction``   -- the measured ~1%, so the cost claim cannot silently rot.
* ``Heartbeat``            -- the alarm fires on *silence*, not on events. An alarm that
  only fires when something happens cannot tell "nothing happened" from "the box died".

Why it never raises
-------------------
The identical pipeline runs on a laptop with no AWS account, in CI with no credentials,
and on an EC2 box with an instance profile. Telemetry is an observation of the run, never
a part of it, so a missing credential must cost the run nothing: every method returns a
bool and swallows everything below it. ``False`` is the documented flagged value -- the
caller can log it, and the silence alarm catches the case where it stays False forever,
which is the only place this failure actually matters.

A failure pauses the sink for ``RETRY_AFTER_SECONDS`` instead of retrying on every call. A
replay is thousands of ticks and a credential-less box would otherwise pay a network
timeout on each one, turning an absent dashboard into a stalled wall. The pause is not
permanent: a transient AWS error on day 2 of a 14-day judging window used to disable the
sink for the life of the process, which turned one throttled request into a silence alarm
nobody could clear without a restart. Each pause logs one WARNING, so a broken sink is
visible in the journal at the rate it is retried and never once per tick.

The region is explicit. ``boto3.client("cloudwatch")`` with no region raises
``NoRegionError`` on a box whose environment does not name one -- which is every box
``deploy/bootstrap.sh`` built before 2026-10-04 -- and that error was logged at INFO, so
the metrics and both alarms were absent with nothing in the journal saying why.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Callable

from .constants import CLOUDWATCH_NAMESPACE

__all__ = ["Telemetry", "METRICS", "enabled_by_environment"]

LOG = logging.getLogger(__name__)

#: The whole published surface. A name outside it is a typo, and a typo silently creates
#: a fourth CloudWatch metric that no dashboard shows and no alarm watches.
METRICS: tuple[str, ...] = ("RealisedAlertRate", "EscalationFraction", "Heartbeat")

#: The deploy region (deploy/launch.md section 1). Used only when the environment names none:
#: the metrics and the alarms that watch them must live in the same region, and the alarms
#: are created there.
DEFAULT_REGION: str = "us-west-2"

#: How long a failed sink stays paused before the next put tries again. Long against a tick
#: (1 s) so a credential-less laptop pays one timeout per five minutes, short against the
#: silence alarm's evaluation window so one transient error never trips it on its own.
RETRY_AFTER_SECONDS: float = 300.0


#: The opt-in switch. deploy/bootstrap.sh writes ``PLUMERANK_TELEMETRY=1`` into the box's
#: env file; nothing else sets it. Off by default because the heartbeat alarm watches for
#: SILENCE: a laptop test run or dev server with AWS credentials in its shell used to
#: publish real ``PlumeRank/Heartbeat`` datapoints into the same metric, so a dead box
#: could read as alive while anyone worked on the code.
TELEMETRY_ENV: str = "PLUMERANK_TELEMETRY"


def enabled_by_environment() -> bool:
    """True only when ``PLUMERANK_TELEMETRY`` is exactly ``1``."""
    return os.environ.get(TELEMETRY_ENV, "").strip() == "1"


def region() -> str:
    """``AWS_REGION``, else ``AWS_DEFAULT_REGION``, else the deploy region."""
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return DEFAULT_REGION


def _cloudwatch_client() -> Any:
    """Built here, in one named function, so tests can replace it without touching AWS."""
    import boto3

    return boto3.client("cloudwatch", region_name=region())


class Telemetry:
    """A CloudWatch sink that degrades to a no-op instead of failing a run."""

    __slots__ = ("namespace", "_client", "_enabled", "_paused_until", "_clock")

    def __init__(
        self,
        namespace: str = CLOUDWATCH_NAMESPACE,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.namespace = namespace
        self._client: Any = None
        self._enabled = bool(enabled)
        self._paused_until: float | None = None
        self._clock = clock
        if self._enabled:
            self._connect()

    def _connect(self) -> bool:
        try:
            self._client = _cloudwatch_client()
        except Exception as exc:  # no boto3, no region, no credentials -- all the same here
            self._pause(f"client unavailable: {type(exc).__name__}: {exc}")
            return False
        return True

    def _pause(self, why: str) -> None:
        self._paused_until = self._clock() + RETRY_AFTER_SECONDS
        LOG.warning(
            "telemetry paused for %.0f s (region %s, namespace %s): %s",
            RETRY_AFTER_SECONDS,
            region(),
            self.namespace,
            why,
        )

    @property
    def live(self) -> bool:
        """False while the sink is disabled or paused after a failure, which is what a caller
        reports rather than inferring from a run of False returns."""
        if not self._enabled:
            return False
        return self._paused_until is None or self._clock() >= self._paused_until

    def put(self, metric: str, value: float, unit: str = "None") -> bool:
        """Publish one datapoint. True if it reached CloudWatch, False otherwise."""
        if not self.live:
            return False
        if self._paused_until is not None:
            # The pause has expired: one retry, rebuilding the client if it never existed.
            self._paused_until = None
            if self._client is None and not self._connect():
                return False
        if metric not in METRICS:
            LOG.warning("refusing unknown metric %r; the published surface is %s", metric, METRICS)
            return False
        if not math.isfinite(value):
            LOG.warning("refusing non-finite value %r for %s", value, metric)
            return False
        try:
            self._client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[{"MetricName": metric, "Value": float(value), "Unit": unit}],
            )
        except Exception as exc:
            self._pause(f"{type(exc).__name__} on {metric}: {exc}")
            return False
        return True

    def heartbeat(self) -> bool:
        """One tick of proof-of-life. The silence alarm watches this and nothing else."""
        return self.put("Heartbeat", 1.0, unit="Count")

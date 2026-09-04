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

The first failure disables the sink for the process. A replay is thousands of ticks and a
credential-less box would otherwise pay a network timeout on each one, turning an absent
dashboard into a stalled wall. A transient AWS error therefore also disables it -- and is
caught by the same silence alarm, which is exactly the alarm that exists for this.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .constants import CLOUDWATCH_NAMESPACE

__all__ = ["Telemetry", "METRICS"]

LOG = logging.getLogger(__name__)

#: The whole published surface. A name outside it is a typo, and a typo silently creates
#: a fourth CloudWatch metric that no dashboard shows and no alarm watches.
METRICS: tuple[str, ...] = ("RealisedAlertRate", "EscalationFraction", "Heartbeat")


def _cloudwatch_client() -> Any:
    """Built here, in one named function, so tests can replace it without touching AWS."""
    import boto3

    return boto3.client("cloudwatch")


class Telemetry:
    """A CloudWatch sink that degrades to a no-op instead of failing a run."""

    __slots__ = ("namespace", "_client", "_live")

    def __init__(self, namespace: str = CLOUDWATCH_NAMESPACE, enabled: bool = True) -> None:
        self.namespace = namespace
        self._client: Any = None
        self._live = bool(enabled)
        if self._live:
            try:
                self._client = _cloudwatch_client()
            except Exception as exc:  # no boto3, no region, no credentials -- all the same here
                LOG.info("telemetry disabled: %s", exc)
                self._live = False

    @property
    def live(self) -> bool:
        """False once the sink has given up, which is what a caller reports rather than
        inferring from a run of False returns."""
        return self._live

    def put(self, metric: str, value: float, unit: str = "None") -> bool:
        """Publish one datapoint. True if it reached CloudWatch, False otherwise."""
        if not self._live:
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
            LOG.warning("telemetry disabled after %s on %s: %s", type(exc).__name__, metric, exc)
            self._live = False
            return False
        return True

    def heartbeat(self) -> bool:
        """One tick of proof-of-life. The silence alarm watches this and nothing else."""
        return self.put("Heartbeat", 1.0, unit="Count")

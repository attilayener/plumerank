"""The metric's own tests. Written on D1, against a random-score stub, because a harness
that has never produced a believably terrible number has never been shown to work."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plumerank.constants import FRAMES_PER_WINDOW, MINUTES_PER_FRAME
from plumerank.metrics import (
    Latency,
    alerts_per_camera_night,
    budget_ceiling,
    detection_index,
    percentiles,
    summarise,
    time_to_detection,
)


# --- detection -------------------------------------------------------------------


def test_detection_requires_both_top_k_and_alert():
    top = [False] * 10
    alert = [False] * 10
    top[5] = True          # ranked, but silent
    alert[7] = True        # alerted, but not in the top-k
    assert detection_index(top, alert, t0_index=0) is None

    top[7] = True
    assert detection_index(top, alert, t0_index=0) == 7


def test_detection_ignores_frames_before_ignition():
    """A pre-ignition alert must never buy a negative latency."""
    top = [True] * 10
    alert = [True] * 10
    assert detection_index(top, alert, t0_index=6) == 6
    assert time_to_detection(top, alert, t0_index=6) == 0.0


def test_time_to_detection_is_minutes_after_ignition():
    n = FRAMES_PER_WINDOW
    top = [False] * n
    alert = [False] * n
    t0 = 40
    top[t0 + 7] = alert[t0 + 7] = True
    assert time_to_detection(top, alert, t0) == pytest.approx(7 * MINUTES_PER_FRAME)


def test_a_miss_is_inf_not_none():
    top = [False] * 20
    assert time_to_detection(top, top, t0_index=0) == math.inf


def test_parallel_length_is_enforced():
    with pytest.raises(ValueError):
        detection_index([True, True], [True], t0_index=0)


# --- aggregation -----------------------------------------------------------------


def test_percentiles_survive_infinities():
    """The failure this guards: np.percentile turns a set with inf into nan, which
    would render 'half the sequences were missed' as 'not a number'."""
    vals = [1.0, 2.0, 3.0, math.inf]
    p50, p95 = percentiles(vals)
    assert p50 == 2.0
    assert math.isinf(p95)
    assert not math.isnan(p50)


def test_p50_is_inf_when_most_sequences_are_missed():
    vals = [1.0, 2.0] + [math.inf] * 3
    s = summarise(vals)
    assert math.isinf(s.p50)
    assert s.n_missed == 3
    assert s.detection_rate == pytest.approx(2 / 5)


def test_summary_json_renders_inf_as_null_not_nan():
    s = summarise([1.0, math.inf])
    js = s.to_json()
    assert js["p95_minutes"] is None
    assert js["n_missed"] == 1
    # JSON has no NaN; a nan here would produce output no parser accepts.
    assert all(not (isinstance(v, float) and math.isnan(v)) for v in js.values())


def test_summarise_rejects_an_empty_arm():
    with pytest.raises(ValueError):
        summarise([])


def test_percentile_bounds_are_checked():
    with pytest.raises(ValueError):
        percentiles([1.0], (150.0,))


# --- the D1 acceptance test: a random ranker must look terrible ------------------


def test_random_scores_produce_a_believably_terrible_number():
    """The harness is only trustworthy if it can fail. A uniformly random ranker over
    40 cameras enters the top-4 about 10% of minutes, so it stumbles onto the true
    camera eventually -- but late, and it misses outright often enough that the
    detection rate is nowhere near 1.0. Anything better than this from noise would mean
    the metric is crediting something it should not."""
    rng = np.random.default_rng(20260830)
    n_seq, n_after = 60, FRAMES_PER_WINDOW - 41
    ttds = []
    for _ in range(n_seq):
        in_top_k = rng.random(n_after) < (4 / 40)
        alerted = rng.random(n_after) < 0.25
        ttds.append(
            time_to_detection(
                [False] * 41 + list(in_top_k),
                [False] * 41 + list(alerted),
                t0_index=40,
            )
        )
    s = summarise(ttds)
    assert s.n == n_seq
    assert s.p50 > 10.0, "random ranking should be late, not competitive"
    assert s.detection_rate < 1.0, "random ranking should miss some sequences outright"


# --- the budget unit -------------------------------------------------------------


def test_alert_rate_is_per_camera_night():
    # 80 alerts over 40 camera-nights is exactly the 2.0 target.
    assert alerts_per_camera_night(80, 40) == pytest.approx(2.0)


def test_short_windows_report_in_the_same_unit():
    """The micro corpus has shorter windows; its rate must still mean alerts per
    81-frame camera-night, or `make verify` would print a number in private units."""
    full = alerts_per_camera_night(30, 10, frames_per_window=FRAMES_PER_WINDOW)
    third = alerts_per_camera_night(10, 10, frames_per_window=FRAMES_PER_WINDOW // 3)
    assert full == pytest.approx(third)


def test_alert_rate_rejects_a_zero_denominator():
    with pytest.raises(ValueError):
        alerts_per_camera_night(1, 0)


def test_budget_ceiling_is_two_per_camera():
    assert budget_ceiling(40) == 80


# --- latency ---------------------------------------------------------------------


def test_latency_uses_opencv_tickmeter_and_reports_percentiles():
    lat = Latency()
    for _ in range(5):
        with lat.stage("classical"):
            np.linalg.svd(np.ones((24, 24)))
    rep = lat.report()
    assert rep["classical"]["n"] == 5
    assert rep["classical"]["p50_ms"] >= 0.0
    assert rep["classical"]["p95_ms"] >= rep["classical"]["p50_ms"]

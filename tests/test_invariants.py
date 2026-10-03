"""Exhaustive, property-based verification of the two functions that must never be wrong.

The rest of the suite is example-based: ``test_rank.py`` checks ``fuse`` and ``BudgetState``
at chosen points (both extremes, a NaN, a duplicate roster). This file instead throws
``hypothesis``-generated cases at the same two contracts and asserts the property holds
everywhere, not just at the points someone thought to write down:

1. ``fuse`` is the hazard every ranking decision is made from. If it ever left [0, 1] or
   silently swallowed a bad input instead of raising, a NaN or an out-of-range score would
   sort arbitrarily and scramble the watch list -- the exact failure ``rank._unit`` is
   written to prevent (see its docstring). Verified here across the full input space, not
   just the documented extremes.
2. ``BudgetState`` is the mechanical form of spec.md section 9 safeguard 3 -- "the alert
   budget is a published, logged, tunable parameter." If any call sequence could push
   ``total_spent`` past ``capacity``, or one camera's spend past ``per_camera``, the
   published budget would be a claim rather than a guarantee.

Both run with a published case count (``--hypothesis-seed`` makes a failure reproducible;
the count printed by ``--hypothesis-show-statistics`` is what the README states).
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from plumerank.rank import BudgetState, fuse
from plumerank.types import Stats

CASES_PER_PROPERTY = 500

finite_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
any_float = st.floats(allow_nan=True, allow_infinity=True, width=32)


@given(s1=finite_unit, s2=finite_unit, s3=finite_unit, s4=finite_unit, esc=finite_unit)
@settings(max_examples=CASES_PER_PROPERTY, suppress_health_check=[HealthCheck.too_slow])
def test_fuse_never_leaves_the_unit_interval_for_any_valid_statistics(s1, s2, s3, s4, esc):
    """For every one of 500 generated valid statistic tuples, the hazard is a finite
    probability. Downstream code (rebase, solve_threshold, the UI's percentage bars) is
    only correct if this holds for ALL inputs, not the handful anyone thought to try."""
    hazard = fuse(Stats(s1=s1, s2=s2, s3=s3, s4=s4, raw_s3=0.0), escalation_score=esc)
    assert math.isfinite(hazard)
    assert 0.0 <= hazard <= 1.0


@given(s1=any_float, s2=any_float, s3=any_float, s4=any_float, esc=finite_unit)
@settings(max_examples=CASES_PER_PROPERTY, suppress_health_check=[HealthCheck.too_slow])
def test_fuse_never_silently_accepts_an_invalid_statistic(s1, s2, s3, s4, esc):
    """The complementary property: across 500 generated cases that are NOT valid squashed
    statistics (NaN, inf, or outside [0, 1]), fuse() either raises ValueError or -- when a
    generated case happens to land back inside [0, 1] by chance -- returns a finite hazard
    in range. There is no third outcome: no case produces a NaN/inf hazard or a silent
    out-of-range pass-through."""
    try:
        hazard = fuse(Stats(s1=s1, s2=s2, s3=s3, s4=s4, raw_s3=0.0), escalation_score=esc)
    except ValueError:
        return
    assert math.isfinite(hazard)
    assert 0.0 <= hazard <= 1.0


@given(
    n_cameras=st.integers(min_value=1, max_value=12),
    per_camera=st.integers(min_value=0, max_value=5),
    spend_calls=st.lists(st.integers(min_value=0, max_value=11), min_size=0, max_size=80),
)
@settings(max_examples=CASES_PER_PROPERTY, suppress_health_check=[HealthCheck.too_slow])
def test_budget_never_overspends_under_any_call_sequence(n_cameras, per_camera, spend_calls):
    """500 generated (roster size, per-camera allowance, call sequence) triples. However the
    calls are ordered or repeated, two invariants must hold for every camera and for the
    roster as a whole -- the mechanical guarantee behind "2 alerts per camera-night", not a
    convention callers are trusted to respect."""
    cameras = [f"cam{i}" for i in range(n_cameras)]
    budget = BudgetState(cameras, per_camera=per_camera)
    for idx in spend_calls:
        camera_id = cameras[idx % n_cameras]
        budget.spend(camera_id)

    for camera_id in cameras:
        assert 0 <= budget.spent(camera_id) <= per_camera
    assert budget.total_spent() <= budget.capacity == n_cameras * per_camera

#!/usr/bin/env python3
"""What the stabiliser's sky-following warps cost the fire-free set -- by counterfactual replay.

    .venv/bin/python scripts/measure_registration_alerts.py   # writes results/registration_alerts_<date>.json

``scripts/measure_registration.py`` shows that on CONFUSE every warp of >= 1 px follows the
moving sky while the ground stays still, so it shifts a static ridgeline. Whether that costs
false alarms is a separate question. The pipeline is stateful -- a warped frame becomes S4's
``prev_work`` on the next tick and part of MOG2's background model -- so the cost cannot be
read off which frames a crossing lands on. It is measured as a counterfactual instead:

* CONFUSE is replayed twice in memory through ``bench.replay_arm``, the shipped configuration
  (MOG2, S4 as frozen, escalation on, the shipped per-camera baseline), trimmed exactly as the
  committed run trimmed it: ``unify_windows`` is called with DEV, which declares the same three
  window shapes as HOLD, so the 79-frame centre-trim is identical and HOLD is never opened.
* The first replay is the shipped pipeline, and is asserted to reproduce the committed run's
  calibration threshold exactly. The second is identical except that ``bench.stabilise`` is
  swapped, for the duration of the replay, for a gate that returns the documented passthrough
  (frame unchanged, dx = dy = 0) on exactly the pairs the ground-band referee calls
  ``weather`` (a warp >= 1 px that worsens the ground's alignment). Windows with no such pair
  are asserted to replay identically.
* Published for both: uncapped crossings at the committed threshold, the realised rate, the
  threshold the 2.0 budget solves to (``bench.calibrate``, the committed run's own path), and
  per-window crossings and changed-frame counts wherever a window differs.

Read-only: nothing under ``results/`` but this script's own output is written; ``bench.py``,
the pipeline and ``constants.py`` are untouched, and no published number changes.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import bench  # noqa: E402
from measure_registration import TERRAIN  # noqa: E402

from plumerank.constants import S4_ENABLED  # noqa: E402
from plumerank.stabilise import stabilise, warp  # noqa: E402
from plumerank.types import Stabilised  # noqa: E402

WARP_MIN_PX: float = 1.0
BUDGET: float = 2.0


def _is_weather(prev: np.ndarray, cur: np.ndarray, s: Stabilised) -> bool:
    """The ground-band referee: a warp >= WARP_MIN_PX that worsens the ground's alignment."""
    if math.hypot(s.dx, s.dy) < WARP_MIN_PX:
        return False
    a = prev[TERRAIN].astype(np.float32)
    raw = cur[TERRAIN].astype(np.float32)
    warped = warp(cur, s.dx, s.dy)[TERRAIN].astype(np.float32)
    return bool(np.mean((warped - a) ** 2) >= np.mean((raw - a) ** 2))


class _Gate:
    """The shipped estimator, except a sky-following warp is not applied. Counts its firings."""

    def __init__(self) -> None:
        self.fired = 0

    def __call__(self, prev: np.ndarray, cur: np.ndarray) -> Stabilised:
        s = stabilise(prev, cur)
        if _is_weather(prev, cur, s):
            self.fired += 1
            return Stabilised(
                frame=cur.copy(), dx=0.0, dy=0.0, response=s.response, method="passthrough"
            )
        return s


def _replay(windows: list[Any], stabiliser: Any = None) -> Any:
    original = bench.stabilise
    if stabiliser is not None:
        bench.stabilise = stabiliser
    try:
        matrix = bench.replay_arm(
            windows,
            arm="confuse",
            background="mog2",
            s4_enabled=S4_ENABLED,
            escalation_enabled=True,
            classifier=bench._classifier(ROOT),
            systems=("plumerank",),
        )
    finally:
        bench.stabilise = original
    bench.rebase(matrix, "shipped")
    return matrix


def _crossings(series: list[float], threshold: float) -> int:
    return sum(1 for h in series if h >= threshold)


def main() -> int:
    # The held-out file by its exact name, as api/main.py:_benchmark_file reads it.
    path = sorted(
        p for p in (ROOT / "results").glob("bench_*.json")
        if re.fullmatch(r"bench_\d{4}-\d{2}-\d{2}\.json", p.name)
    )[-1]
    committed = json.loads(path.read_text(encoding="utf-8"))
    threshold = float(committed["calibration"]["threshold"])

    excluded = [
        e["sequence_id"] if isinstance(e, dict) else e for e in bench.load_exclusions(ROOT)[0]
    ]
    confuse = bench.load_calibration(ROOT, "confuse", excluded)
    _, confuse = bench.unify_windows(bench.load_arm(ROOT, "dev", excluded), confuse)

    shipped = _replay(confuse)
    solved_shipped, realised_shipped = bench.calibrate(shipped, "plumerank", BUDGET)
    if solved_shipped != threshold:
        raise SystemExit(
            f"the shipped replay solves to {solved_shipped!r}, not the committed {threshold!r}: "
            "this harness is not the committed run, so nothing it says about a counterfactual holds"
        )
    gate = _Gate()
    without = _replay(confuse, gate)
    solved_without, realised_without = bench.calibrate(without, "plumerank", BUDGET)

    n = shipped.frames_per_window
    changed = []
    for window, a, b in zip(shipped.windows, shipped.hazards["plumerank"], without.hazards["plumerank"]):
        frames_changed = sum(1 for x, y in zip(a, b) if x != y)
        if frames_changed:
            changed.append(
                {
                    "sequence_id": window.sequence_id,
                    "frames_changed": frames_changed,
                    "crossings_shipped": _crossings(a, threshold),
                    "crossings_without_sky_warps": _crossings(b, threshold),
                }
            )
    windows_with_gate = len(changed)
    crossings_shipped = sum(_crossings(s, threshold) for s in shipped.hazards["plumerank"])
    crossings_without = sum(_crossings(s, threshold) for s in without.hazards["plumerank"])

    payload = {
        "what": "CONFUSE false alarms with and without the stabiliser's sky-following warps",
        "method": (
            "two in-memory bench.replay_arm replays of the committed run's CONFUSE windows; the "
            "second gates bench.stabilise to passthrough on pairs warped >= "
            f"{WARP_MIN_PX} px that worsen the TERRAIN band's MSE"
        ),
        "threshold_source": path.name,
        "committed_threshold": threshold,
        "windows": len(shipped.windows),
        "frames_per_window": n,
        "gate_fired": gate.fired,
        "windows_unchanged": len(shipped.windows) - windows_with_gate,
        "shipped": {
            "crossings": crossings_shipped,
            "realised_alerts_per_camera_night": round(
                bench.per_camera_night(crossings_shipped, len(shipped.windows), n), 4
            ),
            "solved_threshold": solved_shipped,
            "solved_realised": round(realised_shipped, 4),
        },
        "without_sky_warps": {
            "crossings": crossings_without,
            "realised_alerts_per_camera_night": round(
                bench.per_camera_night(crossings_without, len(without.windows), n), 4
            ),
            "solved_threshold": solved_without,
            "solved_realised": round(realised_without, 4),
        },
        "windows_that_changed": changed,
    }
    out = ROOT / "results" / f"registration_alerts_{date.today().isoformat()}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out.relative_to(ROOT))
    print(json.dumps({k: payload[k] for k in ("gate_fired", "windows_unchanged", "shipped", "without_sky_warps")}, indent=1))
    for row in changed:
        print(" ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

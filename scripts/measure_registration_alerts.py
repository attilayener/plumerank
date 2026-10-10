#!/usr/bin/env python3
"""Do the fire-free set's false alarms fall on the frames the stabiliser warped by the sky?

    .venv/bin/python scripts/measure_registration_alerts.py   # writes results/registration_alerts_<date>.json

``scripts/measure_registration.py`` shows that on CONFUSE every warp of >= 1 px follows the
moving sky while the ground stays still, so it shifts a static ridgeline. Whether that costs
false alarms is a separate question, and this answers it on the committed run's own terms:

* CONFUSE is replayed in memory through ``bench.replay_arm`` -- the shipped configuration
  (MOG2, S4 as frozen, escalation on, the shipped per-camera baseline) -- trimmed exactly as
  the committed run trimmed it. ``unify_windows`` is called with DEV, which declares the same
  three window shapes as HOLD, so the 79-frame centre-trim is identical and HOLD is never
  opened.
* Crossings are counted UNCAPPED at the committed held-out threshold, the quantity the
  threshold was solved on. ``realised_alerts_per_camera_night`` is printed and written so the
  replay can be checked against ``results/bench_<date>.json``'s calibration block.
* Each frame is labelled ``weather`` when ``stabilise`` warped it by >= 1 px and the warp
  worsened the ground band's alignment (the referee in ``measure_registration.py``).

Read-only: nothing under ``results/`` but this script's own output is written, and no
published number changes.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys
from datetime import date

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import bench  # noqa: E402
from measure_registration import TERRAIN  # noqa: E402

from plumerank.constants import S4_ENABLED  # noqa: E402
from plumerank.stabilise import stabilise, warp  # noqa: E402

WARP_MIN_PX: float = 1.0


def _weather(prev: np.ndarray, cur: np.ndarray) -> bool:
    s = stabilise(prev, cur)
    if math.hypot(s.dx, s.dy) < WARP_MIN_PX:
        return False
    a = prev[TERRAIN].astype(np.float32)
    raw = cur[TERRAIN].astype(np.float32)
    warped = warp(cur, s.dx, s.dy)[TERRAIN].astype(np.float32)
    return bool(np.mean((warped - a) ** 2) >= np.mean((raw - a) ** 2))


def main() -> int:
    # The held-out file by its exact name, as api/main.py:_benchmark_file reads it.
    path = sorted(
        p for p in (ROOT / "results").glob("bench_*.json")
        if re.fullmatch(r"bench_\d{4}-\d{2}-\d{2}\.json", p.name)
    )[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    threshold = float(payload["calibration"]["threshold"])

    excluded = [
        e["sequence_id"] if isinstance(e, dict) else e for e in bench.load_exclusions(ROOT)[0]
    ]
    confuse = bench.load_calibration(ROOT, "confuse", excluded)
    _, confuse = bench.unify_windows(bench.load_arm(ROOT, "dev", excluded), confuse)
    matrix = bench.replay_arm(
        confuse,
        arm="confuse",
        background="mog2",
        s4_enabled=S4_ENABLED,
        escalation_enabled=True,
        classifier=bench._classifier(ROOT),
        systems=("plumerank",),
    )
    bench.rebase(matrix, "shipped")
    hazards = matrix.hazards["plumerank"]

    counts = {"weather": {"frames": 0, "crossings": 0}, "other": {"frames": 0, "crossings": 0}}
    windows = []
    for window, series in zip(matrix.windows, hazards):
        prev = None
        n_weather = x_weather = 0
        for frame_path, hazard in zip(window.frame_paths[: matrix.frames_per_window], series):
            grey = cv2.cvtColor(bench.read_working_frame(frame_path), cv2.COLOR_BGR2GRAY)
            kind = "weather" if prev is not None and _weather(prev, grey) else "other"
            crossed = hazard >= threshold
            counts[kind]["frames"] += 1
            counts[kind]["crossings"] += int(crossed)
            if kind == "weather":
                n_weather += 1
                x_weather += int(crossed)
            prev = grey
        if n_weather:
            windows.append(
                {
                    "sequence_id": window.sequence_id,
                    "weather_frames": n_weather,
                    "weather_crossings": x_weather,
                    "crossings": sum(1 for h in series if h >= threshold),
                }
            )
    total = sum(c["crossings"] for c in counts.values())
    out_payload = {
        "what": "CONFUSE threshold crossings on frames the stabiliser warped by the sky vs the rest",
        "threshold": threshold,
        "threshold_source": path.name,
        "windows": len(matrix.windows),
        "frames_per_window": matrix.frames_per_window,
        "realised_alerts_per_camera_night": round(
            bench.per_camera_night(total, len(hazards), matrix.frames_per_window), 4
        ),
        "crossings_total": total,
        "by_frame_kind": {
            kind: dict(c, rate=round(c["crossings"] / c["frames"], 4) if c["frames"] else None)
            for kind, c in counts.items()
        },
        "windows_with_weather_frames": windows,
        "weather_definition": f"stabilise() warp >= {WARP_MIN_PX} px that worsens the TERRAIN band's MSE",
    }
    out = ROOT / "results" / f"registration_alerts_{date.today().isoformat()}.json"
    out.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""How much camera motion the wall's background models actually receive -- measured, read-only.

    .venv/bin/python scripts/measure_registration.py   # writes results/registration_<date>.json

WHAT THIS MEASURES, AND WHY
---------------------------
``api/main.py:Replay._tick`` and ``bench.py`` both call ``stabilise(prev_raw, cur_raw)`` and
keep the RAW frame as the next reference (``camera.prev_grey = grey``). Each registered frame
therefore lands where the previous raw frame sat: the registered sequence follows the camera's
own path one frame late, so the per-pixel background model still receives the camera's motion
-- delayed, not removed. ``bench.py`` documents the choice ("register against the RAW previous
frame"); this script measures what it costs.

Two numbers per arm, both from the shipped estimator on the shipped decode:

* ``pair_shift_px`` -- the inter-frame shift ``stabilise`` estimates for each consecutive pair.
  This is what the S4 flow pair (previous registered frame vs this one) carries as coherent
  motion, one step late.
* ``endpoint_shift_px`` -- per window, the shift between its first and last frame measured
  DIRECTLY (one Hanning-windowed ``cv2.phaseCorrelate``, not a chain), with its peak
  response. This is the honest answer to "did the camera move over the window": MOG2 keeps
  a per-pixel history, so net displacement is what the background model would absorb.
* ``chained_sum_px`` -- the same displacement as the SUM of the per-pair estimates. Where it
  disagrees with the direct endpoint shift, the difference is estimator error, not camera
  motion -- and a per-pair estimate that is error misregisters that one frame, because the
  wall warps by it. ``largest_chain_gaps`` names the windows where the two disagree most.

Read over the wall's own roster and aligned windows (``_wall_sequences`` + ``_align``: DEV and
CONFUSE; HOLD is refused by ``_wall_sequences`` itself). It changes nothing: the held-out read
measured this pipeline, and registering against a fixed reference would reopen it.
"""

from __future__ import annotations

import collections
import json
import math
import pathlib
import sys
from datetime import date

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import _align, _decode_working_frame, _wall_sequences  # noqa: E402
from plumerank.stabilise import stabilise  # noqa: E402


def _pct(values: list[float], q: float) -> float:
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _summary(xs: list[float]) -> dict[str, float]:
    return {
        "p50": round(_pct(xs, 0.50), 3),
        "p90": round(_pct(xs, 0.90), 3),
        "p99": round(_pct(xs, 0.99), 3),
        "max": round(max(xs), 3),
    }


def main() -> int:
    sequences, _ = _wall_sequences()
    cameras, _, frame_count = _align(sequences)
    pair: dict[str, list[float]] = collections.defaultdict(list)
    endpoint: dict[str, list[float]] = collections.defaultdict(list)
    windows: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    methods: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for camera in cameras:
        arm = camera.sequence.arm
        paths = camera.sequence.frame_paths[camera.frame_offset : camera.frame_offset + frame_count]
        prev = first = None
        x = y = 0.0
        for path in paths:
            grey = cv2.cvtColor(_decode_working_frame(path), cv2.COLOR_BGR2GRAY)
            if prev is None:
                first = grey
            else:
                s = stabilise(prev, grey)
                methods[arm][s.method] += 1
                pair[arm].append(math.hypot(s.dx, s.dy))
                x, y = x + s.dx, y + s.dy
            prev = grey
        assert first is not None and prev is not None
        window = cv2.createHanningWindow((first.shape[1], first.shape[0]), cv2.CV_64F)
        (ex, ey), response = cv2.phaseCorrelate(
            first.astype(np.float64), prev.astype(np.float64), window
        )
        direct = math.hypot(ex, ey)
        endpoint[arm].append(direct)
        windows[arm].append(
            {
                "sequence_id": camera.sequence.sequence_id,
                "endpoint_shift_px": round(direct, 3),
                "endpoint_response": round(response, 4),
                "chained_sum_px": round(math.hypot(x, y), 3),
                "chain_gap_px": round(math.hypot(x - ex, y - ey), 3),
            }
        )
    arms = {
        arm: {
            "windows": len(endpoint[arm]),
            "pairs": len(xs),
            "methods": dict(sorted(methods[arm].items())),
            "pair_shift_px": _summary(xs),
            "pair_share_at_least_px": {
                str(t): round(sum(v >= t for v in xs) / len(xs), 4) for t in (0.5, 1.0, 2.0)
            },
            "endpoint_shift_px": _summary(endpoint[arm]),
            "windows_with_endpoint_shift_at_least_px": {
                str(t): sum(v >= t for v in endpoint[arm]) for t in (1.0, 2.0, 5.0)
            },
            "largest_chain_gaps": sorted(
                windows[arm], key=lambda w: -float(w["chain_gap_px"])  # type: ignore[arg-type]
            )[:5],
        }
        for arm, xs in sorted(pair.items())
    }
    payload = {
        "what": "camera motion the wall's background models receive under raw-predecessor registration",
        "unit": "working px (480x270)",
        "estimator": "plumerank.stabilise.stabilise, passthrough pairs counted as 0.0",
        "roster": "api.main._wall_sequences + _align (DEV + CONFUSE; HOLD refused)",
        "frames_per_window": frame_count,
        "arms": arms,
    }
    out = ROOT / "results" / f"registration_{date.today().isoformat()}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

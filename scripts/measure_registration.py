#!/usr/bin/env python3
"""What the stabiliser's warps actually correct on the wall's footage -- measured, read-only.

    .venv/bin/python scripts/measure_registration.py   # writes results/registration_<date>.json

WHAT THIS MEASURES, AND WHY
---------------------------
``api/main.py:Replay._tick`` and ``bench.py`` both call ``stabilise(prev_raw, cur_raw)`` and
keep the RAW frame as the next reference (``camera.prev_grey = grey``). Each registered frame
therefore lands where the previous raw frame sat, so real camera motion still reaches the
per-pixel background model -- one frame late, not removed. ``bench.py`` documents the choice
("register against the RAW previous frame"); this script measures what the warps do.

The estimate ``stabilise`` returns is a whole-frame phase correlation. A large shift can be the
camera moving -- or the sky moving under a still camera. So every pair the wall warps by at
least 0.5 px is REFEREED against the ground: a band of the frame (``TERRAIN``, rows 55-88%,
columns 5-95%) clear of the sky, where cloud drifts, and of any top/bottom overlay. If the warp
improves that band's alignment (lower mean squared difference against the previous frame) the
shift was the camera's -- ``camera``; if it worsens it, the ground was still and the shift
followed something else -- ``weather``. The band's own phase-correlation shift and response
are published beside each verdict, so the referee can be checked too, and so is the sky band's
(rows 0-40%) -- context for what a 'weather' estimate was pulled toward.

Also published, with their limits stated:

* ``pair_shift_px`` -- the per-pair estimates themselves, the quantity the wall warps by and the
  S4 flow pair carries one step late.
* ``endpoint_shift_px`` -- one direct first-to-last ``phaseCorrelate`` per window, with its
  response and ``endpoint_trusted`` (response >= ``PHASE_RESPONSE_MIN``, the stabiliser's own
  trust floor). It cannot see a camera that moves and comes back, and below the floor it
  measures nothing; it is context, not a referee.

Read over the wall's own roster and aligned windows (``_wall_sequences`` + ``_align``: DEV and
CONFUSE; HOLD is refused by ``_wall_sequences`` itself). It changes nothing: the held-out read
measured this pipeline, and a different registration would reopen it.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import math
import pathlib
import statistics
import sys
from datetime import date

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import _align, _decode_working_frame, _wall_sequences  # noqa: E402
from plumerank.constants import PHASE_RESPONSE_MIN, WORK_HEIGHT, WORK_WIDTH  # noqa: E402
from plumerank.stabilise import stabilise, warp  # noqa: E402

#: The referee: ground, clear of the sky (where cloud moves) and of any top/bottom overlay.
TERRAIN = (
    slice(int(WORK_HEIGHT * 0.55), int(WORK_HEIGHT * 0.88)),
    slice(int(WORK_WIDTH * 0.05), int(WORK_WIDTH * 0.95)),
)
#: The sky band, for context only: when the ground is still, what did the estimate follow?
SKY = (slice(0, int(WORK_HEIGHT * 0.40)), slice(int(WORK_WIDTH * 0.05), int(WORK_WIDTH * 0.95)))
#: Pairs warped by at least this much are refereed.
REFEREE_MIN_PX: float = 0.5
THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 2.0)


@dataclasses.dataclass(frozen=True)
class Ref:
    """One refereed pair."""

    sequence_id: str
    shift_px: float
    kind: str  # "camera" | "weather"
    terrain_shift_px: float
    terrain_response: float
    sky_shift_px: float


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


def _band_shift(prev: np.ndarray, cur: np.ndarray, band: tuple[slice, slice]) -> tuple[float, float]:
    a, b = prev[band].astype(np.float64), cur[band].astype(np.float64)
    hann = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_64F)
    (tx, ty), response = cv2.phaseCorrelate(a, b, hann)
    return math.hypot(tx, ty), float(response)


def _referee(prev: np.ndarray, cur: np.ndarray, dx: float, dy: float) -> tuple[float, float, bool]:
    """(ground-band shift px, its response, does the warp improve the ground's alignment?)."""
    a, b = prev[TERRAIN], cur[TERRAIN]
    t_shift, response = _band_shift(prev, cur, TERRAIN)
    warped = warp(cur, dx, dy)[TERRAIN]
    raw_mse = float(np.mean((b.astype(np.float32) - a.astype(np.float32)) ** 2))
    warped_mse = float(np.mean((warped.astype(np.float32) - a.astype(np.float32)) ** 2))
    return t_shift, response, warped_mse < raw_mse


def main() -> int:
    sequences, _ = _wall_sequences()
    cameras, _, frame_count = _align(sequences)
    pair: dict[str, list[float]] = collections.defaultdict(list)
    refereed: dict[str, list[Ref]] = collections.defaultdict(list)
    endpoints: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    methods: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for camera in cameras:
        arm, seq_id = camera.sequence.arm, camera.sequence.sequence_id
        paths = camera.sequence.frame_paths[camera.frame_offset : camera.frame_offset + frame_count]
        prev = first = None
        for path in paths:
            grey = cv2.cvtColor(_decode_working_frame(path), cv2.COLOR_BGR2GRAY)
            if prev is None:
                first = grey
            else:
                s = stabilise(prev, grey)
                methods[arm][s.method] += 1
                shift = math.hypot(s.dx, s.dy)
                pair[arm].append(shift)
                if shift >= REFEREE_MIN_PX:
                    t_shift, t_resp, helps = _referee(prev, grey, s.dx, s.dy)
                    refereed[arm].append(
                        Ref(
                            seq_id, shift, "camera" if helps else "weather", t_shift, t_resp,
                            _band_shift(prev, grey, SKY)[0],
                        )
                    )
            prev = grey
        assert first is not None and prev is not None
        hann = cv2.createHanningWindow((first.shape[1], first.shape[0]), cv2.CV_64F)
        (ex, ey), response = cv2.phaseCorrelate(
            first.astype(np.float64), prev.astype(np.float64), hann
        )
        endpoints[arm].append(
            {
                "sequence_id": seq_id,
                "endpoint_shift_px": round(math.hypot(ex, ey), 3),
                "endpoint_response": round(float(response), 4),
                "endpoint_trusted": bool(response >= PHASE_RESPONSE_MIN),
            }
        )

    arms: dict[str, object] = {}
    for arm, xs in sorted(pair.items()):
        refs = refereed[arm]
        by_threshold = {
            str(t): {
                kind: sum(1 for r in refs if r.shift_px >= t and r.kind == kind)
                for kind in ("camera", "weather")
            }
            for t in THRESHOLDS
        }
        at_1px: dict[str, list[Ref]] = collections.defaultdict(list)
        for r in refs:
            if r.shift_px >= 1.0:
                at_1px[r.sequence_id].append(r)
        windows_at_1px = [
            {
                "sequence_id": seq,
                "pairs": len(rs),
                "camera": sum(1 for r in rs if r.kind == "camera"),
                "weather": sum(1 for r in rs if r.kind == "weather"),
                "shift_px_max": round(max(r.shift_px for r in rs), 3),
                "terrain_shift_px_median": round(statistics.median(r.terrain_shift_px for r in rs), 3),
                "terrain_response_median": round(statistics.median(r.terrain_response for r in rs), 3),
                "sky_shift_px_median": round(statistics.median(r.sky_shift_px for r in rs), 3),
            }
            for seq, rs in sorted(at_1px.items(), key=lambda kv: -len(kv[1]))
        ]
        weather_1px = [r.terrain_shift_px for r in refs if r.shift_px >= 1.0 and r.kind == "weather"]
        eps = endpoints[arm]
        arms[arm] = {
            "windows": len(eps),
            "pairs": len(xs),
            "methods": dict(sorted(methods[arm].items())),
            "pair_shift_px": _summary(xs),
            "pair_share_at_least_px": {
                str(t): round(sum(v >= t for v in xs) / len(xs), 4) for t in THRESHOLDS
            },
            "refereed_at_least_px": by_threshold,
            "weather_at_least_1px_terrain_shift_px": _summary(weather_1px) if weather_1px else None,
            "windows_at_least_1px": windows_at_1px,
            "endpoint_shift_px_trusted": (
                _summary([float(e["endpoint_shift_px"]) for e in eps if e["endpoint_trusted"]])  # type: ignore[arg-type]
                if any(e["endpoint_trusted"] for e in eps)
                else None
            ),
            "endpoint_untrusted_windows": sum(1 for e in eps if not e["endpoint_trusted"]),
            "endpoints": eps,
        }
    payload = {
        "what": "what stabilise()'s warps correct on the wall's footage, refereed against the ground",
        "unit": "working px (480x270)",
        "estimator": "plumerank.stabilise.stabilise (whole-frame phase correlation)",
        "referee": (
            "TERRAIN band rows 55-88%, cols 5-95%; a warp that lowers the band's MSE against the "
            f"previous frame is 'camera', one that raises it is 'weather'; pairs >= {REFEREE_MIN_PX} px"
        ),
        "endpoint_trust_floor": PHASE_RESPONSE_MIN,
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

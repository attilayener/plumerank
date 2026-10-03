#!/usr/bin/env python3
"""What KIND of peak beats the fire: the shape of the 11 pre-ignition maxima.

WHY THIS EXISTS
---------------
``results/alert_budget_2026-09-06.md`` §3 is the sharpest statement of the open defect: in 11 of
29 DEV sequences the camera's highest hazard of the night happens **before** ignition, all 11
are misses, and no threshold, budget policy or cross-camera ranking can reach them -- they lose
to themselves. The same document closes the budget direction and names the lever:
*discrimination on the same camera*.

Before anyone touches S1-S4 again, there is a cheaper question. A plume is a process: it
persists and grows over minutes. Most things that are not a plume -- an exposure step at dawn,
a bird, a passing vehicle's headlights, a cloud edge crossing a tile -- are events, and an
event's hazard is a spike. **If the 11 pre-ignition peaks are spikes and the post-ignition
hazard is a plateau, the two are separable in TIME without changing any per-frame statistic**,
and a persistence transform of the hazard series is the mechanism.

This file measures that, and only that.

WHAT IT MEASURES
----------------
1. **Peak shape.** For every sequence, how many pre-ignition frames sit at or above the
   post-ignition maximum, and the longest run of them. A run of 1 is a spike; a run of 10 is
   a plateau that persistence cannot separate.
2. **Inversions under temporal transforms**, threshold-free. Each transform maps the hazard
   series to a new series; an inversion is ``max(pre) > max(post)``. The transforms:

   - ``min_k``: the running minimum over the last ``k`` frames -- "hazard held for k minutes".
     The strictest persistence statistic; a single spike is erased at k = 2.
   - ``median_k``: running median over the last ``k`` frames -- tolerant of one dropped frame.
   - ``ema_a``: exponential moving average -- the cheapest to ship, one float of state.

3. **Rate-matched detections**, through the real ``rank.emit``. For each transform the
   threshold is re-solved on DEV so that the number of frames at or above it equals the
   shipped configuration's count -- the same number of alerts, spent on a different series.

   **This is NOT the calibrated operating point.** The product solves its threshold on the
   CONFUSE arm for a 2.00 alerts/camera-night budget (``bench.py``), and the CONFUSE feature
   dump is not on disk. A DEV-rate-matched threshold is the honest proxy available here and
   it is labelled as such everywhere it is printed. A transform that wins here has earned a
   ``bench.py`` run, not a headline.

WHAT IT DOES NOT DO
-------------------
It changes nothing in ``plumerank/``. Every persistence transform costs latency by
construction -- ``min_k`` cannot fire until k frames after the plume appears, and the cost is
printed next to every gain in the same units (minutes). Whether a minute of median TTD is
worth a sequence of detection rate is the metric's own trade-off, and choosing a transform to
ship is a design decision with a fitted threshold behind it, not a spike's.

Reproduce::

    python3 spike/peak_shape.py            # shape table + inversion table + rate-matched table
    python3 spike/peak_shape.py --json     # the same, as one JSON document on stdout
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spike"))

from why_missed import NO_STATS, SLOTS_PER_CAMERA, THRESHOLD, load  # noqa: E402

from plumerank import constants as K  # noqa: E402
from plumerank import rank as rk  # noqa: E402
from plumerank.metrics import MINUTES_PER_FRAME, detection_index  # noqa: E402

# ---------------------------------------------------------------------------------
# Transforms: series -> series, same length, causal (frame t uses frames <= t only)
# ---------------------------------------------------------------------------------


def identity(series: list[float]) -> list[float]:
    return list(series)


def running_min(series: list[float], k: int) -> list[float]:
    """Hazard held for ``k`` consecutive frames. Frames before k exist take the min so far."""
    out: list[float] = []
    for t in range(len(series)):
        out.append(min(series[max(0, t - k + 1) : t + 1]))
    return out


def running_median(series: list[float], k: int) -> list[float]:
    out: list[float] = []
    for t in range(len(series)):
        out.append(statistics.median(series[max(0, t - k + 1) : t + 1]))
    return out


def ema(series: list[float], alpha: float) -> list[float]:
    out: list[float] = []
    state = series[0]
    for value in series:
        state = alpha * value + (1.0 - alpha) * state
        out.append(state)
    return out


def zscore_trailing(series: list[float], window: int, warmup: int = 5) -> list[float]:
    """Hazard relative to the camera's OWN recent night: (h - mean) / std over the trailing window.

    Causal. Frames inside the warm-up carry the raw value shifted to the same scale as the
    rest (z = 0), so a camera cannot alert before it has a baseline to depart from.
    """
    out: list[float] = []
    for t in range(len(series)):
        past = series[max(0, t - window) : t]
        if len(past) < warmup:
            out.append(0.0)
            continue
        mu = statistics.mean(past)
        sd = statistics.pstdev(past) or 1e-6
        out.append((series[t] - mu) / sd)
    return out


def zscore_expanding(series: list[float], warmup: int = 5) -> list[float]:
    """Same, against everything the camera has seen so far tonight."""
    return zscore_trailing(series, window=10**9, warmup=warmup)


def quantile_trailing(series: list[float], window: int, warmup: int = 5) -> list[float]:
    """Fraction of the camera's trailing window this frame outscores. Scale-free."""
    out: list[float] = []
    for t in range(len(series)):
        past = series[max(0, t - window) : t]
        if len(past) < warmup:
            out.append(0.0)
            continue
        out.append(sum(1 for v in past if series[t] > v) / len(past))
    return out


def transforms() -> list[tuple[str, float, object]]:
    """(name, latency_cost_frames, fn). Latency is the frames a fresh plateau needs to register."""
    out: list[tuple[str, float, object]] = [("shipped (raw)", 0, identity)]
    # -- per-camera baseline family: zero added latency, a warm-up instead
    out.append(("z_expanding", 0, zscore_expanding))
    for w in (15, 30):
        out.append((f"z_trail_{w}", 0, lambda s, w=w: zscore_trailing(s, w)))
    for w in (15, 30):
        out.append((f"q_trail_{w}", 0, lambda s, w=w: quantile_trailing(s, w)))
    # -- baseline THEN persistence: does a plume depart from its own night for longer than a spike does?
    out.append(("z_exp+min_3", 2, lambda s: running_min(zscore_expanding(s), 3)))
    out.append(("z_exp+med_3", 1, lambda s: running_median(zscore_expanding(s), 3)))
    # -- persistence-only family
    for k in (2, 3, 4, 5, 7, 10):
        out.append((f"min_{k}", k - 1, lambda s, k=k: running_min(s, k)))
    for k in (3, 5, 7):
        out.append((f"median_{k}", (k - 1) / 2, lambda s, k=k: running_median(s, k)))
    for a in (0.5, 0.3, 0.2):
        # Frames for an EMA to reach 90% of a step: ln(0.1)/ln(1-a)
        import math

        out.append((f"ema_{a}", math.log(0.1) / math.log(1 - a), lambda s, a=a: ema(s, a)))
    return out


# ---------------------------------------------------------------------------------
# 1. Peak shape
# ---------------------------------------------------------------------------------


def longest_run(flags: list[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def peak_shape(keys, hazards, t0) -> list[dict]:
    rows = []
    for key in keys:
        h = hazards[key]
        ign = t0[key]
        pre, post = h[:ign], h[ign:]
        pre_max, post_max = max(pre), max(post)
        above = [v >= post_max for v in pre]
        rows.append(
            {
                "sequence": key,
                "inverted": pre_max > post_max,
                "pre_max": pre_max,
                "post_max": post_max,
                "excess": pre_max - post_max,
                # how many pre-ignition minutes outscore the fire's best minute, and for how long at a stretch
                "pre_frames_at_or_above_post_max": sum(above),
                "pre_longest_run_at_or_above_post_max": longest_run(above),
                # the mirror: how long the fire holds ABOVE the quiet night's best minute
                "post_longest_run_above_pre_max": longest_run([v > pre_max for v in post]),
            }
        )
    return rows


# ---------------------------------------------------------------------------------
# 2. Inversions per transform, threshold-free
# ---------------------------------------------------------------------------------


def inversions(keys, hazards, t0, fn) -> tuple[int, list[str]]:
    bad = []
    for key in keys:
        s = fn(hazards[key])
        ign = t0[key]
        if max(s[:ign]) > max(s[ign:]):
            bad.append(key)
    return len(bad), bad


# ---------------------------------------------------------------------------------
# 3. Rate-matched detections through the real emit()
# ---------------------------------------------------------------------------------


def rate_matched_threshold(series_by_key: dict[str, list[float]], target_count: int) -> float:
    """The threshold at which exactly ``target_count`` frames sit at or above it (ties: fewer)."""
    values = sorted((v for s in series_by_key.values() for v in s), reverse=True)
    if target_count <= 0:
        return float("inf")
    return values[min(target_count, len(values)) - 1]


def detections(keys, series_by_key, t0, n_frames, threshold) -> tuple[int, list[float]]:
    budget = rk.BudgetState(keys, per_camera=SLOTS_PER_CAMERA)
    index = {key: i for i, key in enumerate(keys)}
    in_top_k = [[False] * n_frames for _ in keys]
    alerted = [[False] * n_frames for _ in keys]
    for t in range(n_frames):
        scores = [
            rk.CameraScore(
                camera_id=key,
                frame_index=t,
                hazard=series_by_key[key][t],
                stats=NO_STATS,
                component=None,
            )
            for key in keys
        ]
        for entry in rk.emit(scores, threshold, budget):
            i = index[entry.camera_id]
            in_top_k[i][t] = entry.rank <= K.TOP_K
            alerted[i][t] = entry.alerted
    ttds = []
    for i, key in enumerate(keys):
        found = detection_index(in_top_k[i], alerted[i], t0[key])
        if found is not None:
            ttds.append((found - t0[key]) * MINUTES_PER_FRAME)
    return len(ttds), ttds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", action="store_true", help="emit one JSON document instead of tables")
    args = parser.parse_args(argv)

    keys, hazards, t0, n_frames = load()
    shape = peak_shape(keys, hazards, t0)
    shipped_count = sum(1 for s in hazards.values() for v in s if v >= THRESHOLD)

    table = []
    for name, latency, fn in transforms():
        series = {key: fn(hazards[key]) for key in keys}
        n_inv, which = inversions(keys, hazards, t0, fn)
        thr = rate_matched_threshold(series, shipped_count)
        n_det, ttds = detections(keys, series, t0, n_frames, thr)
        table.append(
            {
                "transform": name,
                "latency_cost_min": latency * MINUTES_PER_FRAME,
                "inversions": n_inv,
                "inverted_sequences": which,
                "dev_rate_matched_threshold": thr,
                "detected": n_det,
                "mean_ttd_min": (sum(ttds) / len(ttds)) if ttds else None,
                "median_ttd_min_if_exists": statistics.median(ttds) if n_det * 2 > len(keys) else None,
            }
        )

    doc = {
        "shipped_threshold": THRESHOLD,
        "shipped_frames_at_or_above": shipped_count,
        "sequences": len(keys),
        "frames_per_sequence": n_frames,
        "minutes_per_frame": MINUTES_PER_FRAME,
        "peak_shape": shape,
        "transforms": table,
        "caveat": (
            "Detections are at a DEV-rate-matched threshold, NOT the CONFUSE-calibrated "
            "operating point bench.py ships. A transform that wins here has earned a bench.py "
            "run, not a headline."
        ),
    }
    if args.json:
        print(json.dumps(doc, indent=2))
        return 0

    inv = [r for r in shape if r["inverted"]]
    print(f"shipped threshold {THRESHOLD}   frames at/above on DEV {shipped_count}   sequences {len(keys)}\n")
    print("1. PEAK SHAPE -- the inverted sequences (quiet night outscores the fire)\n")
    print(f"{'sequence':<44}{'excess':>8}{'pre>=post':>10}{'pre run':>8}{'post run':>9}")
    for r in sorted(inv, key=lambda r: -r["excess"]):
        print(
            f"{r['sequence']:<44}{r['excess']:>8.4f}{r['pre_frames_at_or_above_post_max']:>10}"
            f"{r['pre_longest_run_at_or_above_post_max']:>8}{r['post_longest_run_above_pre_max']:>9}"
        )
    runs = [r["pre_longest_run_at_or_above_post_max"] for r in inv]
    print(
        f"\n  {len(inv)} inverted.  longest pre-ignition run at/above the fire's best minute: "
        f"median {statistics.median(runs)}, max {max(runs)}  (1 = a single-frame spike)"
    )
    det = [r for r in shape if not r["inverted"]]
    post_runs = [r["post_longest_run_above_pre_max"] for r in det]
    print(
        f"  {len(det)} not inverted.  fire holds above the quiet night's best minute for: "
        f"median {statistics.median(post_runs)} frames, min {min(post_runs)}"
    )

    print("\n2. INVERSIONS UNDER TEMPORAL TRANSFORMS (threshold-free)  and  3. DEV-RATE-MATCHED DETECTIONS\n")
    print(f"{'transform':<16}{'latency':>8}{'inv':>5}{'thr':>9}{'det':>6}{'mean ttd':>10}{'median':>8}")
    for r in table:
        med = "" if r["median_ttd_min_if_exists"] is None else f"{r['median_ttd_min_if_exists']:.1f}"
        mean = "" if r["mean_ttd_min"] is None else f"{r['mean_ttd_min']:.1f}"
        print(
            f"{r['transform']:<16}{r['latency_cost_min']:>8.1f}{r['inversions']:>5}"
            f"{r['dev_rate_matched_threshold']:>9.4f}{r['detected']:>3}/{len(keys):<2}{mean:>10}{med:>8}"
        )
    print(f"\n  {doc['caveat']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

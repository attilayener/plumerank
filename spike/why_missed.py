#!/usr/bin/env python3
"""Why each missed DEV sequence was missed, and what the two alert slots are spent on.

WHY THIS EXISTS
---------------
``results/SESSION_2026-09-06.md`` §4 records the project's largest open defect: the median
time-to-detection the one-liner is written around does not exist, because the detection rate
is far below 50%. ``results/statistics_confound_2026-09-06.md`` §4b located the loss
downstream of the segmenter -- 25 of 29 sequences carry a segmented component on more than
80% of their post-ignition frames -- and concluded that the ranking and fusion stage is where
the detections are lost.

This file asks the next question: **lost how?** It re-runs the real detection rule over the
per-frame feature dump and attributes every miss to a cause.

WHAT IT REPRODUCES, AND WHY THAT MATTERS
----------------------------------------
Run with no arguments it reproduces ``results/arm_dev_2026-09-06.json`` exactly -- **9 of 29,
31.0%**, the same 9 -- from ``results/features_dev.jsonl`` plus the shipped constants, without
touching a pixel. That check is the reason anything below is quotable, and writing it is what
turned up ``results/artefact_provenance_2026-09-06.md``: the reproduction failed against every
committed artefact until the arm was re-run, at which point the plain-named DEV file turned out
to be an ``--s4 off`` ablation cell that four documents had been reading as the product.

WHAT IT DOES NOT DO
-------------------
It changes nothing. ``--policies`` measures counterfactual ways of spending the two alert
slots and prints what each would have detected; no policy here is implemented in
``plumerank/rank.py``, and choosing one is a design decision about what an alert means, which
is a human decision.

Reproduce::

    python3 spike/why_missed.py               # the attribution
    python3 spike/why_missed.py --policies    # the counterfactual slot policies
    python3 spike/why_missed.py --sweep       # the refractory period, end to end
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plumerank import constants as K  # noqa: E402
from plumerank import rank as rk  # noqa: E402
from plumerank.metrics import MINUTES_PER_FRAME, detection_index  # noqa: E402
from plumerank.types import Stats  # noqa: E402

FEATURES = ROOT / "results" / "features_dev.jsonl"

#: The threshold ``bench.py`` solved on CONFUSE for a 2.00 alerts/camera-night budget, from
#: ``results/arm_dev_2026-09-06.json`` -- the SHIPPED configuration. It is an INPUT here and is
#: never re-solved: the point of every measurement below is what happens at the operating point
#: the project actually ships.
#:
#: It is deliberately not read from the newest ``arm_dev_*.json`` on disk. ``bench.py`` puts no
#: ablation cell in its filename, so the plain artefact has already been an ``--s4 off`` run
#: once, and every document that read it reported an ablation cell as the product
#: (``results/artefact_provenance_2026-09-06.md``). A hard-coded number that a reader can check
#: against one named file is safer here than a glob that silently follows whatever ran last.
#:
#: It is also unaffected by anything in this file. ``rank.realised_rate`` counts every frame
#: at or above the threshold, uncapped and with no reference to the budget, so a different
#: slot policy changes which alerts are EMITTED and leaves the calibration -- and the
#: published realised rate -- exactly where they were.
THRESHOLD: float = 0.740112

SLOTS_PER_CAMERA: int = 2

NO_STATS = Stats(0.0, 0.0, 0.0, 0.0)


def load() -> tuple[list[str], dict[str, list[float]], dict[str, int], int]:
    """Per-camera hazard series from the committed feature dump, at the shipped weights."""
    if not FEATURES.is_file():
        raise SystemExit(
            f"{FEATURES} is missing -- run `bench.py --arm dev --dump-features` first"
        )
    rows = [json.loads(line) for line in FEATURES.read_text().splitlines() if line.strip()]
    by_sequence: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    t0: dict[str, int] = {}
    for row in rows:
        by_sequence[row["sequence_id"]][row["frame_index"]] = row
        t0[row["sequence_id"]] = row["t0_index"]
    keys = sorted(by_sequence)
    n_frames = max(max(frames) for frames in by_sequence.values()) + 1

    def hazard(row: dict) -> float:
        z = (
            K.FUSION_BIAS
            + K.FUSION_W_S1 * row["s1"]
            + K.FUSION_W_S2 * row["s2"]
            + K.FUSION_W_S3 * row["s3"]
            + K.FUSION_W_ESCALATION * row["esc"]
        )
        if K.S4_ENABLED:
            z += K.FUSION_W_S4 * row["s4"]
        return 1.0 / (1.0 + math.exp(-z))

    hazards = {
        key: [hazard(by_sequence[key][t]) for t in range(n_frames)] for key in keys
    }
    return keys, hazards, t0, n_frames


def ranks(keys: list[str], hazards: dict[str, list[float]], n_frames: int) -> dict[str, list[int]]:
    """Rank by hazard at every tick -- descending, ties on camera id, as ``rank.emit``.

    Computed once and shared by every policy below, because no policy in this file changes
    the ORDER: a slot policy decides what an alert costs, not what is worth looking at.
    """
    out = {key: [0] * n_frames for key in keys}
    for t in range(n_frames):
        for position, key in enumerate(sorted(keys, key=lambda k: (-hazards[k][t], k))):
            out[key][t] = position + 1
    return out


# ---------------------------------------------------------------------------------
# 1. The attribution -- run through the real emit(), so it must reproduce the benchmark
# ---------------------------------------------------------------------------------


def attribute() -> int:
    keys, hazards, t0, n_frames = load()
    budget = rk.BudgetState(keys, per_camera=SLOTS_PER_CAMERA)
    index = {key: i for i, key in enumerate(keys)}
    in_top_k = [[False] * n_frames for _ in keys]
    alerted = [[False] * n_frames for _ in keys]
    spent_at: dict[str, list[int]] = collections.defaultdict(list)

    for t in range(n_frames):
        scores = [
            rk.CameraScore(
                camera_id=key,
                frame_index=t,
                hazard=hazards[key][t],
                stats=NO_STATS,
                component=None,
            )
            for key in keys
        ]
        for entry in rk.emit(scores, THRESHOLD, budget):
            i = index[entry.camera_id]
            in_top_k[i][t] = entry.rank <= K.TOP_K
            alerted[i][t] = entry.alerted
            if entry.alerted:
                spent_at[entry.camera_id].append(t)

    detail = []
    reasons: collections.Counter[str] = collections.Counter()
    for i, key in enumerate(keys):
        ignition = t0[key]
        found = detection_index(in_top_k[i], alerted[i], ignition)
        post_max = max(hazards[key][ignition:])
        pre_max = max(hazards[key][:ignition])
        crosses_after = post_max >= THRESHOLD
        spent_before = sum(1 for f in spent_at[key] if f < ignition)
        in_top_after = any(in_top_k[i][t] for t in range(ignition, n_frames))
        if found is not None:
            reason = "DETECTED"
        elif not crosses_after:
            reason = "hazard never crosses after t0"
        elif spent_before >= SLOTS_PER_CAMERA:
            reason = "budget spent BEFORE ignition"
        elif not in_top_after:
            reason = f"crosses, but never in the top {K.TOP_K}"
        else:
            reason = "unexplained -- read this one by hand"
        reasons[reason] += 1
        detail.append(
            (
                key,
                reason,
                post_max,
                pre_max,
                spent_before,
                None if found is None else (found - ignition) * MINUTES_PER_FRAME,
            )
        )

    print(f"threshold {THRESHOLD}   slots/camera {SLOTS_PER_CAMERA}   cameras {len(keys)}\n")
    print(f"{'sequence':<44}{'why':<32}{'maxH post':>10}{'maxH pre':>9}{'pre':>5}{'ttd':>7}")
    for key, reason, post_max, pre_max, spent_before, ttd in sorted(
        detail, key=lambda d: (d[1], -d[2])
    ):
        shown = "" if ttd is None else f"{ttd:7.1f}"
        print(
            f"{key:<44}{reason:<32}{post_max:>10.4f}{pre_max:>9.4f}"
            f"{spent_before:>5}{shown:>7}"
        )
    detected = reasons["DETECTED"]
    print(f"\n  {detected}/{len(keys)} = {detected / len(keys):.1%} detected")
    for reason, count in reasons.most_common():
        print(f"  {count:>3}  {reason}")
    print(
        "\nThe benchmark reports 9/29 = 31.0% at this threshold "
        "(results/arm_dev_2026-09-06.json, the shipped configuration). A different number "
        "here means this file and bench.py disagree about the detection rule, and nothing "
        "above is quotable."
    )
    return 0


# ---------------------------------------------------------------------------------
# 2. The counterfactual -- what else the two slots could have been spent on
# ---------------------------------------------------------------------------------


def _shipped(series: list[float]) -> list[int]:
    """What ``rank.BudgetState`` does today: the first two crossings of the night."""
    out: list[int] = []
    for t, value in enumerate(series):
        if value >= THRESHOLD and len(out) < SLOTS_PER_CAMERA:
            out.append(t)
    return out


def _rising_edge(series: list[float]) -> list[int]:
    """One slot per contiguous excursion, so a long crossing costs one alert and not two."""
    out: list[int] = []
    armed = True
    for t, value in enumerate(series):
        if value >= THRESHOLD:
            if armed and len(out) < SLOTS_PER_CAMERA:
                out.append(t)
                armed = False
        else:
            armed = True
    return out


def _refractory(series: list[float], frames: int) -> list[int]:
    """No second alert on the same camera within ``frames`` of the first."""
    out: list[int] = []
    last = -(10**9)
    for t, value in enumerate(series):
        if value >= THRESHOLD and t - last >= frames and len(out) < SLOTS_PER_CAMERA:
            out.append(t)
            last = t
    return out


def _improvement(series: list[float], delta: float) -> list[int]:
    """The second slot is spent only on a hazard ``delta`` above the first alert's."""
    out: list[int] = []
    first = 0.0
    for t, value in enumerate(series):
        if value < THRESHOLD:
            continue
        if not out:
            out.append(t)
            first = value
        elif len(out) < SLOTS_PER_CAMERA and value >= first + delta:
            out.append(t)
    return out


def _oracle(series: list[float]) -> list[int]:
    """The two highest hazards of the night. NOT CAUSAL -- an upper bound, never a policy."""
    crossings = [t for t, value in enumerate(series) if value >= THRESHOLD]
    return sorted(sorted(crossings, key=lambda t: -series[t])[:SLOTS_PER_CAMERA])


def _score(policy, keys, hazards, t0, order) -> tuple[int, int, list[float]]:
    detected = 0
    alerts = 0
    ttds: list[float] = []
    for key in keys:
        fired = policy(hazards[key])
        alerts += len(fired)
        ignition = t0[key]
        hits = [t for t in fired if t >= ignition and order[key][t] <= K.TOP_K]
        if hits:
            detected += 1
            ttds.append((min(hits) - ignition) * MINUTES_PER_FRAME)
    return detected, alerts, ttds


def _line(name: str, detected: int, total: int, alerts: int, ttds: list[float], note: str = "") -> None:
    ttds = sorted(ttds)
    exists = len(ttds) * 2 > total
    p50 = f"{ttds[len(ttds) // 2]:.1f}" if exists else "n/a"
    mean = f"{sum(ttds) / len(ttds):.1f}" if ttds else "n/a"
    print(
        f"{name:<36}{detected:>3}/{total} = {detected / total:>6.1%}"
        f"{alerts:>8}{mean:>9}{p50:>8}   {note}"
    )


def policies() -> int:
    keys, hazards, t0, n_frames = load()
    order = ranks(keys, hazards, n_frames)
    total = len(keys)
    print(
        f"threshold {THRESHOLD} held FIXED -- realised_rate() is uncapped and budget-blind, "
        f"so no policy here moves the calibration or the published alert rate.\n"
    )
    print(f"{'policy':<36}{'detected':>15}{'alerts':>8}{'mean':>9}{'p50':>8}")

    def row(name: str, policy, note: str = "") -> None:
        detected, alerts, ttds = _score(policy, keys, hazards, t0, order)
        _line(name, detected, total, alerts, ttds, note)

    row("shipped (first 2 crossings)", _shipped)
    row("rising edge (1 per excursion)", _rising_edge)
    for frames in (5, 10, 20, 40):
        row(
            f"refractory {frames} frames ({frames * MINUTES_PER_FRAME:.0f} min)",
            lambda s, f=frames: _refractory(s, f),
        )
    for delta in (0.02, 0.05, 0.10):
        row(f"2nd slot needs +{delta:.2f} hazard", lambda s, d=delta: _improvement(s, d))
    row("ORACLE: the two highest", _oracle, "NOT CAUSAL")
    return 0


def sweep() -> int:
    """The refractory period end to end -- a plateau is a finding, a spike is a fit."""
    keys, hazards, t0, n_frames = load()
    order = ranks(keys, hazards, n_frames)
    total = len(keys)
    print(f"{'frames':>7}{'min':>7}{'detected':>13}{'alerts':>8}{'p50':>8}")
    for frames in list(range(0, 46, 2)) + [50, 60, n_frames]:
        detected, alerts, ttds = _score(
            lambda s, f=frames: _refractory(s, f), keys, hazards, t0, order
        )
        ttds = sorted(ttds)
        p50 = f"{ttds[len(ttds) // 2]:.1f}" if len(ttds) * 2 > total else "n/a"
        print(
            f"{frames:>7}{frames * MINUTES_PER_FRAME:>7.1f}"
            f"{detected:>6}/{total}{detected / total:>7.1%}{alerts:>8}{p50:>8}"
        )
    print(
        f"\nThe pre-ignition half of every DEV window is {t0[keys[0]]} frames long. A "
        f"refractory period at or above that number forces the two slots to straddle "
        f"ignition BY CONSTRUCTION, so anything from there upward is fitted to the corpus's "
        f"geometry rather than to a dispatcher's day. Read the low end of the plateau only."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", action="store_true", help="counterfactual slot policies")
    parser.add_argument("--sweep", action="store_true", help="sweep the refractory period")
    args = parser.parse_args(argv)
    if args.policies:
        return policies()
    if args.sweep:
        return sweep()
    return attribute()


if __name__ == "__main__":
    raise SystemExit(main())

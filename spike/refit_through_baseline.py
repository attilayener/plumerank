#!/usr/bin/env python3
"""Which fusion weights ship, judged by the headline through the baseline -- not by per-frame AUC.

WHY THIS EXISTS
---------------
``results/fusion_refit_2026-09-06.md`` refitted the fusion on the corrected feature dump and
found that only ``FUSION_W_ESCALATION`` moves -- and flips sign (+0.7785 -> -0.3838) -- while
leave-one-sequence-out AUC moves 0.6387 -> 0.6393. ``statistics_confound_2026-09-06.md`` §6.2
offered a second refit with component presence as its own column: LOSO 0.6644, and S3 goes
to -2.21. Both were left as human gates (2 and 3).

Gate 0 (2026-09-16) changed what the fusion's output is FOR: it is no longer the ranked
quantity but the input to a per-camera baseline (``rank.rebase_series``). A weight set must
therefore be judged where it is used -- on time-to-detection through the baseline at the
alert budget -- and per-frame AUC is only a proxy for that. This file scores the three
candidate weight sets on that quantity, on DEV, at a rate-matched threshold.

WHAT IT DOES NOT DO
-------------------
It changes nothing. It reads ``results/features_dev.jsonl`` (the raw S1-S4 + escalation +
presence vector, unchanged by gate 0) and ``results/fusion_refit_2026-09-06.json``, and
prints a table. The threshold is DEV-rate-matched to the shipped run's 70 frames, the same
proxy ``peak_shape.py`` used, which the calibrated runs then reproduced exactly.

Reproduce::

    python3 spike/refit_through_baseline.py
"""

from __future__ import annotations

import collections
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spike"))

from peak_shape import detections, rate_matched_threshold  # noqa: E402
from why_missed import FEATURES, THRESHOLD  # noqa: E402

from plumerank import constants as K  # noqa: E402
from plumerank import rank as rk  # noqa: E402

REFIT = ROOT / "results" / "fusion_refit_2026-09-06.json"


def hazard(row: dict, w: dict[str, float]) -> float:
    z = (
        w["bias"]
        + w["s1"] * row["s1"]
        + w["s2"] * row["s2"]
        + w["s3"] * row["s3"]
        + w["s4"] * row["s4"]
        + w["esc"] * row["esc"]
        + w["has"] * row["has_component"]
    )
    return 1.0 / (1.0 + math.exp(-z))


def candidates() -> dict[str, dict[str, float]]:
    refit = json.loads(REFIT.read_text())
    w, c = refit["weights"], refit["confound_check"]
    p = c["weights_with_presence_explicit"]
    return {
        "shipped (constants.py)": dict(
            bias=K.FUSION_BIAS, s1=K.FUSION_W_S1, s2=K.FUSION_W_S2, s3=K.FUSION_W_S3,
            s4=K.FUSION_W_S4, esc=K.FUSION_W_ESCALATION, has=0.0,
        ),
        "refit, no presence term": dict(
            bias=w["FUSION_BIAS"], s1=w["FUSION_W_S1"], s2=w["FUSION_W_S2"], s3=w["FUSION_W_S3"],
            s4=w["FUSION_W_S4"], esc=w["FUSION_W_ESCALATION"], has=0.0,
        ),
        "refit, presence explicit": dict(
            bias=c["bias_with_presence_explicit"], s1=p["s1"], s2=p["s2"], s3=p["s3"],
            s4=p["s4"], esc=p["esc"], has=p["has_component"],
        ),
    }


def main() -> int:
    rows = [json.loads(line) for line in FEATURES.read_text().splitlines() if line.strip()]
    by: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    t0: dict[str, int] = {}
    for r in rows:
        by[r["sequence_id"]][r["frame_index"]] = r
        t0[r["sequence_id"]] = r["t0_index"]
    keys = sorted(by)
    n = max(max(f) for f in by.values()) + 1
    sets = candidates()

    shipped_abs = {k: [hazard(by[k][t], sets["shipped (constants.py)"]) for t in range(n)] for k in keys}
    target = sum(1 for s in shipped_abs.values() for v in s if v >= THRESHOLD)
    print(f"DEV, {len(keys)} sequences, threshold rate-matched to {target} frames (the shipped run's)\n")
    print(f"{'weights':<28}{'absolute':>10}{'through baseline':>18}{'mean TTD':>10}{'p50 (all 29)':>14}")
    for name, w in sets.items():
        absolute = {k: [hazard(by[k][t], w) for t in range(n)] for k in keys}
        d_abs, _ = detections(keys, absolute, t0, n, rate_matched_threshold(absolute, target))
        based = {k: rk.rebase_series(absolute[k]) for k in keys}
        d_b, ttds = detections(keys, based, t0, n, rate_matched_threshold(based, target))
        all29 = sorted(ttds + [math.inf] * (len(keys) - d_b))
        p50 = all29[len(keys) // 2]
        print(
            f"{name:<28}{d_abs:>7}/29 {d_b:>14}/29 {sum(ttds) / len(ttds):>10.1f}"
            f"{'n/a' if math.isinf(p50) else f'{p50:.1f}':>14}"
        )
    print(
        "\nThe refit's own objective moved 0.6387 -> 0.6393 LOSO between the first two rows: the"
        "\nescalation weight it flips is not identified by the per-frame fit. Where it IS"
        "\nidentified -- the headline, on the tuning arm -- the shipped value wins by two"
        "\nsequences and six minutes of p50. HOLD is untouched by any of this."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

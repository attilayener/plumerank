#!/usr/bin/env python3
"""What each confusion class costs of the alert budget -- the D11 measurement.

``results/gonogo_2026-09-01.md`` §4 wrote the D11 abort criterion in terms of what "the fog
windows alone contribute" at the threshold the whole CONFUSE arm calibrates to.
``results/ruling_d11_contribute.md`` then fixed, before any fitted number existed, that the
binding quantity is **fog's OWN RATE on fog windows alone**, not its share of the arm's
budget -- on three grounds: §4 states its cutlines in alerts/camera-night, which is a rate
and not a dimensionless share; §4 glosses its own branch as "fog eats the budget every
morning", which is a statement about a foggy camera-night; and the share moves with the
arm's class composition, which §7.5 records as supply-limited (fog 13 windows against a
quota of 18) while the own rate does not.

Both are printed here anyway. Publishing only the binding one would leave a reader unable to
check that the ruling was applied rather than asserted.

This exists as committed tooling rather than a scratch script for the reason §7.1 gives: a
number that decides whether a project continues has to be reproducible by someone else.
"""

import argparse
import collections
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import bench  # noqa: E402

from plumerank import rank as rk  # noqa: E402

#: From gonogo §4, written 2026-09-01 before any of this was measured. Not adjustable.
CLEAR_BELOW = 0.5
ABORT_ABOVE = 1.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=None)
    ap.add_argument("--bg", default="mog2")
    args = ap.parse_args()
    root = (args.root or bench.repo_root()).resolve()

    manifest = json.loads((root / "data" / "manifest_confuse.json").read_text())
    klass = {
        w["sequence_id"]: w.get("confusion_class") or w.get("class")
        for w in manifest["windows"]
    }

    exclusion_records, _ = bench.load_exclusions(root)
    excluded = [r["sequence_id"] for r in exclusion_records]
    windows = bench.load_arm(root, "confuse", excluded)
    (windows,) = bench.unify_windows(windows)

    matrix = bench.replay_arm(
        windows,
        arm="confuse",
        background=args.bg,
        s4_enabled=True,
        escalation_enabled=True,
        classifier=bench._classifier(root),
    )
    hazards = matrix.hazards["plumerank"]

    # The threshold is solved on the WHOLE arm, exactly as bench.py does and exactly as §4
    # specifies -- never per class. A per-class threshold would make each class hit 2.0 by
    # construction and the criterion would be unfalsifiable.
    threshold = rk.solve_threshold(hazards)
    whole = rk.realised_rate(hazards, threshold)

    by_class: dict[str, list[list[float]]] = collections.defaultdict(list)
    for w, hz in zip(matrix.windows, hazards):
        by_class[klass.get(w.sequence_id, "unlabelled")].append(hz)

    total_alerts = sum(1 for hz in hazards for h in hz if h >= threshold)
    rows = []
    for name, group in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        alerts = sum(1 for hz in group for h in hz if h >= threshold)
        rows.append(
            {
                "class": name,
                "windows": len(group),
                "alerts": alerts,
                "own_rate": round(rk.realised_rate(group, threshold), 4),
                "share_of_arm_budget": round(
                    (alerts / total_alerts * whole) if total_alerts else 0.0, 4
                ),
            }
        )

    fog = next((r for r in rows if r["class"] == "fog_marine"), None)
    verdict, binding = "UNDETERMINED", None
    if fog:
        binding = fog["own_rate"]
        if binding < CLEAR_BELOW:
            verdict = "CLEAR"
        elif binding > ABORT_ABOVE:
            verdict = "ABORT-BRANCH: escalate to the human gate"
        else:
            verdict = "PROCEED, published as a named limitation"

    print(f"threshold solved on the whole arm : {threshold:.6f}")
    print(f"whole-arm realised rate           : {whole:.4f} alerts/camera-night\n")
    print(f"{'class':<20}{'win':>5}{'alerts':>8}{'own rate':>11}{'budget share':>14}")
    for r in rows:
        print(f"{r['class']:<20}{r['windows']:>5}{r['alerts']:>8}"
              f"{r['own_rate']:>11.4f}{r['share_of_arm_budget']:>14.4f}")
    print(f"\nD11 binding quantity (ruling §2): fog_marine OWN RATE = {binding}")
    print(f"cutlines from gonogo §4, fixed 2026-09-01: clear < {CLEAR_BELOW} · "
          f"abort > {ABORT_ABOVE}")
    print(f"VERDICT: {verdict}")

    payload = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": bench.git_commit(root),
        "arm": "confuse",
        "background": args.bg,
        "n_windows": len(windows),
        "threshold": threshold,
        "whole_arm_rate": whole,
        "binding_quantity": "fog_marine own rate (results/ruling_d11_contribute.md §2)",
        "binding_value": binding,
        "cutlines": {"clear_below": CLEAR_BELOW, "abort_above": ABORT_ABOVE},
        "verdict": verdict,
        "classes": rows,
        "constants_sha256": bench.K.freeze_hash(),
    }
    out = root / "results" / f"class_split_{dt.date.today()}.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwritten -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

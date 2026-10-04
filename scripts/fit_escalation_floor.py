#!/usr/bin/env python3
"""Solve the escalation hazard floor on CONFUSE — the same method the alert budget uses.

WHY THIS EXISTS
---------------
``escalate.should_escalate`` gates the ONNX forward on ``rank <= ESCALATION_TOP_K and
hazard >= ESCALATION_HAZARD_MIN``. The floor used to be the absolute constant 0.35, chosen
against the a-priori ``FUSION_BIAS`` of -3.2, where a quiet camera sat at sigmoid(-3.2)=0.0392
and the gate had ten times its own margin to work in.

Fitting the fusion weights on 2026-09-06 moved the bias to -0.6961. That put a quiet camera at
**0.3327, below the 0.35 floor**, and the gate stopped gating: escalation ran at 5.0% of frames
against a 2% design maximum and fired on pre-ignition frames. Re-deriving the floor as a fixed
logit margin above the bias fixed that and overshot: escalation fell to **0.00%** and the ONNX
path never executed at all, which is worse. A ``dnn`` integration that never runs is the
decorative-SDK failure.

Neither an absolute constant nor an inherited margin survives a refit of the fusion. So the
floor is solved, on CONFUSE, against a design target — **which is exactly what
``rank.solve_threshold`` already does for the alert budget**, and is defended in the same
terms: the operating point is chosen a priori and the threshold that achieves it is derived
from data rather than guessed.

WHAT THIS CHANGES ABOUT THE COST CLAIM, STATED PLAINLY
------------------------------------------------------
``should_escalate``'s docstring said the realised escalation fraction is "measured" and that
"an adaptive rule would make the escalation fraction a fitted quantity, and the whole point is
that it is measured." **After this change that sentence is no longer true of the floor**, and
it has been corrected in place rather than left standing. What remains true, and is the part
the cost argument actually needs:

* the ``rank <= ESCALATION_TOP_K`` half of the gate is still structural and still bounds cost
  at ``TOP_K / N`` whatever the hazards do -- no calibration can loosen it;
* the floor is calibrated on the NON-FIRE arm only, so it is never tuned against fire windows;
* the fraction realised on any fire arm is still measured and published, not targeted.
"""

import argparse
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import bench  # noqa: E402

from plumerank import constants as K  # noqa: E402

#: The design target, chosen a priori and not derived from any run -- the "~1%" that
#: should_escalate's docstring has always named. It is the analogue of
#: BUDGET_ALERTS_PER_CAMERA_NIGHT = 2.0: an operating point, not a fitted value.
TARGET_FRACTION = 0.01


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=None)
    ap.add_argument("--target", type=float, default=TARGET_FRACTION)
    args = ap.parse_args()
    root = (args.root or bench.repo_root()).resolve()

    records, _ = bench.load_exclusions(root)
    windows = bench.load_arm(root, "confuse", [r["sequence_id"] for r in records])
    (windows,) = bench.unify_windows(windows)

    # Escalation is decided BEFORE the tile score folds in, on the classical hazard alone,
    # so the floor is solved on a replay with escalation disabled. Solving it on hazards
    # that already contain an escalation term would be circular.
    matrix = bench.replay_arm(
        windows,
        arm="confuse",
        background="mog2",
        s4_enabled=True,
        escalation_enabled=False,
        classifier=bench._classifier(root),
        systems=("plumerank",),
    )
    hz = matrix.hazards["plumerank"]
    n_frames = len(hz) * len(hz[0])

    # Only cameras inside the structural gate can ever escalate, so only they can be spent.
    eligible = []
    for t in range(len(hz[0])):
        tick = sorted(
            ((hz[i][t], windows[i].sequence_id) for i in range(len(hz))),
            key=lambda p: (-p[0], p[1]),
        )
        eligible.extend(h for h, _ in tick[: K.ESCALATION_TOP_K])

    eligible.sort(reverse=True)
    budget = int(round(args.target * n_frames))
    ceiling = len(eligible) / n_frames

    if budget <= 0:
        raise SystemExit(f"target {args.target} is below one frame of {n_frames}")
    if budget >= len(eligible):
        floor = 0.0
        note = "target exceeds the structural ceiling; the rank gate binds, not the floor"
    else:
        # The floor that admits exactly `budget` of the eligible frames. Taking the value
        # BELOW the cut rather than at it keeps `hazard >= floor` from admitting one extra
        # through a tie.
        floor = float(eligible[budget])
        note = "solved"

    admitted = sum(1 for h in eligible if h >= floor)
    quiet = 1.0 / (1.0 + pow(2.718281828459045, -K.FUSION_BIAS))

    print(f"arm                        : confuse, {len(windows)} windows x {len(hz[0])} frames")
    print(f"frames total               : {n_frames}")
    print(f"structurally eligible      : {len(eligible)}  (rank <= {K.ESCALATION_TOP_K}) "
          f"= {ceiling * 100:.2f}% ceiling")
    print(f"target fraction            : {args.target * 100:.2f}%  ({budget} frames)")
    print(f"eligible hazard range      : [{min(eligible):.4f}, {max(eligible):.4f}]")
    print(f"quiet-camera hazard        : {quiet:.4f}  (sigmoid(FUSION_BIAS))")
    print(f"\nSOLVED ESCALATION_HAZARD_MIN = {floor:.6f}   ({note})")
    print(f"  admits {admitted} frames = {admitted / n_frames * 100:.2f}% of the arm")
    if floor <= quiet:
        print("\n  WARNING: the solved floor is at or below a quiet camera's hazard. The "
              "gate\n  would not be gating. Raise the target or re-examine the fusion.")

    payload = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": bench.git_commit(root),
        "method": "rank-eligible hazard quantile on CONFUSE, escalation disabled",
        "target_fraction": args.target,
        "n_frames": n_frames,
        "n_eligible": len(eligible),
        "structural_ceiling": ceiling,
        "solved_floor": floor,
        "realised_fraction": admitted / n_frames,
        "quiet_camera_hazard": quiet,
        "fusion_bias": K.FUSION_BIAS,
    }
    out = root / "results" / f"escalation_floor_{dt.date.today()}.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwritten -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

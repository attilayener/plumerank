#!/usr/bin/env python3
"""Class-blind box proposal for the S3 separability gate.

WHY THIS EXISTS, AND WHAT IT CHANGES
------------------------------------
``results/gonogo_2026-09-01.md`` §5 specified 120 boxes drawn by hand in ``box.html``, and
declared their non-blindness as a limitation: "boxes are hand-drawn by one person who knows
which class each image belongs to." Those boxes were never drawn -- ``spike/boxes.json`` does
not exist -- so the gate has never run and the project's premise is unmeasured.

This proposes the boxes mechanically instead. It is **not** a like-for-like substitute and
the difference cuts both ways:

  BETTER  The proposer cannot see the label. §5's non-blindness limitation is gone, not
          mitigated -- the same code, the same constants and the same threshold run on a
          smoke frame and a fog frame, and nothing in this file branches on ``class``.
          The boxes are also reproducible by anyone with the repo, which hand boxes are not.

  WORSE   The proposer becomes part of what is being measured. If it localises smoke well
          and fog badly, a high AUC could mean "we can find plumes and not fog banks"
          rather than "S3 separates them". That confound does not exist with hand boxes.

The ``centre`` mode exists to bound exactly that confound, and it is the reason to trust or
distrust the ``motion`` number -- see ``--mode``.

THE PROPOSAL, WHICH IS THE PRODUCTION PATH AND NOT A NEW ONE
------------------------------------------------------------
``motion`` runs the project's own ``plumerank.stabilise.stabilise`` at the project's own
WORK_WIDTH x WORK_HEIGHT, differences the registered pair, and takes the largest connected
component of the thresholded difference. That is what ``segment.py`` does in production, so
the boxes the gate scores are the kind of region the deployed system actually hands to the
statistic -- rather than the kind a human would draw knowing what they were looking for.

Every constant below is fixed a priori and none is fitted. They are stated as module
constants rather than buried in the code so that a reader can check that claim.
"""

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from plumerank import constants as K  # noqa: E402
from plumerank.stabilise import stabilise  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FRAMES = HERE / "frames"
MANIFEST = HERE / "manifest.js"

# --- constants, fixed a priori, none fitted -------------------------------------------

#: The proposal runs at the pipeline's own working resolution, so the component the gate
#: scores is the component the deployed segmenter would have produced.
PROPOSE_W, PROPOSE_H = K.WORK_WIDTH, K.WORK_HEIGHT

#: Smoothing before thresholding. One sigma, not a scale search: a search would be a fitted
#: parameter, and there is no held-out set here to fit it on.
DIFF_BLUR_SIGMA = 2.0

#: The difference threshold is the Nth percentile OF EACH IMAGE'S OWN difference
#: distribution -- adaptive to the frame, identical in its logic across classes. A fixed
#: grey level would silently favour whichever class happens to be brighter.
DIFF_PERCENTILE = 99.0

#: Below this a component is noise, in working pixels.
MIN_COMPONENT_AREA = 12

#: Pad the component so the boundary band has ground to sit on: S3 measures a ring OUTSIDE
#: the interior, and a box drawn tight to the blob has no ring.
PAD_FRACTION = 0.35

#: Minimum box side in FULL-RESOLUTION pixels. s3() needs only ~10 px to place its band, but
#: a 10 px crop of a 3072 px frame is noise wearing a number.
MIN_SIDE_FULL = 96

#: The degenerate control: a fixed fraction of the frame, centred, identical in every image.
CENTRE_FRACTION = 0.40


def load_manifest():
    """``manifest.js`` is a JS assignment wrapping a JSON array; take the array."""
    text = MANIFEST.read_text(encoding="utf-8")
    return json.loads(text[text.index("[") : text.rindex("]") + 1])


def _pad_and_clamp(x, y, w, h, full_w, full_h, scale_x, scale_y):
    """Component bbox in working pixels -> padded box in full-resolution pixels."""
    x, y = x * scale_x, y * scale_y
    w, h = w * scale_x, h * scale_y
    px, py = PAD_FRACTION * w, PAD_FRACTION * h
    x, y, w, h = x - px, y - py, w + 2 * px, h + 2 * py
    # Grow -- never shrink -- to the minimum side, about the same centre.
    cx, cy = x + w / 2.0, y + h / 2.0
    w, h = max(w, MIN_SIDE_FULL), max(h, MIN_SIDE_FULL)
    x, y = cx - w / 2.0, cy - h / 2.0
    # Clamp into the frame, preserving size where the frame allows it.
    w, h = min(w, full_w), min(h, full_h)
    x = min(max(0.0, x), full_w - w)
    y = min(max(0.0, y), full_h - h)
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def propose_motion(rec):
    """Largest changed component between this frame and its predecessor.

    Returns ``None`` when there is no predecessor or no component clears
    ``MIN_COMPONENT_AREA``. A refusal is recorded and counted; it is never a silent
    fallback to some other box, because a fallback box would enter the AUC as though the
    proposer had found something.
    """
    if not rec.get("prev"):
        return None, "no predecessor frame"
    cur = cv2.imread(str(FRAMES / rec["file"]), cv2.IMREAD_GRAYSCALE)
    prev = cv2.imread(str(FRAMES / rec["prev"]), cv2.IMREAD_GRAYSCALE)
    if cur is None or prev is None:
        return None, "frame unreadable"
    if cur.shape != prev.shape:
        return None, "frame pair differs in shape"

    full_h, full_w = cur.shape
    cur_s = cv2.resize(cur, (PROPOSE_W, PROPOSE_H), interpolation=cv2.INTER_AREA)
    prev_s = cv2.resize(prev, (PROPOSE_W, PROPOSE_H), interpolation=cv2.INTER_AREA)

    reg = stabilise(prev_s, cur_s)
    diff = cv2.absdiff(reg.frame, prev_s).astype(np.float32)
    diff = cv2.GaussianBlur(diff, (0, 0), DIFF_BLUR_SIGMA)

    cut = float(np.percentile(diff, DIFF_PERCENTILE))
    mask = (diff > cut).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, "no connected component above threshold"
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[idx, cv2.CC_STAT_AREA]) < MIN_COMPONENT_AREA:
        return None, "largest component below MIN_COMPONENT_AREA"

    box = _pad_and_clamp(
        int(stats[idx, cv2.CC_STAT_LEFT]),
        int(stats[idx, cv2.CC_STAT_TOP]),
        int(stats[idx, cv2.CC_STAT_WIDTH]),
        int(stats[idx, cv2.CC_STAT_HEIGHT]),
        full_w,
        full_h,
        full_w / float(PROPOSE_W),
        full_h / float(PROPOSE_H),
    )
    return box, f"stabilise={reg.method} response={reg.response:.3f}"


def propose_centre(rec):
    """The control. A fixed central box, identical in every frame, blind by construction.

    If ``motion``'s AUC is no better than this, the motion proposer is contributing nothing
    and the gate is measuring S3 on an arbitrary crop. If ``motion`` is much better, the
    localisation is doing work -- and that work is then a confound to declare, because part
    of the separation lives in the proposer rather than in the statistic.
    """
    img = cv2.imread(str(FRAMES / rec["file"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, "frame unreadable"
    h, w = img.shape
    bw, bh = int(CENTRE_FRACTION * w), int(CENTRE_FRACTION * h)
    return ((w - bw) // 2, (h - bh) // 2, bw, bh), "fixed central box"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["motion", "centre"], default="motion")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "boxes_motion_proposed.json")
    args = ap.parse_args()

    propose = {"motion": propose_motion, "centre": propose_centre}[args.mode]
    records, refusals = [], []
    for rec in load_manifest():
        box, why = propose(rec)
        if box is None:
            refusals.append((rec["file"], rec["class"], why))
            continue
        x, y, w, h = box
        records.append(
            {
                "file": rec["file"],
                "class": rec["class"],
                "prev": rec.get("prev"),
                "x": x, "y": y, "w": w, "h": h,
                "proposer": args.mode,
                "note": why,
            }
        )

    args.out.write_text(json.dumps(records, indent=1, sort_keys=True), encoding="utf-8")
    kept = {c: sum(1 for r in records if r["class"] == c) for c in ("smoke", "fog")}
    print(f"{args.mode}: {len(records)} boxes -> {args.out.name}  "
          f"(smoke={kept['smoke']} fog={kept['fog']})")
    if refusals:
        by_class = {c: sum(1 for _, k, _ in refusals if k == c) for c in ("smoke", "fog")}
        print(f"  refused {len(refusals)} (smoke={by_class['smoke']} fog={by_class['fog']}):")
        for f, c, why in refusals:
            print(f"    {c:<6} {f:<28} {why}")


if __name__ == "__main__":
    main()

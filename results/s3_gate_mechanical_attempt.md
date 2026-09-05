# The S3 gate cannot be run mechanically — an attempt, and why it does not substitute

**Date:** 2026-09-06 · **Author:** automated session · **Status: INCONCLUSIVE. NOT a verdict.**

> **Read this first.** This document reports two NO-GO numbers and then argues that **neither
> of them is a verdict on S3**, and that §6 of `gonogo_2026-09-01.md` must stay `_pending_`
> until a human draws the boxes. Nothing here may be quoted as a gate result.

---

## 1. What was attempted and why

`spike/boxes.json` did not exist, so the Sep 1 separability gate had never run and the
project's premise was unmeasured five days after its own deadline. The gate's instrument
(`spike/s3_separation.py`), its 120 frames and its decision rule were all committed and ready;
the only missing input was 120 hand-drawn boxes.

`spike/propose_boxes.py` was written to supply them mechanically, in two modes, neither of
which can see the class label:

- **`motion`** — the production path: `plumerank.stabilise.stabilise` at the pipeline's own
  480×270 working resolution, frame differencing against the predecessor, a per-image
  percentile threshold, and the largest connected component, padded. This is what
  `segment.py` does in deployment.
- **`centre`** — a degenerate control: a fixed central box covering 40% of the frame,
  identical in every image. It exists to bound how much of any separation lives in the
  localiser rather than in the statistic.

## 2. What they returned

Every variant, both box sources, `--all-variants` (v0, v1, v2b, v3, v4, v5):

| boxes | n (smoke/fog) | best variant | AUC | 95% CI | rule says |
|---|---|---|---:|---|---|
| `motion` | 59 / 38 | v1 | 0.514 | [0.401, 0.633] | NO-GO |
| `centre` | 60 / 60 | v1 | 0.544 | [0.439, 0.651] | NO-GO |

The highest CI upper bound anywhere in the twelve runs is **0.651**, below the 0.70 cutline,
so §2's NO-GO branch fires on the most favourable number available — and that number is
already optimistically biased by a max-of-six selection. Four of the twelve point estimates
sit **below 0.5**, meaning fog scored higher than smoke. Overlap was near-total: 96 of 97
boxes for `motion` v0, 120 of 120 for `centre` v0.

Taken at face value that is an unambiguous NO-GO and the end of the project.

## 3. It must not be taken at face value: the boxes are on the wrong things

Both box sets were rendered onto their frames and inspected. **The proposer is not finding the
phenomenon.**

On smoke frames, of the first six: one box is on the plume. The other five are on **a drifting
cumulus cloud**, **empty sky above the ridge the plume is actually on**, **a building façade**,
**empty sky**, and **a white vehicle**. On fog frames the boxes are worse and they are
systematically worse in one direction: they land on **sun glare and the sky brightening at
dawn**, because over a ten-minute gap at 06:00–08:59 Pacific the largest change in the frame is
the sun coming up, not the marine layer moving.

The mechanism is plain in hindsight. FIgLib publishes the smoke frames at t+15/+25/+35 min, so
consecutive frames are ~10 minutes apart. Over ten minutes a cumulus translates hundreds of
pixels and the sun climbs measurably, while a plume — which is *anchored*, the project's whole
S1 premise — moves comparatively little. A largest-change detector on a ten-minute baseline is
therefore close to a **cloud-and-sunrise detector**, and it will reliably prefer them to the
thing the gate is asking about.

So the AUCs in §2 are a measurement of `propose_boxes.py`. They say almost nothing about S3.

## 4. Why the control does not rescue it either

`centre` agreeing with `motion` initially looks like corroboration — two independent instruments,
one conclusion. It is not. A fixed box covering 40% of a 3072×2048 frame contains sky, horizon,
terrain and buildings in both classes; its S3 is a whole-scene texture statistic. Two instruments
that both fail to isolate the phenomenon agreeing that they cannot separate it is one finding,
not two.

**A NO-GO requires evidence that S3 fails on regions that contain the phenomenon. Neither box
set contains it reliably, so neither can produce that evidence.**

## 5. What this does establish

1. **The gate's shortcut is closed.** Mechanical box proposal was the only route to the Sep 1
   verdict that did not cost human time, and it does not work. §5 of the gate document
   specified hand-drawn boxes; that specification turns out to be load-bearing rather than
   incidental, because *localising the phenomenon is the hard part and the gate exists to
   factor it out.*
2. **The instrument downstream of the boxes is verified end to end.** `s3_separation.py` runs,
   scores 120 boxes, bootstraps, applies the rule and prints a verdict. When the real boxes
   arrive the gate returns its answer immediately; nothing else is missing.
3. **A concern about the fog class, noticed while looking.** Several frames drawn as
   `fog` show a clear dawn over hills with no visible marine-layer boundary, and the dominant
   feature is the rising sun. §5 says the fog set was "visually confirmed as genuine marine
   layer"; on this inspection at least some frames do not obviously satisfy that. **A boundary
   statistic measured on a frame with no boundary is noise entering the gate as data.** Whoever
   draws the boxes should mark such frames unusable rather than box something, and the count
   should be published.

## 6. What is required, and it is about an hour of human time

`spike/box.html` is committed and is the tool for it: open it, draw one box per frame over the
120 in `spike/frames/`, save `boxes.json` beside them, then run

```
python3 spike/s3_separation.py --all-variants
```

and paste the output into §6 of `gonogo_2026-09-01.md`, whatever it says. That hour is the
single highest-value hour available to this project, because every number downstream of it —
including the fitted D11 re-run — is conditional on a premise that remains unmeasured.

**§6 stays `_pending_`. The verdict is still owed.**

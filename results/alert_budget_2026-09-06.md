# The alert budget is not where the detections are lost, and here is the ceiling that proves it

**Date:** 2026-09-06 · **Status: a measurement, and a negative result that closes a direction.**
Every number below is at the **shipped** operating point — 9/29, threshold 0.740112
(`results/arm_dev_2026-09-06.json`) — reproduced frame-for-frame by `spike/why_missed.py`
before anything was attributed. Read
`results/artefact_provenance_2026-09-06.md` first if the number 41.4% is in your head; it is
an ablation cell's.

---

## 1. Every miss, by cause

| cause | sequences |
|---|---:|
| **DETECTED** | 9 |
| **hazard never crosses the threshold after ignition** | **15** |
| budget spent before ignition | 5 |

Nothing lands in "crosses but never in the top 4" — ranking against the other 28 cameras is not
what loses a detection here.

The 5 budget-exhausted sequences are the interesting ones, because they are the only misses
where the system *did* score the fire above its own alert line and still said nothing: the
camera's two slots for the night were gone before the fire started.

## 2. The tempting fix, measured, and it is not enough

The two slots are spent on the night's **first two crossings**. Nothing about that was ever
decided — it is what a counter does. So four causal alternatives were run at the same fixed
threshold. (Nothing here moves the calibration: `rank.realised_rate` counts every frame at or
above the threshold, uncapped and budget-blind, so a slot policy changes which alerts are
emitted and leaves the published rate exactly where it was.)

| policy | detected | alerts spent | mean TTD |
|---|---:|---:|---:|
| **shipped** — first two crossings | **9 / 29 = 31.0%** | 25 | 21.3 min |
| rising edge — one slot per excursion | 9 / 29 = 31.0% | 25 | 21.3 |
| refractory 10 frames (10 min) | 9 / 29 = 31.0% | 23 | 21.3 |
| refractory 20 frames (20 min) | **11 / 29 = 37.9%** | 21 | 20.5 |
| second slot needs +0.02 hazard | 10 / 29 = 34.5% | 19 | 19.6 |
| second slot needs +0.05 hazard | 8 / 29 = 27.6% | 17 | 23.6 |
| **ORACLE — the night's two highest hazards** | **12 / 29 = 41.4%** | 25 | 18.6 |

The oracle is not a policy. It picks the two best frames of the night knowing the whole night
in advance, and it is here for one reason: **it is the ceiling.** No causal rule for spending
two slots can beat it.

**That ceiling is 12 of 29. The median still does not exist at the ceiling.** The most the
alert-budget policy is worth on this corpus is three sequences, and the headline metric is not
among the things it can buy.

The refractory sweep says the same thing less kindly: across R = 18…44 frames the detection
count wanders 11, 11, 11, 11, 12, 12, 13, 11, 12, 11, 11, 11, 11 — a two-sequence band with no
plateau. Picking the 13 would be picking noise, and `s4_ablation_2026-09-06.md` §3 has already
established what a two-sequence swing is worth on 29 windows: nothing.

## 3. Why the ceiling is so low — the sharpest statement of the real defect

> **In 11 of the 29 DEV sequences, the camera's highest hazard of the night happens BEFORE the
> fire starts.** Median excess of the pre-ignition peak over the post-ignition peak: 0.0515.

All 11 are misses. **None of the 9 detected sequences has this property.**

For those 11 there is no threshold that admits the fire and rejects the same camera's own
fire-free minutes, because the fire-free minutes score higher. No budget policy reaches them.
No ranking reaches them — they lose to themselves, not to another camera. They are recoverable
only by a model that scores a plume above a quiet dawn on the same camera, which is exactly the
discrimination `statistics_confound_2026-09-06.md` measured as near-absent:

> *"Restricted to the 1804 frames that actually have a component, the statistics collapse …
> s3 0.4396 — below chance, actively anti-predictive."*

This document is the same finding arriving from the other end. §4b of that document said effort
on the segmenter will not move the detection rate; this says effort on the alert policy will
not either, and puts a number on it — **three sequences, and no median.**

## 4. What this does not say

- **Not that the slot policy is fine.** The refractory rule reaches 11/29 while spending 21
  alerts instead of 25 — more detections for fewer interruptions. If it is adopted it should be
  adopted for that, and *not* quoted as progress on the headline.
- **Not that the corpus is too small to say anything.** The 11-of-29 inversion in §3 is not a
  two-sequence swing; it is a property of more than a third of the arm, and it has a clean split
  against detection (11 of 20 misses, 0 of 9 detections).
- **Not a verdict on S3.** The Sep 1 gate on hand-drawn boxes at working resolution, on a fog
  class that contains fog, is still the only measurement designed to answer that, and it is
  still unrun.

## 5. What follows, and what is not an agent's call

1. **Stop looking at the budget for the headline.** The ceiling is measured; the direction is
   closed.
2. **Adopting the refractory rule is a design decision about what an alert means** — is a
   second alert on the same camera ten minutes later a new event or a repeat? It changes shipped
   behaviour in `plumerank/rank.py` and it is not implemented here.
3. **The lever is discrimination on the same camera.** The concrete target is now stated as a
   number a change has to move: the 11 sequences whose quiet minutes outrank their own fire.

## 6. Reproduce

```
python3 spike/why_missed.py             # the attribution -- must print 9/29 = 31.0%
python3 spike/why_missed.py --policies  # the table in section 2
python3 spike/why_missed.py --sweep     # the refractory sweep
```

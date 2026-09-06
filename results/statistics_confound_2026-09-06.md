# The four statistics add almost nothing over "a component exists"

**Date:** 2026-09-06 · **Status: a measurement, reproducible from committed files.**
**This document retracts a claim made earlier the same day.** See §5.

---

## 1. The measurement

`results/features_dev.jsonl` — 2291 frames over 29 DEV sequences, 1160 positive by the ignition
clock. Reproduce with `python3 scripts/fit_fusion.py`, which now runs this control on every fit.

| feature | AUC alone (in-sample) |
|---|---:|
| **component presence** (is there a component at all?) | **0.6551** |
| s3 | 0.6192 |
| s4 | 0.5830 |
| s1 | 0.5660 |
| s2 | 0.5304 |
| escalation score | 0.5119 |
| **the whole fused model** | **0.6566** |

**The fused model beats component presence alone by 0.0015.** Four statistics, a learned tile
classifier and five fitted weights are worth roughly nothing over a single bit.

## 2. Why every statistic looked informative when none was

`Stats(s1=0, s2=0, s3=0, s4=0)` is what the pipeline records when the segmenter finds no
component. And component presence is strongly label-correlated:

| | frames with **no** component |
|---|---:|
| positive (at or after ignition) | 69 of 1160 — **5.9%** |
| negative (before ignition) | 418 of 1131 — **37.0%** |

So a zero in any statistic is mostly a report that nothing was found, and every statistic
inherits presence's discriminating power for free. **Restricted to the 1804 frames that
actually have a component, the statistics collapse:**

| statistic | AUC on frames where it is defined |
|---|---:|
| s4 | 0.5382 |
| s1 | 0.5280 |
| s2 | 0.5017 — chance |
| **s3** | **0.4396 — below chance, actively anti-predictive** |

## 3. The consequence for the fitted weights

A logistic given five features that all encode presence hands its largest weight to whichever
one encodes it best. Give presence its own column and the fit changes character:

| | as shipped | presence explicit |
|---|---:|---:|
| s1 | +0.6164 | +0.3920 |
| s2 | +0.3856 | **−0.0411** |
| **s3** | **+4.0931** | **−2.2369** |
| s4 | +0.9335 | +0.2462 |
| escalation | +0.7785 | +1.1390 |
| has_component | — | **+2.4028** |
| **LOSO AUC** | 0.6387 | **0.6626** |

**S3 changes sign, and the model gets better.** A weight that flips under a control was
measuring the control.

## 4. What this does and does not say

**Does not say the project is dead.** Out of sample the statistics do add something: LOSO rises
from 0.5610 (presence alone) to 0.6626 (all features, presence explicit). That gap is real and
it is where the primitive lives. But it is a much smaller claim than "S3 separates smoke from
fog", and the direction of S3's contribution is now *against* the premise.

**Does say the premise is unmeasured and looking weaker.** This is the fourth independent
observation pointing the same way, and the first on real data with a proper control:

1. The Sep 1 gate at native resolution: AUC < 0.5 on four of six variants.
2. The synthetic fixture: cloud raw_s3 above fire, so the fitted model ranks a translating
   hard-edged circle above an anchored growing plume.
3. `cloud_shadow` (3.86) and `low_cloud` (4.25) dominating the alert budget while fog sits at
   1.15 — S3 failing hardest against the class it was predicted to reject most cleanly.
4. **This**: s3 AUC 0.4396 on the frames where it is defined.

It also explains the 41% detection rate directly. A system whose discrimination is
substantially "something moved" is doing what B2 frame-differencing does, and the margin over
B2 comes from ranking and budget calibration rather than from the plume primitive.

## 4b. Where the 41% detection rate is actually lost — and it is not segmentation

Component presence by frame index, over all 29 DEV sequences (t0 = 39 after unification):

| frames | has a component |
|---|---:|
| 0–9 (MOG2 warm-up) | 57.9% |
| 10–38 (pre-ignition, warm) | 64.8% |
| **39+ (post-ignition)** | **94.1%** |

Presence climbs smoothly from ~65% before ignition to **100% by frame 52 — thirteen minutes
after t0.** That is not a warm-up artefact; it is the plume becoming reliably segmentable. The
segmenter is doing real work, and presence being label-correlated is a genuine signal rather
than a bug. (One real artefact: frame 0 sits at 100% because MOG2's first frame flags the whole
image as foreground.)

Which makes the arithmetic below the finding that matters:

> **25 of 29 sequences carry a segmented component on more than 80% of their post-ignition
> frames. Only 12 of 29 are ever detected.**

**CORRECTION (same day):** the detection count here is the **S4-OFF ablation cell's**. The
shipped configuration detects **9 of 29**, so the gap this paragraph names is wider than
written: of the 25 sequences that carry a component on more than 80% of their post-ignition
frames, 8 are detected and **17 never alert** — not thirteen. (One detected sequence sits
outside the 25.) The presence arithmetic is measured from `features_dev.jsonl` and is unaffected.
See `results/artefact_provenance_2026-09-06.md`.

**Seventeen sequences contain a visible, segmented, tracked plume that never becomes an alert**
(thirteen as this line originally read, at the S4-OFF cell's detection count).
The loss is not in stabilisation or MOG2 — it is downstream, in the hazard fusion and the
ranking/threshold stage. And §3 explains why that stage fails: the statistics it fuses carry
almost no information beyond the presence bit, so it has nothing with which to promote a real
plume above a cloud that is also moving.

This is the single most actionable result in this document. **Effort spent on the segmenter
will not move the detection rate; effort spent on what distinguishes a plume from a moving
cloud will.**

## 5. Retraction

Earlier on 2026-09-06, `plumerank/constants.py` and `results/d11_fitted_2026-09-06.md` both
recorded that the fit **falsified** `gonogo §7.4`'s second reading — that unfitted weights
over-weighted S3 — on the grounds that DEV supported *more* weight on S3, not less.

**That was wrong.** The inference was drawn from the uncontrolled fit, where S3's weight was
mostly a presence proxy. Controlled, S3 is negative. **§7.4's reading 2 is supported.** Both
documents now carry the retraction, and the control runs automatically on every fit so the
error cannot be repeated silently.

## 6. What to do

1. **Do not quote +4.0931 as evidence about S3.** It is not a measurement of boundary
   diffuseness. `check_claims.py` cannot catch this one; only reading can.
2. **Decide whether the fusion carries a presence term.** It is worth +0.024 LOSO and it makes
   the other weights interpretable — but it changes what the UI's per-row S1/S2/S3 chips mean,
   so it is a design decision, not an agent's. The uncontrolled weights still ship pending it.
3. **Fix the segmenter before re-reading anything into the statistics.** 37% of pre-ignition
   frames producing no component at all is the largest single signal in this corpus, and it is
   a property of MOG2's warm-up and thresholds rather than of smoke.
4. **Run the Sep 1 gate.** Nothing here replaces it. Hand-drawn boxes isolate the statistic from
   the segmenter, which is exactly the confound this document is about, and it remains the only
   clean measurement of S3 the project has ever designed.

# S3 runs backwards on the project's own fixture — three observations that agree

**Date:** 2026-09-06 · **Status: a finding about the premise. Not yet a verdict.**

---

## 1. What S3 claims

`plumerank/statistic.py`:

> *"Smoke is soft-edged and turbulent inside: high interior Laplacian energy against low Sobel
> energy in the band around it. A cloud shadow, a fog bank's leading edge or a lens artefact
> **inverts that ratio.** This is the only one of the four statistics that separates smoke from
> fog, which is why it carries the largest fusion weight."*

So: **smoke high, cloud low.** The whole project rests on this. It is the statistic the Sep 1
gate exists to test, and after fitting it carries a weight of 4.09 against 0.62 for S1.

## 2. Three independent observations, none of which agree with it

**(a) The synthetic fixture — `tests/test_bench.py`, built to embody the theory.**
`_fire_window` draws an anchored growing ellipse and blurs it ("a soft edge, which is S3's
claim"). `_cloud_window` draws a hard-edged translating circle and does not blur it. Measured
through the real `Segmenter` → `Tracker` → `measure` path:

| window | raw_s3 median | range |
|---|---:|---|
| synthetic FIRE (soft edge, by construction) | **0.0798** | 0.0626 – 0.4202 |
| synthetic CLOUD (hard edge, by construction) | **0.1733** | 0.0689 – 0.7643 |

**The cloud scores 2.2× the fire on the statistic built to tell them apart.** At the best frame
of each, the fitted model ranks them `cloud 0.7459 > fire 0.7300` — the false alarm outranks
the fire. Under the a-priori weights it did not (`fire 0.5019 > cloud 0.1117`), because S1 at
1.9 carried the decision and S1 is 1.000 for the fire and 0.000 for the cloud. **Fitting did
not create this; it revealed it, by moving weight onto the statistic that was already
inverted.**

**(b) The real spike frames, at native resolution.** All six pre-enumerated variants of the
Sep 1 gate returned AUC ≤ 0.514, four of them **below 0.5** — fog scoring more smoke-like than
smoke (`s3_gate_mechanical_attempt.md` §2, §4b).

**(c) The real CONFUSE arm.** `cloud_shadow` is the worst adversary at 4.00 alerts/camera-night
— the class `seed-data.md` predicted S3 "should reject most cleanly", because its boundary is
hard and that is exactly what S3 measures (`d11_fitted_2026-09-06.md` §4).

## 3. The one observation that points the other way, and it matters

At the pipeline's **working resolution** on the same real spike frames, every variant moved
above 0.5 and the medians ordered correctly (smoke 0.168 vs fog 0.145, v0). So S3 is not
uniformly inverted — it behaves as designed on real frames at the scale production uses, and
backwards on the synthetic fixture and on real frames at native scale.

**The honest summary: S3's direction is not robust.** It depends on resolution and on what the
region contains. A statistic carrying a fusion weight of 4.09 whose *sign* moves with scale is
not yet understood, and the project has never measured it on hand-boxed regions at working
resolution — which is precisely what the Sep 1 gate was for.

## 4. Why the synthetic result is not dismissible as "just a caricature"

It partly is: a blurred ellipse on a blurred background is not smoke, and the blur is applied
to the whole frame rather than to the plume, so the fixture may not produce a genuinely soft
boundary against a sharp background at all. That is a real criticism of the fixture and it
should be fixed.

But two things survive it:

1. **It is the fixture the project's own tests trust.** Three tests in
   `tests/test_results_contract.py` currently fail because the fitted model does not detect
   this fire. Whatever the fixture is, the repository treats it as ground truth for "the
   pipeline works", and it no longer agrees with the shipped constants.
2. **It reproduces (b) and (c) on data where the answer is known by construction.** Two real-
   data observations and one synthetic one, pointing the same way, is not a coincidence about
   caricatures.

## 5. What to do, in order — and none of it is "adjust S3"

1. **Run the Sep 1 gate properly**: hand-drawn boxes, at working resolution, on a fog class
   that contains fog. That is the measurement that decides whether S3 works, and all three
   preconditions are now known and none is expensive. See `s3_gate_mechanical_attempt.md` §6
   and `fog_class_review.md` §6.
2. **Fix the synthetic plume** so its boundary is soft *relative to its surroundings* — blur
   the plume before compositing, not the whole frame afterwards — and re-measure. If the sign
   corrects, the fixture was the problem. If it does not, S3 is.
3. **Do not retune S3 to make any of this come out right.** Its parameters (`S3_BAND_FRACTION`,
   `S3_BAND_MIN_PX`) are a priori and the gate's whole authority comes from their being fixed
   before the measurement. Changing them now converts the gate into a search.

## 6. Reproduce

```
python3 - <<'PY'   # §2(a), the synthetic comparison
import sys, cv2; sys.path[:0]=['tests','.']
import test_bench as fx
from plumerank.segment import Segmenter
from plumerank.statistic import Tracker, measure
...
PY
python3 spike/s3_separation.py --all-variants --scale work   # §3, needs boxes.json
python3 scripts/class_split.py                               # §2(c)
```

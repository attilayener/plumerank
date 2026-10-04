# The fog class is mostly not fog — a visual census of `spike/urls_fog.txt`

**Date:** 2026-09-06 · **Method:** every frame rendered and inspected; 54 of 60 seen.
**Status: a finding that needs human confirmation, not a decision.**

---

## 1. Why anyone looked

While checking whether the mechanical box proposer had landed on the right regions
(`s3_gate_mechanical_attempt.md` §3), the fog frames themselves looked wrong. So all 60 were
rendered as contact sheets and inspected. 54 were reviewed; the census below is of those.

## 2. What the class is supposed to be

`gonogo_2026-09-01.md` §5: *"Fog: 60 frames from ~20 dawn mornings at 06:00–08:59 Pacific …
selected by a within-camera fog score, then **visually confirmed as genuine marine layer**."*

The gate's whole question is whether S3 separates smoke's soft edge from **a fog bank's**. The
negative class must therefore contain fog banks. And S3 measures a *boundary* — interior
Laplacian energy against Sobel energy in a band around the region — so a frame with no visible
boundary contributes noise rather than a negative example.

## 3. What is actually in it

| category | count (of 54 inspected) | what it looks like |
|---|---:|---|
| **Clear marine layer** — undercast sea of cloud below the ridge, or the camera immersed in dense fog | **~8** | frames 12, 13, 14, 38, 42, 47, 50, 53 |
| **Overcast ceiling** — a grey stratus deck filling the upper frame, no boundary visible in shot | ~9 | 21, 23, 24, 26, 27, 31, 48, 51, 52 |
| **No marine layer at all** — clear or lightly-clouded dawn over hills; the dominant feature is the rising sun, sky gradient, cirrus or contrails | **~27** | 7, 8, 15–17, 19, 20, 22, 25, 28–30, 32–37, 39–41, 43–46, 49, 54 |

**Roughly half the negative class is a clear dawn sky.** Fewer than one frame in six shows the
adversary the project was designed against.

### Method, and its limits — this is a judgement, not a measurement
The classification above is a visual call made from downscaled contact sheets by a
non-meteorologist, in one pass, knowing that every frame was labelled `fog`. The
middle category is the soft one: a stratus deck that has closed over the camera *is* the marine
layer, meteorologically, and someone who selected these frames may have meant exactly that. The
first and third categories are much harder to argue with. **Before anything is done about this,
a human who knows the HPWREN cameras should re-walk the sheets.**

## 4. Why this is worse than it looks: it has already propagated

`gonogo_2026-09-01.md` §7.5 recorded — without an explanation for it — that the confusion set's
original fog predicate admitted **0 of these 60 frames**: `edge_energy` measured 12.6–60.1
against a `<= 18.0` gate, and `sky_ground_ratio` 1.46–11.27 against a `<= 1.05` gate with the
sign inverted.

**That is exactly what a contaminated class would produce.** Cutlines written a priori for a
marine layer should reject clear dawn sky, and they did — all 60 times.

The response recorded in §7.5 was to **re-derive the predicate from these same 60 frames**,
setting each cutline to the median of the labelled class (`deck_fraction` 0.12847, AUC 0.706;
`sky_greyness` -0.13140, AUC 0.674). So:

> The `fog_marine` class of the CONFUSE arm — the 13 windows whose 1.5385 alerts/camera-night
> came within a hair of aborting the project on 2026-09-05 — is selected by a predicate
> calibrated on a set that is roughly half clear sky.

That does not make the 1.5385 wrong. It makes it a measurement of something whose name may not
be "fog", and it means the D11 criterion may not have been evaluated against the adversary it
was written for.

## 5. What this does NOT establish

- **Not that the project is unsound.** A weak fog class makes the gate *uninformative*, not
  failed. It is equally consistent with S3 being fine and never having been tested.
- **Not a reason to relabel anything.** Re-deriving a class definition from a set that turned
  out to be contaminated is what produced this situation once already. The fix is new frames,
  not a new rule over the old ones.
- **Not an abort trigger.** `ruling_d11_contribute.md` §4 fixed what binds at D11, and this is
  not on that list.

## 6. What is worth doing, in order

1. **Have a human re-walk the 60 frames** and mark each `marine_layer` / `overcast` / `neither`.
   An hour at most, and it decides everything below.
2. **If the census holds, rebuild the fog class from frames that contain a fog bank.** The
   original a-priori predicate (`edge_energy <= 18.0`, `sky_ground_ratio <= 1.05`) is the honest
   starting point — it was written before anyone saw data, and its rejection of all 60 frames
   now reads as the predicate working rather than failing. **Restore it before re-deriving
   anything.**
3. **Only then draw the gate's boxes.** Boxing the current set would spend the hour producing a
   confident AUC about clear dawn sky. The box work and the class work are the same hour if
   done in the right order and two hours if not.
4. **Re-run the CONFUSE arm afterwards**, because `fog_marine`'s membership will change.

## 7. The one piece of good news

Every artefact needed to redo this is committed and cheap: `spike/urls_fog.txt` is a URL list,
`scripts/build_confusion_set.py` regenerates the arm from a predicate, and the HPWREN
medium-term archive is still open. This is a data problem found on **2026-09-06, with 50 days
left**, and it is recoverable. Found at the held-out run it would not have been.

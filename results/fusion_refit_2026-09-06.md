# Refitted on the corrected dump: only one weight moves, and it changes sign

**Date:** 2026-09-06 · **Status: a measurement. No constant has been changed.**
`results/artefact_provenance_2026-09-06.md` §6 found that `results/features_dev.jsonl` carried
**94** escalated frames while the shipped pipeline produces **51**, and that
`scripts/fit_fusion.py` reads that dump — so the shipped `FUSION_W_ESCALATION` was fitted
against an escalation column no committed benchmark ever produced. The dump has been
regenerated and the fit re-run. Raw: `results/fusion_refit_2026-09-06.json`.

---

## 1. Old fit against new

| weight | shipped (`constants.py`) | refit on the corrected dump | change |
|---|---:|---:|---|
| `FUSION_BIAS` | −0.6961 | −0.7014 | — |
| `FUSION_W_S1` | +0.6164 | +0.6622 | — |
| `FUSION_W_S2` | +0.3856 | +0.4348 | — |
| `FUSION_W_S3` | +4.0931 | +4.0984 | — |
| `FUSION_W_S4` | +0.9335 | +0.9413 | — |
| **`FUSION_W_ESCALATION`** | **+0.7785** | **−0.3838** | **sign flip** |
| LOSO AUC | 0.6387 | 0.6393 | — |

Exactly what §6 predicted: the S1–S4 weights are a property of the classical statistics and
are untouched by the escalation column; the escalation weight is not, and it is the one that
moves.

## 2. What the sign means

`fit_fusion.py` prints its own verdict, and it is the strongest wording that script has:

> **FINDING: esc fitted NEGATIVE.** The spec asserts every statistic is evidence FOR fire, so
> a negative weight means the data contradict the premise for that statistic on DEV. Publish
> it; do not clip it to zero.

On DEV, at the shipped operating point, the learned tile classifier's score is evidence
**against** fire. The `dnn` escalation path is the project's only learned component and the
reason `readNetFromONNX` / `blobFromImage` / `forward` are defended surfaces. A term that
enters the hazard with a negative fitted weight is not decoration in the R1 sense — it is
running and it is being read — but it is not doing what `opencv5-defense.md` §4 says it does.

## 3. What this does not say

- **Not that the classifier is broken.** It reports val precision 0.8498 / recall 0.8021 on
  Pyro-SDIS (`arm_dev_2026-09-06.json:classifier`). What is measured here is its contribution
  to *this* hazard on *this* arm through *this* gate — 51 frames of 2291, chosen by
  `should_escalate`, which only fires on cameras that are already ranked top-2 and already
  above a hazard floor. A statistic evaluated only where the hazard is already high is being
  asked a much harder question than its validation set asked.
- **Not that the refit should be adopted.** Adopting it changes the shipped hazard on the 51
  escalated frames, which moves the threshold, which moves the headline. That is a decision.
- **Not a re-litigation of S1–S4.** `statistics_confound_2026-09-06.md` stands unchanged, and
  its control still runs on every fit: presence-alone LOSO 0.5610, as-fitted 0.6393, presence
  explicit 0.6644, and S3 still flips sign under the control.

## 4. What follows — and none of it is an agent's call

1. **Decide whether the weights are refitted.** The honest default is yes: D11 specifies
   weights fit on DEV, and the shipped ones were fit on an input the pipeline does not
   produce. The cost is that the headline moves again.
2. **Decide what a negative escalation weight means for the `dnn` defence** before any
   judge-facing document claims the learned component helps. The measurable claim available
   today is *"it is invoked on 2.23% of frames and its fitted contribution on DEV is
   negative"*, which is a true sentence and not a good one.
3. **Consider whether `should_escalate`'s gate is the problem rather than the classifier.**
   Escalating only cameras that are already top-2 and already above a hazard floor means the
   classifier is asked to discriminate inside the hardest slice of the arm. Widening the gate
   is a design change with a cost in model calls per camera-night — the number the whole
   efficiency argument against B1 rests on.

## 5. Reproduce

```
python3 scripts/bench.py --arm dev --budget 2.0 --dump-features --seed 20261026
python3 scripts/fit_fusion.py --out results/fusion_refit_2026-09-06.json
```

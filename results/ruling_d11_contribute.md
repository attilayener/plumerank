# Ruling — what "contribute" means in the D11 abort criterion

**Written:** 2026-09-05 · **Status at time of writing:** the fitted-weight re-run has NOT been run.
No number produced by a fitted instrument existed when this document was written, and the
commit that carries it lands before the commit that carries that number.

> This exists because `gonogo_2026-09-01.md` §7.2 found §4's wording ambiguous, noticed that one
> reading was favourable, and refused to pick it after the fact. That refusal is only worth
> anything if the ambiguity is then closed *before* the next measurement. This closes it.

---

## 1. The ambiguity

§4 of the gate document says:

> At the threshold `rank.py` calibrates to 2.0 alerts/camera-night over the whole CONFUSE arm,
> measure what the **fog windows alone** contribute:
> **< 0.5 alerts/camera-night from fog** → clear · **> 1.5 from fog** → the primitive has failed.

On the 2026-09-05 unfitted measurement the two readings gave:

| reading | value | verdict under §4 |
|---|---:|---|
| fog's **share of the arm's** 2.00 budget | 0.6061 | *between* → proceed, name it a limitation |
| fog's **own rate**, on fog windows alone | 1.5385 | **above 1.5** → abort, descope to the fallback project |

## 2. The ruling: fog's OWN RATE binds

Three independent arguments, all pointing the same way, and the way they point is the
unfavourable one.

**2.1 Units.** §4 states its cutlines as `alerts/camera-night from fog`. That is a rate, in the
same units as the budget target itself (2.0 alerts/camera-night). A share of a budget is
dimensionless. Only one of the two candidate quantities can be substituted into the sentence §4
actually wrote without a unit error, and it is the own rate. The favourable reading requires
silently reinterpreting a stated unit; the unfavourable one does not.

**2.2 The gloss §4 supplies for itself.** §4 explains its own > 1.5 branch as *"fog eats the
budget every morning."* "Every morning" is a statement about what happens **on a foggy
camera-night**, not about what fog does to a roster average. Fog's own rate of 1.5385 against a
2.0 budget means that on a fog morning, fog takes 77% of that camera's alerts. That is the
sentence's plain meaning. The share reading answers a different question — how much of the whole
roster's budget did this class absorb — and would report fog as harmless simply by adding more
non-fog cameras to the arm.

**2.3 Scale invariance, which is decisive given §7.5.** The share reading is a function of the
arm's class composition. `fog_marine` is **13 windows against a quota of 18** — a supply limit,
recorded in §7.5. So the share is contaminated by a known, documented deviation in how the arm
was built: raise fog's count toward its quota and the share rises with it, while the own rate is
unchanged. A criterion whose verdict moves when a data-collection shortfall is fixed is not
measuring the primitive. **The own rate is the only one of the two that is a property of the
detector rather than a property of the roster.**

## 3. What this ruling does NOT do

It does **not** retroactively fire the abort on 1.5385. §7.3 recorded two unmet preconditions,
and the second is disqualifying on its own: **the fusion weights are not fitted.** `constants.py`
carries a-priori values written before any data was seen, while the original plan (D11) specifies "a
weighted logistic, weights fit on DEV". A criterion evaluated on an instrument the spec says is
not yet built measures the instrument, not the primitive.

So 1.5385 is a number taken with the wrong instrument, and this ruling declines to hang a project
on it — in exactly the same way, and for the same reason, that §7.2 declined to hang one on
0.6061.

## 4. What binds, and it binds whichever way it cuts

**The next measurement of fog's own rate, taken with weights fitted on DEV, is the D11 verdict.**

| fog's own rate, fitted | consequence |
|---|---|
| **< 0.5** | clear — the original GO condition is met on real evidence |
| **0.5 – 1.5** | proceed; published as a named limitation in `REPORT.md` and a row on `/results` |
| **> 1.5** | **the primitive has failed against its adversary.** Escalate to the human gate with a descope recommendation. |

Two constraints on the > 1.5 branch, both fixed now:

- **No further variant search may be used to overturn it.** The weights are fit once, by a
  procedure written down before the fit, on DEV only. If the result is > 1.5, that is the result.
- **The abort itself is a human gate.** No automated step switches this project to the fallback project. The
  measurement is executed and reported; the decision to abandon 51 days of runway belongs to a
  person.

## 5. The finding §4 never anticipated

§7.4 recorded that **cloud shadow** takes 4.43 alerts/camera-night — nearly three times fog's
rate — on a class S3 was predicted to reject most cleanly. §4 wrote cutlines for fog because the
spec named fog as the adversary. It wrote none for cloud shadow because nobody expected to need
them.

**No cutline is invented for cloud shadow now.** Writing an abort criterion for a class after
seeing it fail is the exact move §2 forbids, and inventing a threshold that the current number
happens to sit under would be worse than having none. Cloud shadow is therefore **measured,
published, and carried into `REPORT.md` §limitations regardless of the fog verdict** — as §7.4
already committed — and it is not permitted to trigger or block an abort at D11.

What the fitted re-run *does* legitimately test is §7.4's second reading: that unfitted weights
over-weight S3, letting a hard-edged moving region score high on S1 and S2 without S3 pulling it
down. If cloud shadow's rate falls sharply once the weights are fitted, that reading is supported.
If it does not, the finding survives fitting and is a real property of the primitive.

# Blind fog census — 2026-10-04

**Question.** Does the rule that selected the `fog_marine` confusion class actually pick frames
with a marine layer? `fog_class_review.md` suspected not (about half of the 60 class frames looked
like no marine layer), but that review was a single unblinded pass.

**Method.** `spike/fog_census_pool.json`: 271 fresh dawn frames (every second morning on the six
calibration cameras, no screening) interleaved with the 60 class frames under a fixed seed
(20260906). `spike/census.html` shows an index and a picture only — camera, date and arm are
hidden. Three answers: marine layer (a fog bank with a visible boundary), overcast ceiling (a grey
deck, no boundary), neither.

**Labels used: 162 of 324 (indices 0–161).** From index 162 the answers follow a mechanical 3-1-2
cycle and the labeller confirmed they were entered without judging the frame; they are discarded
(`spike/fog_census_labels.json` → `truncated`). Because the pool order was shuffled before
labelling, indices 0–161 are a random half of the pool.

| Group | n | Marine layer | Overcast | Neither |
|---|---:|---:|---:|---:|
| Fresh, unscreened | 129 | 38 (29%) | 67 (52%) | 24 (19%) |
| Current fog class | 28 | 8 (29%) | 16 (57%) | 4 (14%) |
| In both | 5 | 1 | 2 | 2 |

**Finding.** The selection rule's frames show a marine layer at the same rate as unscreened dawn
frames — 29% against a 29% base rate. On this sample the rule has no measurable enrichment for
marine layer; it mostly selects overcast decks. The CONFUSE windows labelled `fog_marine` should be
read as "dawn cloud/overcast", not as marine layer.

**What this does and does not change.** The alert threshold is solved on all 33 CONFUSE windows
together, whatever their class label, so the calibration and every DEV/HOLD number are unaffected.
What is affected is any per-class claim: the 1.15 alerts/camera-night attributed to fog
(`class_split_2026-09-06.json`) is a rate on a mostly-overcast class, not on marine layer. The
class is not rebuilt for this submission; the per-class fog figure is reported only with this
caveat.

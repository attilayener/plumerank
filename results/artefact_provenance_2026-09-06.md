# The published DEV headline is an ablation cell, not the shipped pipeline

**Date:** 2026-09-06 · **Status: settled by re-running, not by argument.** The shipped
configuration detects **9 of 29 (31.0%)** on DEV, not the 12 of 29 (41.4%) that four documents
and `project.json` quote. Nothing here changes a constant or a detector; it changes what may
be quoted.

---

## 1. How it was found

`spike/why_missed.py` was written to attribute the missed DEV sequences to causes, and it must
first reproduce the benchmark's own headline from `results/features_dev.jsonl` plus the shipped
constants — an attribution computed at a different operating point is an attribution of
something else. **It did not reproduce.** No pairing of `S4_ENABLED` with any committed
threshold gave any committed file's numbers.

So the arm was re-run on the shipped configuration:

```
python3 scripts/bench.py --arm dev --budget 2.0 --dump-features --seed 20261026
```

| | detected | threshold | mean TTD | frames escalated |
|---|---:|---:|---:|---:|
| **fresh run, shipped constants** | **9 / 29** | 0.740112 | 21.333 | 51 |
| `arm_dev_s4on_2026-09-06.json` *(since removed — see §7)* | 9 / 29 | 0.740112 | 21.333 | 51 |
| the plain-named `arm_dev_2026-09-05.json` **← what every document read** | 12 / 29 | 0.687561 | 19.917 | 4 |
| `arm_dev_s4off_2026-09-06.json` *(a byte-identical copy of the row above it)* | 12 / 29 | 0.687561 | 19.917 | 4 |

The fresh run reproduces the **S4-ON** file exactly, to six decimal places and to the frame.
That file is the shipped pipeline. The file carrying the plain name — the one every document
has been reading — is the **S4-OFF ablation cell**.

## 2. The mechanism, and it is one line

`bench.py` names its output `arm_<arm>_<date>.json` (`results_filename`). **`--s4 off` is not
in the name.** So a `--s4 off` run writes the same file a normal run does, and after the D10
gate on 2026-09-05 the plain DEV artefact was an S4-OFF run. It was then copied to
`arm_dev_s4off_2026-09-06.json` — the two are byte-identical, md5 `2f33e509…`, same start
timestamp — and the copy left the original in place under the name that reads as canonical.

Nothing was fabricated and nothing was hidden. A flag that changes the result but not the
filename is enough on its own.

## 3. What has to be re-stated

**PlumeRank's DEV detection rate is 31.0%, not 41.4%.** Four documents read the wrong file.
All four carry a correction as of this change:

- `results/SESSION_2026-09-06.md` §4 — the "12 / 29 (41.4%)" table, and the surrounding
  argument that the median does not exist. *The conclusion survives and gets worse:* 9/29 is
  further from a median than 12/29.
- `results/statistics_confound_2026-09-06.md` §4b — *"Only 12 of 29 are ever detected."*
  Re-measured: of the 25 sequences carrying a component on more than 80% of their
  post-ignition frames, 8 are detected and **17 never alert**, not thirteen. The presence
  arithmetic itself is unaffected.
- `results/s4_ablation_2026-09-06.md` §1 — see §5, which is its own problem.
- `project.json` `predictions_realised` — corrected in the same change as this document.

## 4. What is NOT wrong, and it nearly went into this document as though it were

Two false alarms were chased down and cleared. Both are recorded because the next reader will
hit them too:

**(a) The two ablation cells were NOT run under different constants.** Their `commit` fields
differ (`9b82b76` vs `501e069`, eight minutes apart) and `501e069` is the commit that changed
`ESCALATION_HAZARD_MIN`, which looks exactly like a confounded ablation. But
`constants_sha256` is **identical** in both files (`66f81f05…`): the escalation-floor change
was already in the working tree for the first run, and was committed between them. The hash is
the field that binds; the commit field records when HEAD moved.

**(b) The committed artefacts are NOT stale against today's constants.** Every arm file records
`66f81f05…` and `plumerank/constants.py` now hashes to `1b3147f6…`. The only commit between
them, `62a1d86`, changes **comments only** — the retraction text from
`statistics_confound_2026-09-06.md` §5. `freeze_hash()` hashes the file's bytes on purpose, so
*"it catches a changed comment that reveals a changed intent"*; here it did, and the intent was
a retraction. **The constant values are unchanged**, which is why the fresh run reproduces the
older artefact exactly.

## 5. One disagreement is still unexplained

`s4_ablation_2026-09-06.md` §1 tabulates `S4 ON 12/29 @ 0.757812, mean 18.083` and
`S4 OFF 14/29 @ 0.701904, mean 18.357`, and points at the two files above. **Four of those six
numbers appear in no committed artefact**, and neither threshold does. The table cannot have
been read out of the files it cites. Until that is resolved the table should not be quoted, and
the D10 S4 gate — a human decision either way — does not have a measurement under it.

A clean pair of cells was therefore re-run from one commit (`65ab2f8`), one constants file and
one seed:

| | S4 ON — shipped | S4 OFF |
|---|---:|---:|
| detected | 9 / 29 | 12 / 29 |
| mean TTD | 21.333 min | 19.917 min |
| threshold | 0.740112 | 0.687561 |
| frames escalated | 51 | 4 |

Both reproduce the original artefacts exactly, so **the original pair was right and only its
filing was wrong.** On this pair S4 costs three detections and 1.4 minutes of mean time — worse
on both axes. The gate is still a human decision (`s4_ablation.md` §4) and the median is still
undefined in both cells, but it now has a measurement under it. The correction is written into
`s4_ablation_2026-09-06.md` rather than only here, because that is the document a reader of the
gate will open.

## 6. The feature dump was from a third configuration again, and the fit used it

Before regeneration `results/features_dev.jsonl` carried **94** frames with a non-zero
escalation score. The shipped pipeline produces **51** (`escalated` in `bench.py` counts exactly
the frames the classifier is invoked on). Regenerated, the dump now matches the arm file to the
frame.

This one has a consequence beyond bookkeeping: **`scripts/fit_fusion.py` reads this dump**, so
the shipped `FUSION_W_ESCALATION` was fitted against an escalation column the pipeline does not
produce. What `statistics_confound_2026-09-06.md` measures — that the statistics add almost
nothing over a presence bit — is a property of S1–S4 and is untouched by this. The escalation
weight is not, and a refit on the regenerated dump is warranted before that weight is defended.

## 7. What should stop this recurring

`scripts/check_claims.py` scans `README.md`, `ARCHITECTURE.md`, `REPORT.md` and `DEMO.md` —
**none of which exist yet**. No mechanism compares a `results/*.md` table against the
`results/*.json` it names, and no mechanism notices two artefacts that are byte-identical under
different names. The R6 failure this project already knows about is documents drifting from
code; this is the same failure with results in place of code, and it is currently unguarded.

The cheapest fix is upstream of any checker: **put the cell in the filename.** An artefact that
cannot be told apart from another by its name will eventually be read as the other one.

**Done.** `bench.py:results_filename` now reserves the plain `arm_<arm>_<date>.json` for the
**shipped** configuration and files any other cell as `arm_<arm>_s4<cell>_<date>.json`. The
test of "shipped" is agreement with `constants.S4_ENABLED` rather than which flag was typed, so
deleting S4 at the D10 gate moves the plain name to the new shipped configuration with nothing
else to edit. A HOLD run in a non-shipped cell is **refused**, because `bench_s4off_<date>.json`
still matches the `bench_*.json` glob and sorts after the real headline. Two tests hold it:
`test_an_ablation_cell_cannot_occupy_the_shipped_arm_filename` and
`test_a_hold_ablation_is_refused_rather_than_misfiled`.

The artefacts have been re-filed to match: the 2026-09-05 cell is now
`arm_dev_s4off_2026-09-05.json`, the superseded `arm_dev_s4on_2026-09-06.json` is removed
because the re-run under the plain name reproduces it exactly, and both cells of the clean pair
were regenerated by the fixed code so that each payload's own `results_file` field agrees with
the name it sits under.

## 8. Reproduce

```
python3 scripts/bench.py --arm dev --budget 2.0 --dump-features --seed 20261026
md5 results/arm_dev_s4off_2026-09-05.json results/arm_dev_s4off_2026-09-06.json
python3 -m pytest tests/test_bench.py -k ablation_cell -q
git show 62a1d86 -- plumerank/constants.py        # comments only
```

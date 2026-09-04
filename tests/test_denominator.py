"""One denominator, everywhere.

``data/manifest_hold.json`` names 60 windows. ``data/excluded.json`` drops 6 of them. The
bench therefore divides by 54, and the whole failure this file exists to stop is a
sentence somewhere in the repo that still says 60 -- or, after a well-meaning fix, one
that says 54 with nothing beside it, which reads as six sequences that went missing.

Two guarantees, and they are different in kind:

1. **The published count is derived, not declared.** ``bench.arm_denominator`` reads the
   arm's manifest and the exclusion list and subtracts, and ``run_arm`` refuses to publish
   a count those two files do not produce. A number computed from the files it describes
   cannot drift from them; a number typed beside them always can.
2. **No committed JSON or Markdown in ``data/`` asserts a HOLD sequence count that
   disagrees.** This one SCANS rather than hardcodes, because the recurrence is not a
   typo -- it is a new file, written next month, restating a number that was true when it
   was written. A test listing today's offenders would go green the day after the next
   one is added.

Where the scan deliberately stops: it only reads counts it can tell are about HOLD, from
a marker word next to the number. ``data/manifest_hold.json``'s own "60 sequences, at most
2 per camera" is therefore not flagged, and should not be -- that file is describing
itself, and it is the one place 60 is the whole truth. The file name is NOT used as a
marker for exactly that reason.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Iterator

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench  # noqa: E402

DATA = ROOT / "data"
HOLD_MANIFEST = DATA / "manifest_hold.json"
EXCLUDED = DATA / "excluded.json"


# ---------------------------------------------------------------------------------
# The two files, read here without going through bench.py
# ---------------------------------------------------------------------------------


def _hold_ids() -> set[str]:
    blob = json.loads(HOLD_MANIFEST.read_text(encoding="utf-8"))
    return {str(w["sequence_id"]) for w in blob["windows"]}


def _excluded_ids() -> set[str]:
    """Every id ``data/excluded.json`` drops, across all arms.

    Read straight out of the file rather than through ``bench.load_exclusions`` so that
    the assertion below compares two independent readings of the same bytes. Running the
    subtraction through the same parser on both sides would assert that a function equals
    itself.
    """
    if not EXCLUDED.is_file():
        return set()
    blob = json.loads(EXCLUDED.read_text(encoding="utf-8"))
    records = blob["excluded"] if isinstance(blob, dict) else blob
    if isinstance(records, dict):
        return {str(k) for k in records}
    return {str(r["sequence_id"]) if isinstance(r, dict) else str(r) for r in records}


@pytest.fixture(scope="module")
def hold() -> dict[str, int]:
    if not HOLD_MANIFEST.is_file():
        pytest.skip("data/manifest_hold.json is not on disk")
    ids = _hold_ids()
    dropped = ids & _excluded_ids()
    return {
        "manifest_windows": len(ids),
        "excluded": len(dropped),
        "measured_on": len(ids) - len(dropped),
    }


# ---------------------------------------------------------------------------------
# 1. The count bench.py would publish for HOLD
# ---------------------------------------------------------------------------------


def test_the_hold_count_bench_publishes_is_the_manifest_minus_the_exclusions(
    hold: dict[str, int],
) -> None:
    """``arm_denominator("hold")`` is exactly ``len(manifest.windows) - len(HOLD drops)``.

    The subtraction is spelled out on the right-hand side rather than compared against a
    literal 54, so the test still means something on the day the manifest or the exclusion
    list changes -- which is the day it has to.
    """
    records, _ = bench.load_exclusions(ROOT)
    denominator = bench.arm_denominator(ROOT, "hold", records)

    assert denominator["manifest_windows"] == hold["manifest_windows"]
    assert denominator["excluded"] == hold["excluded"]
    assert denominator["measured_on"] == hold["measured_on"]
    assert (
        denominator["measured_on"]
        == denominator["manifest_windows"] - denominator["excluded"]
    )
    # Only HOLD's own drops count against HOLD's denominator: data/excluded.json spans
    # every arm, and subtracting its whole length would hand a reader an addition that
    # does not come out.
    assert set(denominator["excluded_sequence_ids"]) <= _hold_ids()


def test_the_published_statement_carries_all_three_numbers(hold: dict[str, int]) -> None:
    """The one sentence every surface quotes names the manifest, the drops and the rest.

    This is the "never one without the other" rule made mechanical at the source. Every
    caption in the repo either renders this string or renders the same three fields, so
    checking it here is checking all of them at once.
    """
    records, _ = bench.load_exclusions(ROOT)
    statement = bench.arm_denominator(ROOT, "hold", records)["statement"]
    for value in (hold["manifest_windows"], hold["excluded"], hold["measured_on"]):
        assert re.search(rf"\b{value}\b", statement), (
            f"the published statement omits {value}: {statement!r}"
        )
    assert "data/excluded.json" in statement, statement


def test_load_arm_and_arm_denominator_cannot_disagree() -> None:
    """The identity ``run_arm`` guards, exercised on the one arm whose frames are on disk.

    ``run_arm`` publishes ``n_windows = len(load_arm(...))`` and raises if it differs from
    ``arm_denominator``'s ``measured_on``. HOLD's pixels are not in the repo -- that is the
    point of HOLD -- so the identity is proved here on the committed micro corpus, with a
    real exclusion injected, using the same two functions HOLD will use. What is arm-
    specific about HOLD is the *inputs*, and test 1 above checks those.
    """
    micro = DATA / "micro" / "manifest_micro.json"
    if not micro.is_file():
        pytest.skip("data/micro/manifest_micro.json is not on disk")
    ids = sorted(bench.manifest_sequence_ids(ROOT, "micro"))
    if len(ids) < 2:
        pytest.skip("the micro manifest is too small to drop a window from")

    for records in ([], [{"sequence_id": ids[0], "reason": "injected by this test"}]):
        denominator = bench.arm_denominator(ROOT, "micro", records)
        windows = bench.load_arm(ROOT, "micro", [r["sequence_id"] for r in records])
        assert len(windows) == denominator["measured_on"], (
            f"load_arm replayed {len(windows)} windows where the manifest minus the "
            f"exclusion list is {denominator['measured_on']}"
        )
        assert denominator["excluded"] == len(records)


def test_wall_json_quotes_the_same_three_numbers_bench_will(hold: dict[str, int]) -> None:
    """``data/wall.json`` is the caption the demo video reads out loud.

    It is the highest-risk restatement in the repo: it is prose, it is committed months
    before the bench runs, and it is spoken rather than read. So its declared block is
    checked against the same subtraction, and its caption is required to carry both ends
    of it -- the surviving count so nobody divides by the wrong denominator, and the
    manifest count so the six that left do not look like six that vanished.
    """
    path = DATA / "wall.json"
    if not path.is_file():
        pytest.skip("data/wall.json is not on disk")
    wall = json.loads(path.read_text(encoding="utf-8"))
    declared = wall.get("hold_denominator")
    if declared is None:
        pytest.skip("data/wall.json declares no hold_denominator block")

    assert int(declared["manifest_windows"]) == hold["manifest_windows"]
    assert int(declared["excluded"]) == hold["excluded"]
    assert int(declared["measured_on"]) == hold["measured_on"]

    caption = str(wall.get("caption", ""))
    for value in (hold["measured_on"], hold["manifest_windows"], hold["excluded"]):
        assert re.search(rf"\b{value}\b", caption), (
            f"the wall caption omits {value}, so it publishes half a subtraction: "
            f"{caption!r}"
        )


def test_the_exclusion_disclosure_sits_where_the_count_is_published() -> None:
    """The pre-freeze read of HOLD first frames is disclosed beside the number it moved.

    Producing ``data/excluded.json`` meant a human looking at frame 1 of every HOLD
    window. That is defensible -- the frames are ~40 minutes before ignition, the criteria
    were fixed first, and nothing was scored -- and it is exactly the kind of fact that
    becomes an accusation if a judge finds it before we say it. So it is required to be
    present in every file that publishes the exclusion count, not only in the file that
    produced it.
    """
    for relative in ("excluded.json", "wall.json", "micro/PROVENANCE.md", "micro/manifest_micro.json"):
        path = DATA / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").upper()
        assert "FIRST FRAME" in text, (
            f"data/{relative} publishes the exclusion count without saying that the "
            f"exclusion pass read the first frame of every HOLD window"
        )


# ---------------------------------------------------------------------------------
# 2. The scan: nothing committed under data/ may assert a different HOLD count
# ---------------------------------------------------------------------------------

#: Words that mark a nearby number as being about the held-out arm. Case-sensitive on
#: "HOLD" so the ordinary English verb does not drag every sentence into the scan.
_HOLD_MARKERS = ("HOLD", "held-out", "held out", "manifest_hold", "unseen")

#: ...and words that mark it as being about a different arm. Checked FIRST, and only in
#: the text immediately after the number, because "the 45-window CONFUSE set" sits inside
#: sentences that are otherwise entirely about HOLD.
_OTHER_ARM_MARKERS = ("CONFUSE", "confuse", "DEV arm", "manifest_dev", "Pyro", "pyrosdis", "micro")

#: A number that is being used as a count of sequences or windows: the digits, up to
#: three intervening words, then the noun. Anything further away is not this number's noun.
_COUNT = re.compile(
    r"\b(\d{1,4})\b[\s\-]+((?:[A-Za-z][A-Za-z0-9_\-]*[\s\-]+){0,3}?)(sequence|window)s?\b"
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> Iterator[str]:
    for line in text.splitlines():
        for sentence in _SENTENCE.split(line):
            if sentence.strip():
                yield sentence


def _hold_counts(sentence: str) -> list[int]:
    """Every number in ``sentence`` that is asserting a count of HOLD sequences."""
    found: list[int] = []
    for match in _COUNT.finditer(sentence):
        span = match.group(2)
        # The two windows are deliberately different sizes. Another arm's name has to be
        # ATTACHED to the noun to disown the number -- "the 45-window CONFUSE set" -- and
        # anything further away is a different clause in the same sentence, which is
        # exactly the shape of "60 unseen HOLD sequences against the 45-window CONFUSE
        # adversary set". A HOLD marker, by contrast, is allowed to sit further off,
        # because "data/manifest_hold.json names 60 windows" puts it in front.
        if any(m in span or m in sentence[match.end() : match.end() + 14] for m in _OTHER_ARM_MARKERS):
            continue
        near = (
            span
            + sentence[match.end() : match.end() + 60]
            + sentence[max(0, match.start() - 70) : match.start()]
        )
        if not any(m in near for m in _HOLD_MARKERS):
            continue
        found.append(int(match.group(1)))
    return found


def _committed_text_files() -> list[pathlib.Path]:
    return sorted(
        [p for p in DATA.rglob("*.json") if p.is_file()]
        + [p for p in DATA.rglob("*.md") if p.is_file()]
    )


def test_no_committed_data_file_asserts_a_disagreeing_hold_count(
    hold: dict[str, int],
) -> None:
    """Scan ``data/`` for HOLD sequence counts and hold every one of them to the files.

    Two failures, and the second is the one that keeps coming back:

    * a count that is none of {manifest, excluded, measured-on} -- a stale number;
    * a count quoted **alone** -- the manifest's 60 with no mention of what left, or the
      surviving 54 with no mention of what it was subtracted from. Both are true
      sentences that mislead, which is why the rule is a rule and not a preference.
    """
    allowed = {hold["manifest_windows"], hold["excluded"], hold["measured_on"]}
    partner = {hold["manifest_windows"], hold["excluded"]}
    offences: list[str] = []

    for path in _committed_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for sentence in _sentences(text):
            counts = _hold_counts(sentence)
            if not counts:
                continue
            where = path.relative_to(ROOT)
            stale = [c for c in counts if c not in allowed]
            if stale:
                offences.append(
                    f"{where}: asserts a HOLD count of {stale} -- the manifest holds "
                    f"{hold['manifest_windows']}, {hold['excluded']} are excluded and "
                    f"{hold['measured_on']} remain:\n    {sentence.strip()[:220]}"
                )
                continue
            has = {v for v in allowed if re.search(rf"\b{v}\b", sentence)}
            if hold["measured_on"] not in has or not (has & partner):
                offences.append(
                    f"{where}: quotes a HOLD count without its partner. Publish the "
                    f"measured-on count ({hold['measured_on']}) together with the "
                    f"manifest count ({hold['manifest_windows']}) or the exclusion "
                    f"count ({hold['excluded']}):\n    {sentence.strip()[:220]}"
                )

    assert not offences, "\n".join(offences)


def test_the_scan_recognises_the_sentences_it_exists_to_catch() -> None:
    """A scanner nobody has shown to fire is a scanner that passes on an empty repo.

    The BAD strings below are the ones that were actually in this repo -- verbatim -- plus
    the two failure shapes a well-meaning fix produces. The GOOD strings are sentences the
    scan must stay quiet about: a manifest describing its own length, and another arm's
    window count sitting inside a sentence about HOLD.
    """
    bad = {
        "The headline is measured on 60 unseen HOLD sequences against the "
        "45-window CONFUSE adversary set.": [60],
        "median TTD @ 2 alerts/camera-night, n = 60 held-out sequences": [60],
        "The headline number is measured on 54 HOLD sequences it has never seen.": [54],
        "the 60-window HOLD arm": [60],
    }
    for sentence, expected in bad.items():
        assert _hold_counts(sentence) == expected, sentence

    good = (
        # A manifest describing itself. 60 is the whole truth inside manifest_hold.json.
        "60 sequences, at most 2 per camera so no single camera dominates the headline.",
        # The adversary set, named in a sentence that is otherwise about HOLD.
        "the 45-window CONFUSE adversary set",
        "45 non-fire windows chosen by confusion class",
        "30 sequences on the DEV arm the constants were fit on",
    )
    for sentence in good:
        assert _hold_counts(sentence) == [], sentence

    # And the pair rule: both ends of the subtraction, or the sentence is an offence.
    paired = (
        "The headline is measured on 54 held-out sequences: 60 windows in "
        "data/manifest_hold.json, 6 excluded by data/excluded.json."
    )
    assert _hold_counts(paired) == [54, 60]
    assert all(re.search(rf"\b{v}\b", paired) for v in (54, 60, 6))


def test_a_missing_exclusion_list_is_at_least_recorded_in_what_bench_publishes(
    tmp_path: pathlib.Path,
) -> None:
    """The one hole left open, pinned so it cannot get quieter than it already is.

    On the committed repo ``data/excluded.json`` is tracked and this cannot happen. If it
    ever did, ``load_exclusions`` would answer "nothing excluded", ``arm_denominator``
    would subtract zero, and the published sentence would be "60 sequences, 0 excluded
    from the 60 in manifest_hold.json" -- the overstated denominator this whole file
    exists to stop, internally consistent and so invisible to every other check here.

    The guard that would close it (raise on absence) cannot land yet:
    ``tests/test_results_contract.py`` runs a real HOLD arm over a synthetic tree that
    legitimately has no exclusion list. Until that fixture writes an empty one, the fact
    is carried instead of refused -- ``load_exclusions`` returns a source string naming
    the absence, and ``run_arm`` publishes it verbatim as ``exclusions.source``. This
    test pins that string, so the absence stays visible in the payload a judge reads.
    """
    (tmp_path / "data").mkdir()
    records, source = bench.load_exclusions(tmp_path)

    assert records == []
    assert "excluded.json" in source
    assert "absent" in source, (
        "an absent exclusion list must say so in the string bench publishes as "
        "exclusions.source -- it is the only trace left that the denominator was "
        "never reduced"
    )


def test_the_real_exclusion_list_still_loads_and_agrees_with_the_file(
    tmp_path: pathlib.Path,
) -> None:
    """The raise above must not have been bought by breaking the working path.

    Guards the shape of the fix: a raise that also fires on the committed file would make
    every bench run fail, and a test that only asserted the raise would not notice.
    """
    records, source = bench.load_exclusions(ROOT)
    blob = json.loads((ROOT / "data" / "excluded.json").read_text(encoding="utf-8"))

    assert source.endswith("excluded.json")
    assert len(records) == len(blob["excluded"]) == blob["count"]
    assert {r["sequence_id"] for r in records} == {
        e["sequence_id"] for e in blob["excluded"]
    }
    # Every record carries its reason: a bare id answers the count and refuses the
    # question the count invites.
    assert all(r["reason"].strip() for r in records)

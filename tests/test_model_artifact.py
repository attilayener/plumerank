"""Tests for the shipped model artefact -- the file, not the code that made it.

``tests/test_escalate.py`` proves the ``cv2.dnn`` call path against a hand-built ONNX and
never touches ``models/tile_classifier.onnx``; that is right, because the call path has to
be testable on a box with no trained model. This file is the other half, and it asserts
the opposite thing: that the model which actually ships is the one the manifest describes,
that OpenCV 5's new engine can load it, and that the numbers published beside it were
measured rather than defaulted.

Why a separate file at all. The escalation tests build their own weights precisely so
their arithmetic is analytically known, which means they would pass unchanged against an
empty ``models/`` directory -- and an empty ``models/`` directory is exactly the state in
which ``bench.py --arm micro`` dies and ``make verify``, the judge one-liner, has never
run. Nothing asserted the artefact itself, so nothing failed when it was missing.

Every test here skips LOUDLY when the artefact is absent: the skip message names the
command that produces it. A vacuous skip on a missing model is how this hole stayed open.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import cv2
import numpy as np
import pytest

from plumerank.constants import TILE_SIZE, WORK_HEIGHT, WORK_WIDTH
from plumerank.escalate import TileClassifier

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MODEL = _ROOT / "models" / "tile_classifier.onnx"
_HASH = _ROOT / "models" / "tile_classifier.onnx.sha256"
_MANIFEST = _ROOT / "data" / "tile_manifest.json"
_HOLD = _ROOT / "data" / "manifest_hold.json"

_TRAIN_COMMAND = "python scripts/train_tile_classifier.py  (Pyro-SDIS, hours on CPU)"

#: Same markers as ``scripts/train_tile_classifier.py`` and ``tests/test_corpus_integrity.py``.
#: Repeated rather than imported from the script: this file must fail if the script's list
#: is ever narrowed, and a shared constant would narrow with it.
_FIGLIB_MARKERS: tuple[str, ...] = ("figlib", "hpwren", "data/corpus", "data\\corpus")


def _require_model() -> pathlib.Path:
    if not _MODEL.is_file():
        pytest.skip(
            f"{_MODEL.relative_to(_ROOT)} is absent, so nothing here can be asserted "
            f"about the model that ships. Produce it with `{_TRAIN_COMMAND}`. Until then "
            f"`scripts/bench.py --arm micro` -- and therefore `make verify` -- cannot run."
        )
    return _MODEL


def _require_manifest() -> dict:
    if not _MANIFEST.is_file():
        pytest.skip(
            f"{_MANIFEST.relative_to(_ROOT)} is absent: no training run has recorded what "
            f"the model saw. Produce it with `{_TRAIN_COMMAND}`."
        )
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------
# 1. The bytes that ship are the bytes that were measured
# ---------------------------------------------------------------------------------


def test_the_committed_hash_matches_the_committed_model() -> None:
    """``models/tile_classifier.onnx.sha256`` is the model beside it, and the manifest agrees.

    Three files have to say the same thing: the model, the sidecar hash, and the ``model``
    block of ``data/tile_manifest.json``. They are written by two different steps, and the
    published precision and recall belong to the weights that existed at the moment the
    manifest was written -- so a model swapped in afterwards would carry an evaluation it
    never earned, and nothing else in the suite would notice.
    """
    model = _require_model()
    if not _HASH.is_file():
        pytest.fail(
            f"{model.relative_to(_ROOT)} ships without {_HASH.relative_to(_ROOT)}. The "
            f"model is the one binary in this repository; an unhashed binary is one a "
            f"judge has to take on trust."
        )
    actual = hashlib.sha256(model.read_bytes()).hexdigest()
    # First whitespace token, so a file written by `shasum -a 256` (hash, two spaces,
    # name) and one holding the bare hash both read -- the same rule bench.py applies to
    # results/constants.sha256.
    recorded = _HASH.read_text(encoding="utf-8").split()
    assert recorded, f"{_HASH.relative_to(_ROOT)} is empty"
    assert recorded[0] == actual, (
        f"{_HASH.relative_to(_ROOT)} records {recorded[0]} but the model on disk hashes "
        f"to {actual}. One of the two was changed without the other."
    )

    manifest = _require_manifest()
    declared = (manifest.get("model") or {}).get("sha256")
    assert declared == actual, (
        f"data/tile_manifest.json describes a model with sha256 {declared}, but the "
        f"model on disk hashes to {actual}. The val precision and recall in that manifest "
        f"were measured on different weights than the ones that ship."
    )


# ---------------------------------------------------------------------------------
# 2. OpenCV 5's new engine can load it, and it answers
# ---------------------------------------------------------------------------------


def test_the_model_loads_under_engine_new() -> None:
    """The realistic failure of the export is not "inaccurate", it is "does not load".

    ``ENGINE_NEW`` covers ~75-80% of the ONNX spec against ``ENGINE_CLASSIC``'s ~22%, and
    the architecture was chosen (Conv/Relu/MaxPool/Flatten/Gemm) so that it lands well
    inside that. This asserts the choice paid off on the file that actually shipped rather
    than on a graph a test wrote for itself.
    """
    clf = TileClassifier(_require_model())
    assert clf.engine == cv2.dnn.ENGINE_NEW
    assert clf.invocations == 0


def test_a_forward_pass_on_one_tile_returns_a_finite_probability() -> None:
    """One 96x96 tile in, one probability in [0, 1] out, one forward pass spent.

    The frame is exactly one tile so that nothing about the grid, the slide-inside-the-
    frame rule or the centroid geometry can contribute: what is under test is that the
    shipped weights produce a number a hazard can be fused with. ``escalate.score_tile``
    already refuses a non-finite output, so a NaN here fails as an exception, not as a
    plausible 0.0 -- which is the failure mode this assertion exists to make impossible.
    """
    clf = TileClassifier(_require_model())
    tile = np.full((TILE_SIZE, TILE_SIZE, 3), 128, dtype=np.uint8)
    probability = clf.score_tile(tile, (0, 0, TILE_SIZE, TILE_SIZE))
    assert np.isfinite(probability)
    assert 0.0 <= probability <= 1.0
    assert clf.invocations == 1


def test_the_grid_use_answers_on_a_working_resolution_frame() -> None:
    """Use 1 -- baseline B1 -- on the raster the pipeline actually runs at.

    B1 is the headline baseline and it is the SAME model, so a shipped artefact that
    scored a lone tile but fell over on the full-frame grid would take the comparison down
    with it. The gradient makes the frame something other than a constant, because a net
    that returns the same number for every input would pass a constant-frame check.
    """
    clf = TileClassifier(_require_model())
    ramp = np.linspace(0, 255, WORK_WIDTH, dtype=np.float32)
    frame = np.repeat(ramp[None, :], WORK_HEIGHT, axis=0).astype(np.uint8)
    score = clf.score_frame_grid(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0
    assert clf.invocations > 1, "a full-frame grid is more than one tile"


# ---------------------------------------------------------------------------------
# 3. What it was trained on, and what that buys
# ---------------------------------------------------------------------------------


def test_the_manifest_names_no_figlib_path_and_no_hold_sequence() -> None:
    """The out-of-sample property, asserted at the artefact rather than at the intention.

    ``test_no_figlib_in_training`` checks the marker list; this adds the HOLD sequence ids
    themselves, read out of the committed HOLD manifest. A directory rename defeats a
    marker -- ``data/cal/2024.../frame.jpg`` carries none of them -- and the sequence ids
    are long, dated and unmistakable, so they survive any renaming that keeps the frames
    identifiable. HOLD is the arm the headline is measured on: one of its ids in this
    manifest would mean the model that scores it had already seen it.
    """
    manifest = _require_manifest()
    sources = [str(source) for source in manifest.get("sources", [])]
    assert sources, (
        "data/tile_manifest.json lists no sources. An empty list satisfies every check "
        "below while proving nothing about what the model was trained on."
    )

    hits = [s for s in sources if any(marker in s.lower() for marker in _FIGLIB_MARKERS)]
    assert not hits, (
        f"{len(hits)} FIgLib path(s) in the training corpus, e.g. {hits[0]}. Every "
        f"published Californian number would be in-sample."
    )

    if not _HOLD.is_file():
        pytest.fail("data/manifest_hold.json is absent, so the HOLD ids cannot be checked")
    hold = json.loads(_HOLD.read_text(encoding="utf-8"))
    ids = {str(window["sequence_id"]) for window in hold["windows"]}
    blob = "\n".join(sources)
    named = sorted(sequence_id for sequence_id in ids if sequence_id in blob)
    assert not named, (
        f"the training corpus names HOLD sequence(s) {named[:3]} -- the held-out arm is "
        f"held out no longer."
    )


def test_the_published_val_precision_and_recall_are_measured_numbers() -> None:
    """Real numbers, from a real held-out split, on a real corpus.

    ``precision_recall`` returns ``None`` where a denominator is empty, and that is the
    honest answer -- but ``None`` must never be what reaches ``/results``, which renders
    this pair straight out of the benchmark payload. So the shipped manifest is required
    to carry two measured numbers and a non-zero tile count, and the corpus they were
    measured on is required to be Pyro-SDIS rather than the ``--synthetic`` throwaway
    whose own help text says it does not produce a usable model.
    """
    manifest = _require_manifest()
    assert manifest.get("corpus") == "pyro-sdis", (
        f"the shipped model was trained on corpus {manifest.get('corpus')!r}, not "
        f"pyro-sdis. Only Pyro-SDIS makes the Californian numbers out-of-sample."
    )
    assert manifest.get("throwaway") is False, (
        "the shipped model is flagged throwaway, which is what --synthetic writes. A "
        "throwaway model has no evaluation meaning and B1 is that same model."
    )

    val = manifest.get("val")
    assert isinstance(val, dict), "data/tile_manifest.json carries no val block"
    for field in ("precision", "recall"):
        value = val.get(field)
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"val {field} is {value!r}. None means the denominator was empty -- nothing "
            f"was measured -- and publishing it as a number is how an unevaluated model "
            f"reaches a report."
        )
        assert 0.0 <= float(value) <= 1.0, f"val {field} is {value!r}, outside [0, 1]"
    assert int(val.get("n_tiles", 0)) > 0, "the val split contributed no tiles"

    counts = manifest.get("counts") or {}
    assert int(counts.get("train_positive", 0)) > 0, (
        "the training corpus contains no positive tile, so the model was fitted to a "
        "single class and its val recall cannot mean anything."
    )

"""The published results schema, asserted against a payload ``bench.py`` really produced.

Why this file exists
--------------------
``results/bench_<date>.json`` is read by three things that never see each other:
``web/lib/read.ts`` (which renders ``/results``, the screen carrying the 30%-weighted
technical-execution criterion), ``api/main.py`` (which calibrates the live wall off the
committed threshold) and ``scripts/check_submission_readiness.py`` (which compares the
served headline against the committed one). Before the reconciliation, the reader looked
up four of those quantities under names ``bench.py`` had never written -- ``systems``
against ``baselines``, ``alerts_per_camera_night`` against
``calibration_realised_rate_uncapped``, ``model_invocations_per_camera_night`` against
``model_calls_per_camera_night``, a flat ``headline`` against a per-system map -- and
every one of those misses rendered as the *designed* "not published in
results/bench_<date>.json" state. A designed empty state is the right answer to a missing
number and the worst possible answer to a present one: the page said the benchmark had not
published its own per-system table while sitting on a file that had.

So this file is the seam, written down. It runs the real script over a real (synthetic,
but real-pixel) corpus and asserts that every name the reader looks up is in the document
with the type the reader expects. **It fails if bench.py renames a published key.** That
is its whole purpose, and it is the only mechanism in the repo that has one.

The corpus generator is imported from ``test_bench`` rather than copied. Two files that
each built their own synthetic arm would be two definitions of "a fire window", and a
contract test whose fixture had drifted from the benchmark's own fixture would be checking
a document nobody produces.

What this file does NOT do
--------------------------
It does not assert numeric values -- ``test_bench.py`` owns those, and a contract test that
also owned the arithmetic would fail for two unrelated reasons. It asserts names, types,
and the three structural facts without which a panel renders empty against a real file.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import bench  # noqa: E402
import test_bench as fixtures  # noqa: E402  -- the synthetic corpus, defined once
import train_tile_classifier as trainer  # noqa: E402

from plumerank import constants as K  # noqa: E402

READ_TS = ROOT / "web" / "lib" / "read.ts"

COMMIT = "1" * 40


# ---------------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------------
# Each row is (dotted path in results/bench_<date>.json, expected JSON kind, the component
# that consumes it). "number?" and "string?" mark the quantities the reader is allowed to
# receive as null because a designed empty state exists for them; everything else must be
# present with a value on a run that completed.
#
# The paths are the ones scripts/bench.py writes. Where the reader wanted a vaguer name
# than the file publishes -- one "alerts per camera-night" against a document that
# deliberately publishes two different rates -- the READER moved, because build-contract.md
# requires the published file to say which of the two it means every time it prints one.
# Where the file had no addressable name at all -- a per-system map called `headline`, a
# solved threshold with nowhere to live -- BENCH moved, because api/main.py and the
# readiness check address those exactly as read.ts does and bench was the odd one out.

CONTRACT: tuple[tuple[str, str, str], ...] = (
    # --- run provenance: the header line and ReproduceBlock on /results -----------
    ("arm", "string", "results/page.tsx header -- 'median TTD, <arm> arm'"),
    ("n_windows", "number", "results/page.tsx header -- '<n> sequences read exactly once'"),
    ("run.commit", "string", "ReproduceBlock / BuildInfo -- the checkout the number came from"),
    ("run.instance", "string", "ReproduceBlock -- 'produced on the machine that serves it'"),
    ("run.date", "string", "ReproduceBlock"),
    ("run.wall_clock_seconds", "number", "ReproduceBlock"),
    ("run.constants_sha256", "string", "ReproduceBlock -- the freeze the HOLD read happened after"),
    # --- THE number ---------------------------------------------------------------
    ("headline_system", "string", "read.ts -- which system SensitivityStrip draws"),
    ("headline.p50_minutes", "number?", "results/page.tsx -- the largest figure on the screen"),
    ("headline.p95_minutes", "number?", "results/page.tsx -- the p95 beside it"),
    ("headline.n_missed", "number", "results/page.tsx -- the miss count, never under the p50"),
    # --- the operating point: the Budget calibration panel, and the live wall ------
    (
        "calibration.target_alerts_per_camera_night",
        "number",
        "results/page.tsx budget panel + OperatingCurve's annotated rule",
    ),
    (
        "calibration.realised_alerts_per_camera_night_uncapped",
        "number",
        "results/page.tsx budget panel -- the UNCAPPED rate, as labelled",
    ),
    (
        "calibration.threshold",
        "number",
        "results/page.tsx budget panel AND api/main.py:_calibrated_threshold",
    ),
    ("calibration.drift_tolerance", "number", "read.ts Benchmark.budget.drift_tolerance"),
    # --- the three panels that rendered 'not published' ----------------------------
    ("baselines", "list", "BaselineRace + the per-system table"),
    ("operating_curve", "list", "OperatingCurve"),
    ("sensitivity.rows", "list", "SensitivityStrip"),
    # --- escalation and cost -------------------------------------------------------
    ("escalation.fraction", "number", "results/page.tsx escalation panel"),
    (
        "escalation.structural_ceiling",
        "number",
        "results/page.tsx -- the bound the run asserts, not the design target",
    ),
    # --- the classifier row, quoted from the training manifest ---------------------
    ("classifier.val_precision", "number?", "results/page.tsx -- what stops B1 reading as a strawman"),
    ("classifier.val_recall", "number?", "results/page.tsx"),
    ("classifier.dataset", "string?", "results/page.tsx -- names the corpus the pair came from"),
    (
        "classifier.throwaway",
        "bool?",
        "read.ts readClassifierDataset -- the flag that keeps a five-minute model from "
        "reading as an evaluation",
    ),
    # --- exclusions -----------------------------------------------------------------
    ("exclusions.count", "number", "results/page.tsx exclusions panel"),
    ("exclusions.items", "list", "results/page.tsx exclusions table -- records, not ids"),
)

#: Keys read out of ONE ROW rather than out of the document. Same rule, one level down.
ROW_CONTRACT: tuple[tuple[str, str, str, str], ...] = (
    ("baselines", "system", "string", "read.ts SystemResult.key -- BaselineRace lane identity"),
    ("baselines", "label", "string", "BaselineRace lane name + per-system table"),
    ("baselines", "ttd.p50_minutes", "number?", "BaselineRace lane position"),
    ("baselines", "ttd.p95_minutes", "number?", "per-system table"),
    ("baselines", "ttd.n_missed", "number?", "BaselineRace -- '<n> missed' beside every lane"),
    (
        "baselines",
        "calibration_realised_rate_uncapped",
        "number",
        "per-system table, 'alerts / camera-night' -- the CONFUSE rate the block quotes",
    ),
    (
        "baselines",
        "model_calls_per_camera_night",
        "number",
        "per-system table, 'model calls / camera-night' -- the measured 99/1 claim",
    ),
    ("operating_curve", "budget_alerts_per_camera_night", "number", "OperatingCurve x axis"),
    ("sensitivity.rows", "offset_minutes", "number", "SensitivityStrip x axis"),
)

#: Row keys whose name is built from a system id. read.ts interpolates these, so they are
#: listed as templates and expanded against the roster the document itself publishes.
TEMPLATED_ROW_KEYS: tuple[tuple[str, str, str], ...] = (
    ("operating_curve", "{system}_p50_minutes", "OperatingCurve -- one series per system"),
    ("sensitivity.rows", "{system}_p50_minutes", "SensitivityStrip -- the drawn line"),
    ("sensitivity.rows", "{system}_p95_minutes", "read.ts SensitivityPoint.p95_minutes"),
)


# ---------------------------------------------------------------------------------
# One real run
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One completed ``bench.py --arm hold`` and everything it left behind."""

    root: pathlib.Path
    path: pathlib.Path
    raw: bytes
    payload: dict[str, Any]
    stdout: str


def _two_cell_grid() -> list[tuple[str, bool, bool]]:
    """Two MOG2 cells instead of twelve.

    The twelve-cell grid is asserted by ``test_bench.py``; what this file needs from the
    ablation is the SHAPE of the published rows, and a cell is a cell. Running LSBP and
    GSOC here would add about sixteen seconds to a suite whose whole budget is ninety --
    and would assert nothing this file is about.
    """
    return [("mog2", True, True), ("mog2", False, False)]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Run]:
    """A real HOLD run over the synthetic corpus, in a tmp tree.

    HOLD rather than DEV so the file lands under the name ``/api/benchmark`` actually
    serves -- ``results/bench_<date>.json``. That means the frozen-constants file and a
    commit sha, which is the same discipline a real HOLD read is held to.
    """
    root = tmp_path_factory.mktemp("contract") / "tree"
    corpus = root / "data" / "corpus" / "figlib"
    (root / "results").mkdir(parents=True, exist_ok=True)
    fixtures._write_onnx(root / "models" / "tile_classifier.onnx")
    fixtures._manifest(root / "data" / "manifest_dev.json", fixtures._arm(corpus, "dev", 2, True))
    fixtures._manifest(root / "data" / "manifest_hold.json", fixtures._arm(corpus, "hold", 2, True))
    fixtures._manifest(
        root / "data" / "manifest_confuse.json", fixtures._arm(corpus, "conf", 2, False)
    )
    (root / "results" / "constants.sha256").write_text(K.freeze_hash() + "\n", encoding="utf-8")
    _write_training_manifest(root)

    patch = pytest.MonkeyPatch()
    patch.setattr(bench, "git_commit", lambda _root: COMMIT)
    patch.setattr(bench, "ablation_cells", _two_cell_grid)
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            status = bench.main(
                ["--arm", "hold", "--sensitivity", "--ablation", "--root", str(root)]
            )
    finally:
        patch.undo()
    assert status == 0, buffer.getvalue()

    path = root / "results" / f"bench_{bench._today()}.json"
    raw = path.read_bytes()
    yield Run(
        root=root,
        path=path,
        raw=raw,
        payload=json.loads(raw.decode("utf-8")),
        stdout=buffer.getvalue(),
    )


def _write_training_manifest(root: pathlib.Path) -> None:
    """The tile classifier's manifest, written by the training script's own writer.

    Written by ``train_tile_classifier.write_manifest`` rather than typed here, because the
    classifier row on /results is bench.py quoting that file: if the two ever disagree
    about where ``val.precision`` lives, this is the test that has to notice. The
    precision/recall pair is computed by the trainer's own ``precision_recall`` over two
    small label arrays, so no number in this fixture is typed either.
    """
    predicted = np.array([1, 1, 0, 1, 0, 0], dtype=np.int64)
    truth = np.array([1, 0, 0, 1, 1, 0], dtype=np.int64)
    val = dict(trainer.precision_recall(predicted, truth), n_tiles=int(truth.size))
    trainer.write_manifest(
        root / "data" / "tile_manifest.json",
        sources=["pyro-sdis/images/train/0001.jpg"],
        counts={"source_images": 1, "train_tiles": 6},
        model_path=root / "models" / "tile_classifier.onnx",
        corpus="pyro-sdis",
        throwaway=False,
        val=val,
    )


# ---------------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------------


def _kind_of(value: Any) -> str:
    # bool before number: `isinstance(True, int)` is True in Python, and a boolean
    # published where /results renders a rate would draw a 1 and look like a measurement.
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


_ABSENT = object()


def _rows(payload: dict[str, Any], path: str) -> list[Any]:
    """A published list, or a failure that names the path rather than a bare KeyError."""
    value = _at(payload, path)
    assert isinstance(value, list), f"results/bench_<date>.json:{path} is {_kind_of(value)}"
    return value


def _at(doc: Any, path: str) -> Any:
    """``read.ts``'s dotted lookup, in Python. Absent and null are different answers."""
    cur: Any = doc
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return _ABSENT
        cur = cur[key]
    return cur


def _assert_kind(value: Any, kind: str, where: str, consumer: str) -> None:
    assert value is not _ABSENT, (
        f"{where} is ABSENT from the payload bench.py just wrote. {consumer} reads it, and "
        f"a reader that cannot find it renders the designed 'not published in "
        f"results/bench_<date>.json' state against a real results file."
    )
    optional = kind.endswith("?")
    expected = kind.rstrip("?")
    actual = _kind_of(value)
    if optional and actual == "null":
        return
    assert actual == expected, (
        f"{where} is a JSON {actual}, but {consumer} reads it as a {expected}"
    )


# ---------------------------------------------------------------------------------
# The contract, against the document
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("path,kind,consumer", CONTRACT, ids=[row[0] for row in CONTRACT])
def test_every_key_the_web_reads_is_published(run: Run, path: str, kind: str, consumer: str) -> None:
    """One row of the contract, against a document scripts/bench.py just produced.

    This is the test that fails on a rename. It cannot pass against a hand-written
    fixture, because there is no hand-written fixture: the payload comes out of the same
    ``run_arm`` that writes the committed file.
    """
    _assert_kind(_at(run.payload, path), kind, f"results/bench_<date>.json:{path}", consumer)


@pytest.mark.parametrize(
    "container,key,kind,consumer",
    ROW_CONTRACT,
    ids=[f"{row[0]}[].{row[1]}" for row in ROW_CONTRACT],
)
def test_every_row_key_the_web_reads_is_published(
    run: Run, container: str, key: str, kind: str, consumer: str
) -> None:
    rows = _at(run.payload, container)
    assert isinstance(rows, list) and rows, f"{container} is empty, so its rows assert nothing"
    for index, row in enumerate(rows):
        _assert_kind(_at(row, key), kind, f"{container}[{index}].{key}", consumer)


@pytest.mark.parametrize(
    "container,template,consumer",
    TEMPLATED_ROW_KEYS,
    ids=[f"{row[0]}[].{row[1]}" for row in TEMPLATED_ROW_KEYS],
)
def test_the_per_system_columns_exist_for_every_published_system(
    run: Run, container: str, template: str, consumer: str
) -> None:
    """``read.ts`` builds these names by interpolating a system id.

    The ids come from the document's own ``baselines`` roster rather than from sniffing
    column suffixes: a chart that discovered its own series names would draw whatever a
    future column happened to be called.
    """
    systems = [row["system"] for row in run.payload["baselines"]]
    assert systems, "no systems published, so no per-system column can be checked"
    rows = _at(run.payload, container)
    for system in systems:
        key = template.format(system=system)
        for index, row in enumerate(rows):
            _assert_kind(_at(row, key), "number?", f"{container}[{index}].{key}", consumer)


def test_the_ablation_rows_are_flat_records(run: Run) -> None:
    """``readAblation`` derives its columns from the union of the row keys.

    Deliberately, because the grid is the artefact and the frontend has no business
    knowing its column names -- but that only works while a row is a flat object of
    scalars. A nested value would render as ``[object Object]`` in a table cell on the
    screen carrying the technical-execution criterion.
    """
    rows = _rows(run.payload, "ablation")
    assert rows, "--ablation was passed, so cells must be published"
    for row in rows:
        assert isinstance(row, dict), row
        for key, value in row.items():
            assert _kind_of(value) in {"string", "number", "null"}, f"ablation cell {key}: {value!r}"


def test_the_exclusion_rows_carry_a_reason(run: Run) -> None:
    """``readExclusions`` drops every entry whose ``sequence_id`` is empty, so a list of
    bare strings renders as an empty panel beside a denominator that moved."""
    for item in _rows(run.payload, "exclusions.items"):
        assert isinstance(item, dict), item
        assert isinstance(item.get("sequence_id"), str) and item["sequence_id"]
        assert isinstance(item.get("reason"), str)


# ---------------------------------------------------------------------------------
# The three panels that were rendering 'not published' against a real file
# ---------------------------------------------------------------------------------


def test_every_system_has_a_lane_and_the_headline_system_has_a_median(run: Run) -> None:
    """``BaselineRace`` draws one lane per published system. A lane is drawable when it
    carries EITHER a p50 OR a detection count -- a null p50 beside ``n_detected`` is a
    system that missed more than half the fires, whose median is off the clock. That is a
    result, not an absence: on the real DEV arm B1 finds 1 of 29 and B2 finds 3 of 29.

    Until 2026-09-17 this test demanded a p50 from every system and the component filtered
    null lanes out with the caption "only 2 of 3 lanes published" -- hiding, on the real
    file, the single most important comparison on the page behind a publishing apology.
    """
    rows = _rows(run.payload, "baselines")
    assert {r["system"] for r in rows} == set(bench.SYSTEMS)
    for r in rows:
        ttd = r["ttd"] or {}
        assert isinstance(ttd.get("n_detected"), int) and isinstance(ttd.get("n"), int), (
            f"{r['system']}: a lane with no p50 must still say how many it detected"
        )
        assert ttd.get("p50_minutes") is None or isinstance(ttd["p50_minutes"], (int, float))
    headline = run.payload["headline_system"]
    lead = next(r for r in rows if r["system"] == headline)
    assert lead["ttd"]["p50_minutes"] is not None, (
        "the headline system must publish a median on a fire arm -- the page's largest "
        "figure is read from it"
    )


def test_the_operating_curve_has_two_points_for_the_headline_system(run: Run) -> None:
    """``readOperating`` drops a series with fewer than two points and ``OperatingCurve``
    says 'not published' only when EVERY series is dropped. A baseline that never reaches
    a median at any budget legitimately has no line; the legend names it as such. The
    headline system's line is what the figure exists to draw, so it must be drawable."""
    rows = _rows(run.payload, "operating_curve")
    headline = run.payload["headline_system"]
    drawable = [r for r in rows if r.get(f"{headline}_p50_minutes") is not None]
    assert len(drawable) > 1, f"{headline} has {len(drawable)} plottable operating points"
    # every system has a column at every budget, null or not -- the reader looks the key up
    for system in bench.SYSTEMS:
        assert all(f"{system}_p50_minutes" in r for r in rows), f"{system} column missing"


def test_the_sensitivity_strip_has_two_points_to_draw(run: Run) -> None:
    """``SensitivityStrip`` says 'not published' below two points. Eleven offsets are
    published; the drawn series is the headline system's. Not all eleven need a value --
    an offset that moves the clock PAST a detection reads as the next slot or a miss
    (test_bench.test_the_offset_shifts_time_to_detection_by_exactly_the_offset), and on
    the synthetic corpus PlumeRank detects at 0.0 min, so the positive offsets fall off."""
    headline = run.payload["headline_system"]
    rows = _rows(run.payload, "sensitivity.rows")
    assert [r["offset_minutes"] for r in rows] == list(bench.SENSITIVITY_OFFSETS_MINUTES)
    drawable = [r for r in rows if r.get(f"{headline}_p50_minutes") is not None]
    assert len(drawable) > 1, f"{headline} has {len(drawable)} plottable sensitivity points"


def test_the_classifier_row_is_quoted_from_the_training_manifest(run: Run) -> None:
    """The precision/recall pair on /results is the training run's, read off disk.

    bench.py has no torch and no val split, so a number here that did not come from
    ``data/tile_manifest.json`` would be the one figure in the document nothing measured.
    """
    manifest = json.loads((run.root / "data" / "tile_manifest.json").read_text(encoding="utf-8"))
    published = run.payload["classifier"]
    assert published["source"] == "data/tile_manifest.json"
    assert published["val_precision"] == manifest["val"]["precision"]
    assert published["val_recall"] == manifest["val"]["recall"]
    assert published["dataset"] == manifest["corpus"]


def test_the_classifier_row_is_null_and_not_invented_without_a_training_manifest(
    tmp_path: pathlib.Path,
) -> None:
    """A box with no training manifest publishes nulls, and /results renders its dashes."""
    facts = bench.classifier_facts(tmp_path)
    assert facts["source"] is None
    assert facts["val_precision"] is None and facts["val_recall"] is None
    assert facts["dataset"] is None


# ---------------------------------------------------------------------------------
# The other two readers of the same document
# ---------------------------------------------------------------------------------


def test_the_live_wall_finds_its_operating_point_in_this_file(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``api/main.py`` reads exactly ``calibration.threshold`` then ``threshold``.

    It refuses to search the document for a number by that name -- a search would happily
    calibrate the wall off an ablation cell -- so a file that publishes the solved
    threshold only inside a per-system row leaves the live pipeline in its degraded state
    while a full benchmark sits on disk beside it.
    """
    from api import main

    monkeypatch.setattr(main, "RESULTS_DIR", run.root / "results")
    value, source = main._calibrated_threshold()
    assert value == pytest.approx(run.payload["calibration"]["threshold"])
    assert source is not None and source.endswith(":calibration.threshold")


def test_the_endpoint_serves_the_bytes_the_page_parses(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three-way agreement, end to end: printed block == committed file == served bytes.

    ``/api/benchmark`` returns the file as bytes with no re-serialisation, so the document
    every assertion above was made against is literally the document the browser parses.
    """
    from fastapi.testclient import TestClient

    from api import main

    monkeypatch.setattr(main, "RESULTS_DIR", run.root / "results")
    with TestClient(main.app) as client:
        response = client.get("/api/benchmark")
    assert response.status_code == 200
    assert response.content == run.raw
    assert run.payload["headline_block"] in run.stdout


# ---------------------------------------------------------------------------------
# The other end of the seam
# ---------------------------------------------------------------------------------


def _code_of(path: pathlib.Path) -> str:
    """``read.ts`` with its comments removed.

    The seam check below is textual, and a textual check over a file whose comments quote
    every key it reads is a check that passes on a file that reads none of them: delete
    ``num(doc, "headline.p50_minutes")`` and leave the JSDoc above it mentioning
    ``headline.p50_minutes`` and the assertion is still green while /results renders a
    dash. Stripping ``//`` and block comments first is what makes the assertion about the
    code. String literals in this file never contain ``//`` or ``/*`` -- they are dotted
    JSON paths and template names -- so a naive strip is exact here and a JS tokeniser
    would be a worse authority than a rule somebody can read.
    """
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end == -1 else end
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def test_read_ts_still_looks_these_names_up(run: Run) -> None:
    """The contract above is only a contract while ``read.ts`` reads the same names.

    Asserted textually, and that is the honest limit of what a pytest suite can say about
    a TypeScript file: every path this file holds bench.py to must appear as a quoted
    string literal in ``web/lib/read.ts``. Together with the tests above, a rename on
    EITHER side of the seam fails -- which is the property that was missing when the page
    rendered three empty panels against a full results file.
    """
    text = _code_of(READ_TS)
    missing = [path for path, _, _ in CONTRACT if f'"{path}"' not in text]
    missing += [key for _, key, _, _ in ROW_CONTRACT if f'"{key}"' not in text]
    assert not missing, f"web/lib/read.ts no longer reads: {missing}"

    # The interpolated ones, in read.ts's own spelling.
    for template in ("${s.key}_p50_minutes", "${headlineSystem}_p50_minutes",
                     "${headlineSystem}_p95_minutes"):
        assert template in text, f"web/lib/read.ts no longer builds `{template}`"


def test_read_ts_never_falls_back_to_a_name_bench_does_not_write(run: Run) -> None:
    """Every document-level path ``read.ts`` accepts must exist in a real payload.

    A multi-path lookup is how the original defect hid: the file looked defensive while
    every one of its spellings missed. A fallback is legitimate ONLY as a second name for
    a quantity the document really carries, so the rule enforced here is that at least one
    path in each lookup resolves -- checked over the union, since a fallback that never
    resolves is dead code and a lookup where none resolves is the bug.
    """
    paths = {path for path, _, _ in CONTRACT}
    # The alternates read.ts accepts beside the published spellings, written out rather
    # than parsed: they are few, and a regex over TypeScript is a worse authority than a
    # list somebody has to edit when they add one.
    alternates = {
        "systems": "baselines",
        "threshold": "calibration.threshold",
        "budget_alerts_per_camera_night": "calibration.target_alerts_per_camera_night",
        "escalation.max_allowed": "escalation.structural_ceiling",
        "exclusions.contact_sheet_url": None,
    }
    for alternate, primary in alternates.items():
        if primary is None:
            # Deliberately unpublished: data/contact_sheet.jpg is committed but is not
            # served by the static site, and a link that 404s beside an exclusion count is
            # worse than no link. `null` renders no link, not an error.
            assert _at(run.payload, alternate) is _ABSENT
            continue
        assert primary in paths, f"{primary} left the contract while {alternate} still falls back"
        assert _at(run.payload, primary) is not _ABSENT

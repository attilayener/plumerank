"""Tests for the claim checker, written the only way a checker can honestly be tested.

Asserting that ``check_claims.py`` passes on this repository proves almost nothing: a
script that returns zero unconditionally passes that test too. So every test here builds a
throwaway repository tree with exactly one deliberate lie in it, and asserts that the one
lie is caught and named. The clean tree is then the control, and the ONLY test where a
pass means something.

The three lies that matter most are the three the build contract found in the real source
-- a symbol named in a docstring, a symbol named in a comment, and a symbol that exists
only as a type annotation. Each is a way for a document to look supported by code that
never runs, and each gets its own test below.

No network, no fixtures on disk, nothing shared between tests.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import check_claims  # noqa: E402


# ---------------------------------------------------------------------------------
# A minimal repository, and the knobs that break exactly one thing in it
# ---------------------------------------------------------------------------------

_SOURCE = '''\
"""A module that calls a few real OpenCV surfaces."""

from __future__ import annotations

import cv2
import numpy as np


def edges(grey: np.ndarray) -> np.ndarray:
    return np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))


def stabilise(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    shift, _response = cv2.phaseCorrelate(a, b)
    return shift
'''

_API = '''\
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"ok": True}
'''

_MAKEFILE = """\
.PHONY: test bench

test:
\t.venv/bin/python -m pytest tests

bench:
\t.venv/bin/python scripts/bench.py
"""

_LOCK = """\
opencv-contrib-python-headless==5.0.0.93
numpy==2.1.3
fastapi==0.115.6
"""

_README = """\
# Toy

The pipeline calls `cv2.Sobel` and `cv2.phaseCorrelate` on every frame.

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | liveness |

Run `make test`. The source lives in `plumerank/toy.py`.
"""


def build_tree(
    tmp_path: pathlib.Path,
    *,
    source: str = _SOURCE,
    readme: str | None = _README,
    api: str | None = _API,
    makefile: str = _MAKEFILE,
    lock: str = _LOCK,
) -> pathlib.Path:
    """A tiny but real repository: real Python, a real FastAPI app, a real Makefile."""
    root = tmp_path
    (root / "plumerank").mkdir(parents=True, exist_ok=True)
    (root / "plumerank" / "toy.py").write_text(source, encoding="utf-8")
    if api is not None:
        (root / "api").mkdir(parents=True, exist_ok=True)
        (root / "api" / "main.py").write_text(api, encoding="utf-8")
    (root / "Makefile").write_text(makefile, encoding="utf-8")
    (root / "requirements.lock").write_text(lock, encoding="utf-8")
    if readme is not None:
        (root / "README.md").write_text(readme, encoding="utf-8")
    return root


def failures(report: check_claims.Report) -> dict[str, check_claims.Check]:
    return {c.name: c for c in report.failed}


def status_of(report: check_claims.Report, name: str) -> str:
    return next(c.status for c in report.checks if c.name == name)


# ---------------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------------


def test_a_clean_tree_passes(tmp_path: pathlib.Path) -> None:
    report = check_claims.run(build_tree(tmp_path))
    assert report.failed == [], report.render()
    assert report.exit_code == 0


def test_every_check_actually_ran_on_the_clean_tree(tmp_path: pathlib.Path) -> None:
    # A clean tree passing is only reassuring if the checks were not all skipped.
    report = check_claims.run(build_tree(tmp_path))
    ran = {c.name for c in report.checks if c.status == check_claims.PASS}
    assert ran == {
        "surface-table-complete",
        "opencv-symbols",
        "routes",
        "dependencies",
        "makefile-targets",
        "claimed-paths",
    }


# ---------------------------------------------------------------------------------
# One lie per test
# ---------------------------------------------------------------------------------


def test_a_claimed_symbol_nothing_calls_is_caught(tmp_path: pathlib.Path) -> None:
    readme = _README + "\nWe refine the peak with `cv2.phaseCorrelateIterative`.\n"
    report = check_claims.run(build_tree(tmp_path, readme=readme))
    miss = failures(report)["opencv-symbols"].misses
    assert any("phaseCorrelateIterative" in m for m in miss), miss
    assert report.exit_code == 1


def test_a_symbol_named_only_in_a_docstring_is_not_a_call_site(tmp_path: pathlib.Path) -> None:
    # The trap the build contract found in escalate.py: the doc spelling appears in the
    # source, but only inside a string the interpreter never executes.
    source = _SOURCE + '\n\ndef documented_only() -> None:\n    """Uses cv2.Laplacian, allegedly."""\n'
    readme = _README + "\nInterior energy comes from `cv2.Laplacian`.\n"
    report = check_claims.run(build_tree(tmp_path, source=source, readme=readme))
    assert any("Laplacian" in m for m in failures(report)["opencv-symbols"].misses)


def test_a_symbol_named_only_in_a_comment_is_not_a_call_site(tmp_path: pathlib.Path) -> None:
    # statistic.py names symbols in comments precisely to say they are NOT used; a
    # checker that counted those would credit the code for a decision not to use them.
    source = _SOURCE + "\n\n# We deliberately do not use cv2.calcOpticalFlowPyrLK here.\n"
    readme = _README + "\nThe fallback uses `cv2.calcOpticalFlowPyrLK`.\n"
    report = check_claims.run(build_tree(tmp_path, source=source, readme=readme))
    assert any("calcOpticalFlowPyrLK" in m for m in failures(report)["opencv-symbols"].misses)


def test_a_type_annotation_is_not_a_call_site(tmp_path: pathlib.Path) -> None:
    # `from __future__ import annotations` means this string is never evaluated, so the
    # symbol it names is never resolved and must not count.
    source = _SOURCE + "\n\n_cache: dict[str, cv2.DISOpticalFlow] = {}\n"
    readme = _README + "\nDrift uses `cv2.DISOpticalFlow_create`.\n"
    report = check_claims.run(build_tree(tmp_path, source=source, readme=readme))
    assert any("DISOpticalFlow_create" in m for m in failures(report)["opencv-symbols"].misses)


def test_the_constructor_spelling_satisfies_the_doc_spelling(tmp_path: pathlib.Path) -> None:
    # `cv::DISOpticalFlow::create` in a document and `cv2.DISOpticalFlow_create` in the
    # source are the same surface; the alias table exists so that agreeing is possible.
    source = _SOURCE + (
        "\n\ndef drift() -> object:\n"
        "    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)\n"
    )
    readme = _README + "\nDrift uses `cv::DISOpticalFlow::create`.\n"
    report = check_claims.run(build_tree(tmp_path, source=source, readme=readme))
    assert status_of(report, "opencv-symbols") == check_claims.PASS, report.render()


def test_a_bound_method_claim_needs_its_constructor(tmp_path: pathlib.Path) -> None:
    # `.forward(` on its own is evidence of nothing -- half the objects in this codebase
    # have a method by some ordinary name. Only readNetFromONNX makes it a dnn claim.
    source = _SOURCE + "\n\ndef go(net: object) -> object:\n    return net.forward()\n"
    readme = _README + "\nThe escalation calls `cv2.dnn.Net.forward`.\n"
    report = check_claims.run(build_tree(tmp_path, source=source, readme=readme))
    assert any("dnn.Net.forward" in m for m in failures(report)["opencv-symbols"].misses)

    source_ok = source + (
        "\n\ndef load(path: str) -> object:\n"
        "    return cv2.dnn.readNetFromONNX(path, cv2.dnn.ENGINE_NEW)\n"
    )
    ok = check_claims.run(build_tree(tmp_path / "ok", source=source_ok, readme=readme))
    assert status_of(ok, "opencv-symbols") == check_claims.PASS, ok.render()


def test_an_ordinary_english_word_in_backticks_is_not_a_claim(tmp_path: pathlib.Path) -> None:
    # `mean` is an attribute of cv2 and also a word. Requiring the cv2. prefix for these
    # is what keeps the checker from banning ordinary sentences.
    readme = _README + "\nS3 divides the interior `mean` by the ring `mean`.\n"
    report = check_claims.run(build_tree(tmp_path, readme=readme))
    assert report.failed == [], report.render()


def test_a_documented_route_the_app_does_not_serve_is_caught(tmp_path: pathlib.Path) -> None:
    readme = _README + "\n| GET | `/api/watchlist` | the ranking |\n"
    report = check_claims.run(build_tree(tmp_path, readme=readme))
    assert any("/api/watchlist" in m for m in failures(report)["routes"].misses)


def test_a_served_route_no_document_mentions_is_caught(tmp_path: pathlib.Path) -> None:
    api = _API + '\n\n@app.get("/api/secret")\ndef secret() -> dict:\n    return {}\n'
    report = check_claims.run(build_tree(tmp_path, api=api))
    assert any("/api/secret" in m for m in failures(report)["routes"].misses)


def test_a_documented_method_the_app_does_not_serve_is_caught(tmp_path: pathlib.Path) -> None:
    readme = _README.replace("| GET | `/health` |", "| POST | `/health` |")
    report = check_claims.run(build_tree(tmp_path, readme=readme))
    assert any("POST" in m and "/health" in m for m in failures(report)["routes"].misses)


def test_documents_that_describe_routes_without_an_app_fail(tmp_path: pathlib.Path) -> None:
    report = check_claims.run(build_tree(tmp_path, api=None))
    assert failures(report)["routes"].detail.startswith("documents describe routes")


def test_a_makefile_target_the_docs_invent_is_caught(tmp_path: pathlib.Path) -> None:
    readme = _README + "\nReproduce the headline with `make reproduce`.\n"
    report = check_claims.run(build_tree(tmp_path, readme=readme))
    assert any("make reproduce" in m for m in failures(report)["makefile-targets"].misses)


def test_a_dependency_named_in_the_docs_but_not_locked_is_caught(tmp_path: pathlib.Path) -> None:
    readme = _README + "\nThe operating curve is fitted with `scipy`.\n"
    report = check_claims.run(build_tree(tmp_path, readme=readme))
    assert any("scipy" in m for m in failures(report)["dependencies"].misses)


def test_a_dependency_named_in_a_code_fence_is_caught(tmp_path: pathlib.Path) -> None:
    readme = _README + "\n```python\nimport pandas as pd\n```\n"
    report = check_claims.run(build_tree(tmp_path, readme=readme))
    assert any("pandas" in m for m in failures(report)["dependencies"].misses)


def test_a_claimed_source_path_that_does_not_exist_is_caught(tmp_path: pathlib.Path) -> None:
    # The exact shape of the failure this project has to avoid: a document describing a
    # file that belongs to a branch which never ran.
    readme = _README + "\nOpenCV 5 is built by `deploy/build-opencv5.sh`.\n"
    report = check_claims.run(build_tree(tmp_path, readme=readme))
    assert any("deploy/build-opencv5.sh" in m for m in failures(report)["claimed-paths"].misses)


def test_an_opencv_call_the_table_has_never_seen_is_caught(tmp_path: pathlib.Path) -> None:
    # Not a documentation failure: a code failure. A new surface nobody classified would
    # otherwise be silently dropped from the count, always in the flattering direction.
    source = _SOURCE + "\n\ndef novel(g: object) -> object:\n    return cv2.pyrMeanShiftFiltering(g, 10, 10)\n"
    report = check_claims.run(build_tree(tmp_path, source=source))
    assert any("pyrMeanShiftFiltering" in m for m in failures(report)["surface-table-complete"].misses)


# ---------------------------------------------------------------------------------
# The count
# ---------------------------------------------------------------------------------


def test_the_count_is_measured_from_the_source_not_predicted(tmp_path: pathlib.Path) -> None:
    base = check_claims.run(build_tree(tmp_path)).count
    assert base.total == 2  # Sobel, phaseCorrelate
    assert base.by_module == {"imgproc": ["Sobel", "phaseCorrelate"]}

    with_flow = check_claims.run(
        build_tree(
            tmp_path / "more",
            source=_SOURCE + "\n\ndef d() -> object:\n    return cv2.DISOpticalFlow_create(1)\n",
        )
    ).count
    # One added call, one added surface, one added module -- and nothing else moves.
    assert with_flow.total == base.total + 1
    assert with_flow.by_module["video"] == ["DISOpticalFlow_create"]


def test_ablating_a_surface_lowers_the_count(tmp_path: pathlib.Path) -> None:
    # The S4/GSOC descope path in miniature: delete the call, and the number the documents
    # are allowed to quote drops the same day.
    source = _SOURCE + "\n\ndef d() -> object:\n    return cv2.DISOpticalFlow_create(1)\n"
    before = check_claims.run(build_tree(tmp_path / "before", source=source)).count.total
    after = check_claims.run(build_tree(tmp_path / "after", source=_SOURCE)).count.total
    assert after == before - 1


def test_the_uncounted_surfaces_are_listed_by_name(tmp_path: pathlib.Path) -> None:
    source = _SOURCE + (
        "\n\ndef overlay(frame: object, out: str) -> None:\n"
        "    cv2.goodFeaturesToTrack(frame, 100, 0.01, 10)\n"
        "    cv2.rectangle(frame, (0, 0), (1, 1), (255, 255, 255))\n"
        "    cv2.putText(frame, 'x', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))\n"
        "    cv2.line(frame, (0, 0), (1, 1), (0, 0, 0))\n"
        "    cv2.imwrite(out, frame)\n"
    )
    report = check_claims.run(build_tree(tmp_path, source=source))
    count = report.count
    assert count.features == ["features::goodFeaturesToTrack"]
    assert set(count.drawing) == {
        "imgproc::line",
        "imgproc::putText",
        "imgproc::rectangle",
        "imgcodecs::imwrite",
    }
    # None of them may reach the defended total, whatever else changes.
    assert count.total == 2
    rendered = report.render()
    assert "goodFeaturesToTrack" in rendered
    assert "DELIBERATELY NOT COUNTED" in rendered


# ---------------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------------


def test_no_documents_skips_the_doc_checks_but_still_counts(tmp_path: pathlib.Path) -> None:
    report = check_claims.run(build_tree(tmp_path, readme=None))
    assert status_of(report, "opencv-symbols") == check_claims.SKIP
    assert report.exit_code == 0
    assert report.count.total == 2


def test_a_run_that_skipped_everything_does_not_print_a_clean_bill(tmp_path: pathlib.Path) -> None:
    """Exit zero is honest here; "every claim is backed by code" would not be.

    With no document on disk the five document checks never ran, and a last line reading
    PASS over them is this script committing the offence it exists to catch. Both the
    printed report and the JSON say so, because the JSON is the committed artefact.
    """
    report = check_claims.run(build_tree(tmp_path, readme=None))
    rendered = report.render()
    assert "every claim on disk is backed by code" not in rendered
    assert "SKIPPED" in rendered
    assert report.to_json()["ok"] is True
    assert report.to_json()["complete"] is False
    assert check_claims.run(build_tree(tmp_path / "documented")).to_json()["complete"] is True


def test_a_machine_readable_report_is_written(tmp_path: pathlib.Path) -> None:
    root = build_tree(tmp_path)
    report = check_claims.run(root)
    path = check_claims.write_json(report, root / "results")
    assert path.name.startswith("claims_") and path.suffix == ".json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["surfaces"]["defended_surfaces_found_today"] == 2
    assert payload["surfaces"]["by_module"] == {"imgproc": ["Sobel", "phaseCorrelate"]}
    assert {c["name"] for c in payload["checks"]} >= {"routes", "dependencies"}


def test_main_exits_non_zero_on_a_miss(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    root = build_tree(tmp_path, readme=_README + "\nWe use `cv2.Laplacian`.\n")
    assert check_claims.main(["--root", str(root), "--no-json"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_exits_zero_on_a_clean_tree(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
    root = build_tree(tmp_path)
    assert check_claims.main(["--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "PASS -- every claim on disk is backed by code" in out
    assert list((root / "results").glob("claims_*.json"))


def test_this_repository_passes_its_own_claim_check() -> None:
    """The real tree. This test is also the hook that puts the checker inside `make test`.

    `make test` runs pytest and nothing else, so without this call the claim checker would
    only ever run when somebody remembered to run it -- which is the failure mode it was
    written to remove. It comes last and it is the least meaningful test in the file on
    its own; it earns its place only because every test above proves the checker can fail.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    report = check_claims.run(root)
    assert report.failed == [], report.render()


def test_a_bound_method_in_another_file_is_not_the_same_object(tmp_path: pathlib.Path) -> None:
    """A repo-global `requires` credits `dnn.Net.forward` to any `.forward(` anywhere.

    That is the count inflating on evidence that is not evidence: the net is built in one
    module and something entirely unrelated has a `forward` method in another. The rule is
    file-local, so this pair must not reach the defended total.
    """
    root = build_tree(tmp_path)
    (root / "plumerank" / "loader.py").write_text(
        "from __future__ import annotations\n\nimport cv2\n\n\n"
        "def load(path: str) -> object:\n"
        "    return cv2.dnn.readNetFromONNX(path)\n",
        encoding="utf-8",
    )
    (root / "plumerank" / "unrelated.py").write_text(
        "from __future__ import annotations\n\n\n"
        "def step(model: object) -> object:\n"
        "    return model.forward()\n",
        encoding="utf-8",
    )
    count = check_claims.run(root).count
    assert "dnn.Net.forward" not in count.by_module.get("dnn", [])
    assert "dnn.Net.forward" in count.absent

    # Move the call into the file that builds the net and the same surface is real.
    (root / "plumerank" / "unrelated.py").unlink()
    (root / "plumerank" / "loader.py").write_text(
        "from __future__ import annotations\n\nimport cv2\n\n\n"
        "def load(path: str) -> object:\n"
        "    net = cv2.dnn.readNetFromONNX(path)\n"
        "    return net.forward()\n",
        encoding="utf-8",
    )
    assert "dnn.Net.forward" in check_claims.run(root).count.by_module["dnn"]


def test_the_english_phrase_make_sure_is_not_a_makefile_target(tmp_path: pathlib.Path) -> None:
    """A checker that fails on a sentence teaches the team to reword sentences.

    "make sure the box is up" is prose. Reading it as a target puts a FAIL next to a
    judge-facing README for a reason that has nothing to do with any claim, which is how a
    gate stops being read.
    """
    readme = _README + "\nBefore judging, make sure the box is up.\n"
    report = check_claims.run(build_tree(tmp_path, readme=readme))
    assert report.failed == [], report.render()


def test_a_target_inside_a_code_fence_is_still_checked(tmp_path: pathlib.Path) -> None:
    # Narrowing to code spans must not create a place to hide an invented target.
    readme = _README + "\n```sh\nmake reproduce\n```\n"
    report = check_claims.run(build_tree(tmp_path, readme=readme))
    assert any("make reproduce" in m for m in failures(report)["makefile-targets"].misses)

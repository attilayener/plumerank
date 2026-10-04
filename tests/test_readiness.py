"""Tests for the submit-time gate. Same method as the claim checker: one lie per tree.

A readiness gate is worth exactly as much as its ability to say NO, so every test below
plants a single specific defect -- a placeholder left in the submission text, an access
key in the tree, a ticked award box, a reproduce block that shows one command and calls it
"the headline" -- and asserts that the gate names that one thing. The clean tree is the
control and is the only place a green result is allowed to mean anything.

There is no network here. The two URL checks take their transport as an argument, and the
tests pass a function that answers from a dict. That seam exists for the tests only: there
is no command-line flag that swaps it, because a switch that stops the gate talking to the
deployed system is the switch that gets left on the night it matters.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import check_submission_readiness as readiness  # noqa: E402

LIVE = "https://plumerank.example"
VIDEO = "https://videos.example/plumerank"
SHA = "0123456789abcdef0123456789abcdef01234567"

BENCH = {
    "arm": "hold",
    "headline": {"p50_minutes": 6.0, "p95_minutes": 19.0, "n": 60, "n_detected": 57},
    "realised_alerts_per_camera_night_uncapped": 2.0,
}

_MAKEFILE = """\
.PHONY: fetch bench test verify

fetch:
\t.venv/bin/python scripts/fetch.py --corpus all

bench:
\t.venv/bin/python scripts/bench.py --arm hold

test:
\t.venv/bin/python -m pytest tests

verify:
\t.venv/bin/python scripts/bench.py --arm micro
"""

_API = """\
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"ok": True}
"""

_README = """\
# Toy

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | liveness |

## Reproduce

Block 1, REPRODUCED HERE -- the micro corpus, three minutes:

```sh
make verify
```

Block 2, PUBLISHED, NOT RECOMPUTED HERE -- the 60-sequence headline, 95 minutes:

```sh
make fetch && make bench
```
"""

_SUBMISSION = """\
# PlumeRank -- Devpost submission

## Optional featured paths

- [ ] Best Use of COOL
- [ ] Agentic Vision with OpenCV 5

## Eligibility

The official terms require entrants to be 18 or older and resident outside the
embargoed territories; we confirmed our eligibility against that text, retrieved
2026-08-29.
"""


def build_tree(
    tmp_path: pathlib.Path,
    *,
    readme: str = _README,
    submission_text: str | None = _SUBMISSION,
    makefile: str = _MAKEFILE,
    bench: dict | None = None,
    extra: dict[str, str] | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """A throwaway repository plus its sibling submission directory."""
    root = tmp_path / "build"
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "plumerank").mkdir(parents=True, exist_ok=True)
    (root / "plumerank" / "__init__.py").write_text("", encoding="utf-8")
    # A real FastAPI app, because check_claims.py SKIPS its route check when there is
    # neither an app nor a documented route -- and a skipped check is what the gate now
    # refuses to read as green. A tree with no API is not a submittable tree.
    (root / "api").mkdir(parents=True, exist_ok=True)
    (root / "api" / "main.py").write_text(_API, encoding="utf-8")
    # The real checker, because "check_claims.py exits zero" is a claim about the real
    # command and a stand-in would be testing the stand-in.
    shutil.copy(_SCRIPTS / "check_claims.py", root / "scripts" / "check_claims.py")
    (root / "Makefile").write_text(makefile, encoding="utf-8")
    (root / "requirements.lock").write_text("numpy==2.1.3\n", encoding="utf-8")
    (root / "README.md").write_text(readme, encoding="utf-8")
    (root / "results" / "bench_2026-10-20.json").write_text(
        json.dumps(bench if bench is not None else BENCH, indent=2), encoding="utf-8"
    )
    submission = tmp_path / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    if submission_text is not None:
        (submission / "submission.md").write_text(submission_text, encoding="utf-8")
    for name, body in (extra or {}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root, submission


def build_info_payload(commit: str = SHA, version: str = "5.0.0.93") -> bytes:
    """The shape api/main.py:build_info actually serves -- version nested under opencv."""
    return json.dumps(
        {
            "service": "plumerank",
            "opencv": {
                "version": version,
                "modules": {"dnn": {"ok": True, "symbols": {}}},
                "build_information": "General configuration for OpenCV " + version,
            },
            "dnn": {"engine_requested": "ENGINE_NEW", "loaded": True},
            "constants": {"sha256": "x", "committed_sha256": None, "matches": None},
            "commit": commit,
            "instance_id": "i-0abc",
            "ami_id": "ami-0def",
            "onnx_sha256": None,
        }
    ).encode()


def fetcher(mapping: dict[str, tuple[int, bytes]]) -> readiness.Fetch:
    def fetch(url: str, timeout: float) -> tuple[int, bytes]:
        if url not in mapping:
            raise AssertionError(f"the gate asked for an unexpected URL: {url}")
        return mapping[url]

    return fetch


def live_mapping(root: pathlib.Path, **overrides: tuple[int, bytes]) -> dict:
    bench_bytes = (root / "results" / "bench_2026-10-20.json").read_bytes()
    mapping = {
        LIVE: (200, b"<html>the wall</html>"),
        f"{LIVE}/build-info": (200, build_info_payload()),
        f"{LIVE}/api/benchmark": (200, bench_bytes),
        VIDEO: (200, b"video"),
    }
    mapping.update(overrides)
    return mapping


def full_config() -> readiness.Config:
    return readiness.Config(live_url=LIVE, video_url=VIDEO)


@pytest.fixture(autouse=True)
def _pin_head(monkeypatch: pytest.MonkeyPatch) -> None:
    # A temporary directory is not a git checkout, and `git init` in a test is a side
    # effect on disk. What is under test is the comparison, not the subprocess.
    monkeypatch.setattr(readiness, "head_sha", lambda root: SHA)


def by_name(report: readiness.Report) -> dict[str, readiness.Check]:
    return {c.name: c for c in report.checks}


# ---------------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------------


def test_a_ready_tree_passes_all_twelve(tmp_path: pathlib.Path) -> None:
    root, submission = build_tree(tmp_path)
    report = readiness.run(root, full_config(), submission, fetcher(live_mapping(root)))
    assert [c.name for c in report.checks if c.status != readiness.PASS] == [], report.render()
    assert len(report.checks) == 12
    assert report.exit_code == 0
    assert report.verdict.startswith("READY")


def test_a_skipped_check_is_never_green(tmp_path: pathlib.Path) -> None:
    root, submission = build_tree(tmp_path)
    # No URLs supplied yet -- the state this runs in for most of the build.
    report = readiness.run(root, readiness.Config(), submission, fetcher({}))
    names = {c.name for c in report.skipped}
    assert names == {"live-url", "build-info", "benchmark-matches", "video-url"}
    assert report.failed == []
    assert report.exit_code == 2, "a skipped check must not exit zero"
    assert "NOT READY" in report.verdict


# ---------------------------------------------------------------------------------
# The live endpoint
# ---------------------------------------------------------------------------------


def test_a_dead_live_url_fails(tmp_path: pathlib.Path) -> None:
    root, _ = build_tree(tmp_path)
    check = readiness.check_live_url(full_config(), fetcher(live_mapping(root, **{LIVE: (502, b"")})))
    assert check.status == readiness.FAIL and "502" in check.detail


def test_build_info_must_carry_this_checkout_s_sha(tmp_path: pathlib.Path) -> None:
    root, _ = build_tree(tmp_path)
    mapping = live_mapping(root, **{f"{LIVE}/build-info": (200, build_info_payload(commit="deadbeef"))})
    check = readiness.check_build_info(root, full_config(), fetcher(mapping))
    assert check.status == readiness.FAIL
    assert any("deadbeef" in e for e in check.evidence), check.evidence


def test_build_info_must_report_opencv_5(tmp_path: pathlib.Path) -> None:
    root, _ = build_tree(tmp_path)
    mapping = live_mapping(root, **{f"{LIVE}/build-info": (200, build_info_payload(version="4.10.0"))})
    check = readiness.check_build_info(root, full_config(), fetcher(mapping))
    assert check.status == readiness.FAIL
    assert any("5." in e for e in check.evidence), check.evidence


def test_build_info_must_actually_name_opencv(tmp_path: pathlib.Path) -> None:
    root, _ = build_tree(tmp_path)
    payload = json.dumps({"version": "5.0.0", "commit": SHA}).encode()
    mapping = live_mapping(root, **{f"{LIVE}/build-info": (200, payload)})
    check = readiness.check_build_info(root, full_config(), fetcher(mapping))
    assert check.status == readiness.FAIL
    assert any("OpenCV" in e for e in check.evidence)


def test_a_video_url_that_404s_fails(tmp_path: pathlib.Path) -> None:
    root, _ = build_tree(tmp_path)
    check = readiness.check_video_url(full_config(), fetcher(live_mapping(root, **{VIDEO: (404, b"")})))
    assert check.status == readiness.FAIL and "404" in check.detail


# ---------------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------------


def test_a_served_headline_that_differs_from_results_is_caught(tmp_path: pathlib.Path) -> None:
    root, _ = build_tree(tmp_path)
    served = dict(BENCH)
    served["headline"] = dict(BENCH["headline"], p50_minutes=4.0)
    mapping = live_mapping(root, **{f"{LIVE}/api/benchmark": (200, json.dumps(served).encode())})
    check = readiness.check_benchmark_matches(root, full_config(), fetcher(mapping))
    assert check.status == readiness.FAIL
    assert any("p50_minutes" in e or "headline" in e for e in check.evidence), check.evidence


def test_reformatting_alone_does_not_fail_the_benchmark_check(tmp_path: pathlib.Path) -> None:
    # Key order and whitespace are not the claim; the headline is.
    root, _ = build_tree(tmp_path)
    served = json.dumps(BENCH, indent=4, sort_keys=True).encode()
    mapping = live_mapping(root, **{f"{LIVE}/api/benchmark": (200, served)})
    check = readiness.check_benchmark_matches(root, full_config(), fetcher(mapping))
    assert check.status == readiness.PASS, check.detail


def test_a_headline_field_absent_from_the_served_payload_is_caught(tmp_path: pathlib.Path) -> None:
    root, _ = build_tree(tmp_path)
    mapping = live_mapping(root, **{f"{LIVE}/api/benchmark": (200, json.dumps({"arm": "hold"}).encode())})
    check = readiness.check_benchmark_matches(root, full_config(), fetcher(mapping))
    assert check.status == readiness.FAIL
    assert any("absent" in e for e in check.evidence)


def test_no_results_file_is_a_failure_not_a_skip(tmp_path: pathlib.Path) -> None:
    root, _ = build_tree(tmp_path)
    (root / "results" / "bench_2026-10-20.json").unlink()
    check = readiness.check_benchmark_matches(root, full_config(), fetcher({}))
    assert check.status == readiness.FAIL


# ---------------------------------------------------------------------------------
# The reproduce path
# ---------------------------------------------------------------------------------


def test_a_mocking_token_on_the_reproduce_path_is_caught(tmp_path: pathlib.Path) -> None:
    flag = "--dry" + "-run"
    makefile = _MAKEFILE.replace(
        "\t.venv/bin/python scripts/bench.py --arm micro",
        f"\t.venv/bin/python scripts/bench.py --arm micro {flag}",
    )
    root, submission = build_tree(tmp_path, makefile=makefile)
    check = readiness.check_reproduce_honest(root, submission)
    assert check.status == readiness.FAIL
    assert any(flag in e for e in check.evidence)


def test_prose_denying_a_mock_is_not_a_mock(tmp_path: pathlib.Path) -> None:
    # An honest README says there is no offline mode. A scanner that failed on that
    # sentence would push the documentation into saying less about its best property.
    readme = _README + "\nThere is no " + "M" + "OCK path and no " + "OFF" + "LINE mode.\n"
    root, submission = build_tree(tmp_path, readme=readme)
    assert readiness.check_reproduce_honest(root, submission).status == readiness.PASS


def test_a_reproduce_block_with_only_make_verify_is_caught(tmp_path: pathlib.Path) -> None:
    # The R8 trap: one command, described as reproducing the published headline.
    readme = "# Toy\n\n## Reproduce the headline\n\n```sh\nmake verify\n```\n"
    root, submission = build_tree(tmp_path, readme=readme)
    check = readiness.check_reproduce_both(root, submission)
    assert check.status == readiness.FAIL
    assert any("make fetch" in e for e in check.evidence)
    assert any("REPRODUCED HERE" in e for e in check.evidence)


def test_both_commands_without_their_labels_is_still_a_failure(tmp_path: pathlib.Path) -> None:
    readme = "# Toy\n\n```sh\nmake verify\nmake fetch && make bench\n```\n"
    root, submission = build_tree(tmp_path, readme=readme)
    check = readiness.check_reproduce_both(root, submission)
    assert check.status == readiness.FAIL
    assert len(check.evidence) == 2  # both labels, neither command


def test_check_claims_failure_propagates(tmp_path: pathlib.Path) -> None:
    root, submission = build_tree(tmp_path)
    (root / "README.md").write_text(
        _README + "\nThe pipeline calls `cv2.phaseCorrelateIterative`.\n", encoding="utf-8"
    )
    check = readiness.check_claims_green(root)
    assert check.status == readiness.FAIL
    assert any("phaseCorrelateIterative" in e for e in check.evidence), check.evidence


def test_check_claims_success_reports_the_surface_count(tmp_path: pathlib.Path) -> None:
    root, _ = build_tree(tmp_path)
    check = readiness.check_claims_green(root)
    assert check.status == readiness.PASS
    assert "SURFACES FOUND TODAY" in check.detail


# ---------------------------------------------------------------------------------
# The tree scans
# ---------------------------------------------------------------------------------


def test_a_placeholder_left_in_the_submission_is_caught(tmp_path: pathlib.Path) -> None:
    token = "NOT" + " " + "YET"
    text = _SUBMISSION + f"\n| Live endpoint | [{token} MINTED] |\n"
    _root, submission = build_tree(tmp_path, submission_text=text)
    check = readiness.check_no_placeholders(submission)
    assert check.status == readiness.FAIL
    assert any(token in e for e in check.evidence)


def test_an_access_key_in_the_tree_is_caught(tmp_path: pathlib.Path) -> None:
    key = "AK" + "IA" + "1234567890ABCDEF"
    root, submission = build_tree(tmp_path, extra={"deploy/notes.md": f"key: {key}\n"})
    check = readiness.check_no_aws_key(root, submission)
    assert check.status == readiness.FAIL
    assert any("deploy/notes.md" in e for e in check.evidence)


def test_prose_naming_the_key_prefix_is_not_a_leaked_key(tmp_path: pathlib.Path) -> None:
    # launch.md explains this very check by name, and test_deploy.py carries the pattern.
    prefix = "AK" + "IA"
    root, submission = build_tree(
        tmp_path, extra={"deploy/launch.md": f"The gate greps for {prefix} to prove none reached git.\n"}
    )
    assert readiness.check_no_aws_key(root, submission).status == readiness.PASS


def test_a_superseded_cost_string_is_caught(tmp_path: pathlib.Path) -> None:
    cost = "$" + "25"
    root, submission = build_tree(tmp_path, extra={"REPORT.md": f"Runs for about {cost}/month.\n"})
    check = readiness.check_no_stale_cost(root, submission)
    assert check.status == readiness.FAIL
    assert any(cost in e for e in check.evidence)


# ---------------------------------------------------------------------------------
# The submission form
# ---------------------------------------------------------------------------------


def test_a_ticked_special_award_box_is_caught(tmp_path: pathlib.Path) -> None:
    text = _SUBMISSION.replace("- [ ] Best Use of COOL", "- [x] Best Use of COOL")
    _root, submission = build_tree(tmp_path, submission_text=text)
    check = readiness.check_special_awards(submission, full_config())
    assert check.status == readiness.FAIL
    assert any("TICKED" in e for e in check.evidence)


def test_an_award_with_no_recorded_box_is_caught(tmp_path: pathlib.Path) -> None:
    # "We did not enter it" has to be a fact on disk, not a memory at 23:30.
    text = _SUBMISSION.replace("- [ ] Agentic Vision with OpenCV 5\n", "")
    _root, submission = build_tree(tmp_path, submission_text=text)
    check = readiness.check_special_awards(submission, full_config())
    assert check.status == readiness.FAIL
    assert any("Agentic Vision" in e for e in check.evidence)


def test_the_eligibility_clause_needs_a_retrieval_date(tmp_path: pathlib.Path) -> None:
    text = _SUBMISSION.replace(", retrieved\n2026-08-29", "")
    _root, submission = build_tree(tmp_path, submission_text=text)
    assert readiness.check_eligibility_clause(submission).status == readiness.FAIL


def test_a_date_in_another_paragraph_does_not_date_the_clause(tmp_path: pathlib.Path) -> None:
    text = (
        "# Submission\n\n- [ ] Best Use of COOL\n- [ ] Agentic Vision with OpenCV 5\n\n"
        "We confirmed our eligibility against the official terms.\n\n"
        "Unrelated paragraph, retrieved 2026-08-29.\n"
    )
    _root, submission = build_tree(tmp_path, submission_text=text)
    assert readiness.check_eligibility_clause(submission).status == readiness.FAIL


def test_a_dated_eligibility_clause_passes(tmp_path: pathlib.Path) -> None:
    _root, submission = build_tree(tmp_path)
    check = readiness.check_eligibility_clause(submission)
    assert check.status == readiness.PASS and "2026-08-29" in check.detail


# ---------------------------------------------------------------------------------
# Exit codes and output
# ---------------------------------------------------------------------------------


def test_a_failure_exits_one_and_the_table_names_it(tmp_path: pathlib.Path) -> None:
    token = "NOT" + " " + "YET"
    root, submission = build_tree(tmp_path, submission_text=_SUBMISSION + f"\n[{token}]\n")
    report = readiness.run(root, full_config(), submission, fetcher(live_mapping(root)))
    assert report.exit_code == 1
    rendered = report.render()
    assert "[FAIL] no-placeholders" in rendered
    assert "NOT READY" in rendered


def test_main_reads_urls_from_a_config_file(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, submission = build_tree(tmp_path)
    (submission / "readiness.json").write_text(
        json.dumps({"live_url": LIVE, "video_url": VIDEO}), encoding="utf-8"
    )
    monkeypatch.setattr(readiness, "http_get", fetcher(live_mapping(root)))
    code = readiness.main(["--root", str(root), "--submission", str(submission)])
    assert code == 0


def test_main_prefers_the_command_line_over_the_config_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale URL in the config file must not be the one the gate goes and asks.

    The transport raises on any URL it was not given, so if the config file won this the
    run would blow up on https://stale.example rather than quietly grading the wrong box.
    """
    root, submission = build_tree(tmp_path)
    (submission / "readiness.json").write_text(
        json.dumps({"live_url": "https://stale.example", "video_url": "https://stale.example/v"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(readiness, "http_get", fetcher(live_mapping(root)))
    code = readiness.main(
        ["--root", str(root), "--submission", str(submission), "--live-url", LIVE, "--video-url", VIDEO]
    )
    assert code == 0


def test_a_url_absent_from_the_config_file_stays_absent(tmp_path: pathlib.Path) -> None:
    # Absent means SKIP, and SKIP means exit 2. It must never default to something.
    _root, submission = build_tree(tmp_path)
    (submission / "readiness.json").write_text(json.dumps({"live_url": LIVE}), encoding="utf-8")
    config = readiness.Config.load(submission / "readiness.json")
    assert config.video_url is None
    assert readiness.check_video_url(config, fetcher({})).status == readiness.SKIP


# ---------------------------------------------------------------------------------
# The four holes the run-1 audit found, each with the tree that walks through it
# ---------------------------------------------------------------------------------


def test_a_lowercase_mock_flag_on_the_reproduce_path_is_caught(tmp_path: pathlib.Path) -> None:
    """A real mocking switch is spelled the way a shell flag is spelled.

    The token list is uppercase so that this scanner does not match its own source, and a
    case-sensitive substring test therefore waves `--mock` straight through -- which is the
    single most likely spelling of the thing the check exists to forbid.
    """
    flag = "--" + "m" + "ock"
    makefile = _MAKEFILE.replace(
        "\t.venv/bin/python scripts/bench.py --arm micro",
        f"\t.venv/bin/python scripts/bench.py --arm micro {flag}",
    )
    root, submission = build_tree(tmp_path, makefile=makefile)
    check = readiness.check_reproduce_honest(root, submission)
    assert check.status == readiness.FAIL
    assert any(flag in e for e in check.evidence), check.evidence


def test_an_ordinary_word_containing_a_token_is_not_a_mocking_flag(tmp_path: pathlib.Path) -> None:
    # Word-bounded, so a path or a target whose name merely contains the letters is safe.
    makefile = _MAKEFILE.replace(
        "\t.venv/bin/python scripts/bench.py --arm micro",
        "\t.venv/bin/python scripts/bench.py --arm micro --out results/smockingham.json",
    )
    root, submission = build_tree(tmp_path, makefile=makefile)
    assert readiness.check_reproduce_honest(root, submission).status == readiness.PASS


def test_an_oversized_file_is_still_scanned_for_a_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returning "" for a big file makes the secret scan report clean because it never looked."""
    key = "AK" + "IA" + "1234567890ABCDEF"
    root, submission = build_tree(tmp_path, extra={"deploy/big.log": f"key: {key}\n" + "x" * 4096})
    monkeypatch.setattr(readiness, "MAX_SCAN_BYTES", 64)
    check = readiness.check_no_aws_key(root, submission)
    assert check.status == readiness.FAIL
    assert any("big.log" in e for e in check.evidence), check.evidence


def test_a_claim_check_that_skipped_everything_is_not_green(tmp_path: pathlib.Path) -> None:
    """Exit zero is not "every claim was checked".

    With no judge-facing document on disk, check_claims.py verifies nothing a judge will
    read and exits zero -- correct in week three, and at submit time the exact shape of a
    green tick over an unexamined tree.
    """
    root, _submission = build_tree(tmp_path)
    (root / "README.md").unlink()
    check = readiness.check_claims_green(root)
    assert check.status == readiness.FAIL
    assert "skipped" in check.detail
    assert any("opencv-symbols" in e for e in check.evidence), check.evidence


def test_a_paragraph_saying_make_verify_reproduces_the_headline_is_caught(
    tmp_path: pathlib.Path,
) -> None:
    """The R8 trap in its surviving form: both labels present, and a sentence undoing them.

    `make verify` runs ~400 micro frames in three minutes. The published number is 60
    held-out sequences and ~95 minutes. A judge who runs the short command expecting the
    headline gets a different p50 and concludes the headline does not reproduce.
    """
    readme = _README + "\nRun `make verify` to reproduce the 60-sequence headline in three minutes.\n"
    root, submission = build_tree(tmp_path, readme=readme)
    check = readiness.check_reproduce_both(root, submission)
    assert check.status == readiness.FAIL
    assert any("conflates" in e or "producing the published" in e for e in check.evidence), check.evidence


def test_a_paragraph_denying_the_conflation_is_exactly_what_should_pass(
    tmp_path: pathlib.Path,
) -> None:
    # The sentence the README is supposed to contain. A gate that failed on it would push
    # the documentation into silence about the distinction it exists to draw.
    readme = _README + (
        "\n`make verify` recomputes the micro corpus only. It does not reproduce the\n"
        "60-sequence headline, and never claims to.\n"
    )
    root, submission = build_tree(tmp_path, readme=readme)
    assert readiness.check_reproduce_both(root, submission).status == readiness.PASS


def test_a_denial_elsewhere_in_the_paragraph_does_not_excuse_a_claim(tmp_path: pathlib.Path) -> None:
    # The reason the denial has to sit in the same SENTENCE: a paragraph that says the
    # honest thing once and the attackable thing once is still a document that says the
    # attackable thing, and that is the sentence a judge quotes back.
    readme = _README + (
        "\nThere is no network access in the harness. Run `make verify` to reproduce the\n"
        "60-sequence headline.\n"
    )
    root, submission = build_tree(tmp_path, readme=readme)
    check = readiness.check_reproduce_both(root, submission)
    assert check.status == readiness.FAIL, check.detail
    assert any("60-sequence headline" in e for e in check.evidence), check.evidence


def test_a_sentence_naming_both_commands_is_the_careful_one_and_passes(
    tmp_path: pathlib.Path,
) -> None:
    """The row `submission/submission.md` actually carries, in its real shape.

    It puts both commands in one sentence and attributes the headline to the 95-minute
    pair. That is the most careful writing in the document, and an earlier draft of this
    check failed it -- which would have taught the team to write the vaguer sentence.
    """
    readme = _README + (
        "\n| Reproduce | `make verify` (3 min, recomputes the micro-corpus metrics) and\n"
        "`make fetch && make bench` (95 min, the one that reproduces the published\n"
        "headline). |\n"
    )
    root, submission = build_tree(tmp_path, readme=readme)
    assert readiness.check_reproduce_both(root, submission).status == readiness.PASS


# ---------------------------------------------------------------------------------
# Audit 2026-10-04 #14 and the two false positives
# ---------------------------------------------------------------------------------


def test_the_fixture_has_the_shape_the_real_server_serves() -> None:
    """The fixture drifted from the server once (flat vs nested) and hid a real failure.

    Compared key-for-key against the live app's own /build-info, so the next drift fails
    here instead of against the deployed box.
    """
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        real = client.get("/build-info").json()
    fixture = json.loads(build_info_payload())
    assert set(fixture) == set(real)
    assert set(fixture["opencv"]) == set(real["opencv"])
    assert isinstance(real["opencv"]["version"], str)


def test_a_flat_build_info_shape_is_not_accepted(tmp_path: pathlib.Path) -> None:
    root, _ = build_tree(tmp_path)
    flat = json.dumps(
        {"opencv_version": "5.0.0", "build_information": "OpenCV", "commit": SHA}
    ).encode()
    mapping = live_mapping(root, **{f"{LIVE}/build-info": (200, flat)})
    check = readiness.check_build_info(root, full_config(), fetcher(mapping))
    assert check.status == readiness.FAIL
    assert any("opencv.version" in e for e in check.evidence), check.evidence


def test_the_grant_amount_is_not_a_superseded_cost(tmp_path: pathlib.Path) -> None:
    grant = "$" + "150"
    root, submission = build_tree(
        tmp_path, extra={"DEMO.md": f"Funded by a {grant} AWS compute grant.\n"}
    )
    assert readiness.check_no_stale_cost(root, submission).status == readiness.PASS


def test_the_static_export_is_not_scanned_for_costs(tmp_path: pathlib.Path) -> None:
    ref = "$" + "15"
    root, submission = build_tree(
        tmp_path, extra={"web/out/judge.txt": f'c:["{ref}","$L12",null]\n'}
    )
    assert readiness.check_no_stale_cost(root, submission).status == readiness.PASS


def test_a_trailing_comment_is_not_a_mocking_flag(tmp_path: pathlib.Path) -> None:
    word = "off" + "line"
    readme = _README + f"\n```bash\nmake test   # {word} suite, no network, no credentials\n```\n"
    root, submission = build_tree(tmp_path, readme=readme)
    assert readiness.check_reproduce_honest(root, submission).status == readiness.PASS


def test_a_mocking_flag_before_a_comment_is_still_caught(tmp_path: pathlib.Path) -> None:
    flag = "--" + "off" + "line"
    readme = _README + f"\n```bash\nmake test {flag}   # quick\n```\n"
    root, submission = build_tree(tmp_path, readme=readme)
    assert readiness.check_reproduce_honest(root, submission).status == readiness.FAIL

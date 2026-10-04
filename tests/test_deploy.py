"""Tests for the four files in ``deploy/`` -- shell and config, so tested as text and as shell.

These four files are the only part of the project that cannot be exercised by importing
it, and they are also the part whose failures show up latest and cost most: a Caddyfile
that routes one path wrongly is invisible until a judge opens the site, and by then the
box is the submission. So the tests here do the two things that are possible without an
EC2 instance.

**Behavioural, where the shell can actually be run.** ``bootstrap.sh`` refuses a wrong
invocation before it installs anything, and refusing is a behaviour with a failure mode:
the tests below run the real script and read the real message. That ordering is fragile
in a specific way -- the log lives under ``/var/log``, so opening it before checking for
root turns "you forgot sudo" into a broken pipe and an ERR trap pointing at the wrong
line -- and ``test_non_root_refusal_is_the_only_message`` is what holds it in place.

**Cross-file agreement, where three files have to name the same path.** ``bootstrap.sh``
creates the directories, ``plumerank.service`` declares the one it is allowed to write,
and the ``Caddyfile`` serves two more. Nothing at runtime reconciles those; systemd
simply refuses to start a unit whose ``ReadWritePaths`` does not exist, and Caddy simply
serves 404 from a root the build never produced. Each of these tests fails if one file
is edited and the others are not, which is the way that class of bug is actually born.

Nothing here needs the network, AWS, or a Linux box.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
DEPLOY = REPO / "deploy"

BOOTSTRAP = DEPLOY / "bootstrap.sh"
CADDYFILE = DEPLOY / "Caddyfile"
UNIT = DEPLOY / "plumerank.service"
LAUNCH = DEPLOY / "launch.md"
ALARMS = DEPLOY / "alarms.sh"

ALL_DEPLOY_FILES = [BOOTSTRAP, CADDYFILE, UNIT, LAUNCH, ALARMS]

# The four routes of build-contract.md W5. Written out rather than imported from
# api/main.py on purpose: this test's job is to catch the two drifting apart, and a test
# that reads the same source as the code under test can never do that.
DOCUMENTED_ROUTES = ("/build-info", "/health", "/api/watchlist", "/api/benchmark")


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _fenced_blocks(markdown: str, language: str) -> list[str]:
    """Every ```<language> block in a Markdown file, in document order."""
    pattern = re.compile(r"^```" + language + r"\n(.*?)^```", re.MULTILINE | re.DOTALL)
    return [m.group(1) for m in pattern.finditer(markdown)]


# ---------------------------------------------------------------------------------
# Style -- rule 7 of the build contract
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_DEPLOY_FILES, ids=lambda p: p.name)
def test_deploy_files_are_ascii(path: pathlib.Path) -> None:
    raw = path.read_bytes()
    offending = [(i, b) for i, b in enumerate(raw) if b > 0x7F]
    assert not offending, (
        f"{path.name} has non-ASCII bytes at offsets "
        f"{[i for i, _ in offending[:5]]} -- build contract rule 7"
    )


# ---------------------------------------------------------------------------------
# bootstrap.sh -- behaviour, not grep
# ---------------------------------------------------------------------------------


def test_bootstrap_parses_as_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(BOOTSTRAP)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_bootstrap_is_executable() -> None:
    # It is passed to run-instances as userdata and also invoked directly on the box.
    assert os.access(BOOTSTRAP, os.X_OK), "deploy/bootstrap.sh has lost its +x bit"


@pytest.mark.skipif(os.geteuid() == 0, reason="this test is about the non-root path")
def test_non_root_refusal_is_the_only_message() -> None:
    """Forgetting sudo must produce one message, and it must be the cause.

    The failure this guards is not the refusal -- that never went missing. It is the
    ordering: ``$LOG_FILE`` is under ``/var/log``, so if the ``exec > >(tee ...)`` is set
    up before the root check, ``tee`` dies of EACCES, the script writes into a closed
    pipe, and the ERR trap fires on the first ``say`` and reports *its* line number. The
    operator's first read is then a wrong line in the middle of the file, and the actual
    "must run as root" is somewhere further down, if it survived at all.
    """
    result = subprocess.run(
        ["bash", str(BOOTSTRAP)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env={**os.environ, "PLUMERANK_REPO": "https://example.invalid/x.git"},
    )
    combined = result.stdout + result.stderr

    assert result.returncode != 0, "a non-root run must fail"
    assert "must run as root" in combined, combined

    # Exactly one FATAL: any second one is the trap firing on collateral damage.
    assert combined.count("FATAL") == 1, (
        "a non-root run emitted more than one FATAL, so the first thing the operator "
        f"reads is not the cause:\n{combined}"
    )
    # Nothing may be reported as a failing line number: the refusal is a decision, not
    # a crash, and a line number here means the ERR trap ran when it should not have.
    assert "line " not in combined, combined
    assert "Permission denied" not in combined, (
        "the log file was opened before the root check -- see this test's docstring"
    )


def test_preconditions_run_before_anything_is_installed() -> None:
    """Ordering, asserted on the file, so a refactor cannot quietly reverse it.

    The behavioural test above only proves the root check comes first on THIS machine,
    where the log happens to be unwritable. On a box where /var/log is writable by the
    invoking user the symptom disappears and the ordering bug does not.
    """
    text = _read(BOOTSTRAP)
    root_check = text.index('[ "$(id -u)" -eq 0 ]')
    log_open = text.index("exec > >(tee -a")
    first_dnf = text.index("dnf install -y")
    assert root_check < log_open < first_dnf


def test_opencv_five_is_a_gate_and_not_a_report() -> None:
    """A 4.x import must end the run, not be printed and stepped over.

    This is the one line in the deploy unit that carries the project's central claim: a
    box that finished provisioning is a box that proved OpenCV 5, so ``/build-info`` is
    reporting a fact rather than a hope.
    """
    text = _read(BOOTSTRAP)
    gate = text[text.index("CV_VERSION=") : text.index("# 5. The frontend")]
    assert "import cv2" in gate
    assert "5.*)" in gate, "the accept arm must match 5.x specifically"
    # The catch-all arm has to die; `say` there would make a 4.x box provision cleanly.
    catch_all = next(
        line for line in gate.splitlines() if line.strip().startswith("*)")
    )
    assert "die " in catch_all, gate


def test_bootstrap_never_mocks_the_judged_capability() -> None:
    # Build contract rule 3. A deploy script is the likeliest place for an
    # "if the pipeline is not ready, serve something" escape hatch to appear.
    text = _read(BOOTSTRAP)
    for forbidden in ("MOCK", "OFFLINE", "--dry-run"):
        assert forbidden not in text, f"{forbidden} in bootstrap.sh violates rule 3"


def test_no_long_lived_credential_anywhere_in_deploy() -> None:
    # the scope record (A15): "no long-lived access key ever lands on the box", and
    # check_submission_readiness.py greps for AKIA. This is the same check, earlier.
    # The literal shape, not the prefix: launch.md legitimately says the word AKIA when
    # it explains what check_submission_readiness.py greps for.
    key_id = re.compile(r"AKIA[0-9A-Z]{16}")
    for path in ALL_DEPLOY_FILES:
        text = _read(path)
        assert not key_id.search(text), f"{path.name} contains an access key id"
        assert "aws_secret_access_key" not in text.lower(), path.name


def test_env_file_is_not_world_readable() -> None:
    """/etc/plumerank.env carries PLUMERANK_REPO, which can be a URL with a token in it.

    systemd reads EnvironmentFile as PID 1, before dropping to the plumerank and caddy
    users, so 0600 costs nothing that 0644 buys.
    """
    text = _read(BOOTSTRAP)
    env_write = text[text.index('cat > "$ENV_FILE"') - 800 : text.index('cat > "$ENV_FILE"')]
    assert "install -m 0600" in env_write, (
        "the env file is written under the default umask; a private-repo clone URL with "
        "a PAT in it would then be readable by every user on the box"
    )


def test_the_run_checks_the_path_a_judge_uses() -> None:
    """A green /health on 127.0.0.1:8000 is not evidence about the public endpoint.

    Caddy can start, validate and still serve nothing -- an export that was never built,
    a matcher that never fires, an evidence directory it cannot traverse. All of those
    look healthy from inside the box, so the run has to make the outside request before
    it is allowed to print DONE.
    """
    text = _read(BOOTSTRAP)
    report = text[text.index("# 8. The report") :]
    before_report = text[: text.index("# 8. The report")]
    assert "http://127.0.0.1${path}" in before_report, (
        "nothing exercises Caddy: the deploy can report DONE with a dead web tier"
    )
    # Both legs, because they break independently.
    assert '"/health:' in before_report and '"/:' in before_report
    assert "DONE in" in report


# ---------------------------------------------------------------------------------
# Caddyfile
# ---------------------------------------------------------------------------------


def test_caddy_proxies_exactly_the_documented_routes() -> None:
    """The @backend matcher and build-contract W5's route table must not drift.

    A fifth route added to api/main.py without a matching prefix here is a 404 that every
    unit test passes and only production shows, because the static file_server catches
    it and answers with the export instead of the API.
    """
    # `.strip()`: since 2026-09-17 the site is an explicit block, so the matcher is indented.
    matcher = next(
        line.strip() for line in _read(CADDYFILE).splitlines() if line.strip().startswith("@backend")
    )
    patterns = matcher.split()[2:]
    for route in DOCUMENTED_ROUTES:
        assert any(
            route == p or (p.endswith("*") and route.startswith(p[:-1]))
            for p in patterns
        ), f"{route} is documented in W5 but the Caddyfile sends it to the static export"
    # And nothing beyond them: a stray /* here would put the whole static site through
    # a single-worker Uvicorn.
    assert "/*" not in patterns and "*" not in patterns, patterns


def test_caddy_never_compresses_the_sse_stream() -> None:
    """encode must come after reverse_proxy in the route, or the live wall stalls.

    Inside a ``route`` block Caddy evaluates in written order, so this is an ordering
    property and not a matcher property: encode written first would wrap the proxy and
    buffer a stream that is only useful unbuffered.
    """
    text = _read(CADDYFILE)
    assert text.index("reverse_proxy") < text.index("encode")
    assert "flush_interval -1" in text, "SSE behind a buffering proxy arrives in clumps"


def test_caddy_evaluates_in_written_order() -> None:
    # `handle` blocks are sorted by matcher specificity; only `route` preserves the
    # order these three handlers are written in, and the catch-all is written last.
    text = _read(CADDYFILE)
    assert re.search(r"^\s*route \{", text, re.MULTILINE), (
        "the handlers are not inside a route block, so Caddy is free to reorder them "
        "and the static catch-all can be sorted ahead of the API"
    )


def test_caddy_falls_back_to_plain_http_before_a_domain_exists() -> None:
    # First boot has no A record, so an unconditional domain would be an ACME loop.
    # `{$VAR:default}` is Caddy's env placeholder, and the default here is the port.
    assert "{$PLUMERANK_DOMAIN::80}" in _read(CADDYFILE)


# ---------------------------------------------------------------------------------
# Cross-file: three files, one filesystem layout
# ---------------------------------------------------------------------------------


def _app_dir() -> str:
    line = next(
        line for line in _read(BOOTSTRAP).splitlines() if line.startswith("APP_DIR=")
    )
    return line.split("=", 1)[1].strip().strip('"')


def test_unit_runs_the_interpreter_bootstrap_actually_builds() -> None:
    app_dir = _app_dir()
    assert f"ExecStart={app_dir}/.venv/bin/uvicorn" in _read(UNIT), (
        f"the unit and bootstrap.sh disagree about where the venv lives ({app_dir})"
    )
    assert 'python3.11 -m venv "$APP_DIR/.venv"' in _read(BOOTSTRAP)


def test_the_one_writable_path_is_created_before_the_unit_starts() -> None:
    """ProtectSystem=strict + a missing ReadWritePaths = a unit systemd refuses to start.

    The refusal is a namespacing error with no application text in it, which is why this
    is worth a test rather than a comment: the symptom names neither directory.
    """
    unit_text = _read(UNIT)
    rw = next(
        line for line in unit_text.splitlines() if line.startswith("ReadWritePaths=")
    ).split("=", 1)[1]
    assert "ProtectSystem=strict" in unit_text
    boot = _read(BOOTSTRAP)
    created = rw.replace(_app_dir(), "$APP_DIR")
    assert f'"{created}"' in boot and (
        f'mkdir -p "{created}"' in boot or 'install -d -o "$APP_USER"' in boot
    ), (
        f"{rw} is the unit's only writable path but bootstrap.sh never creates it"
    )
    assert boot.index(f'"{created}"') < boot.index("systemctl restart plumerank")


def test_caddy_serves_the_directories_bootstrap_produces() -> None:
    caddy_text = _read(CADDYFILE)
    boot = _read(BOOTSTRAP)
    # The static export: bootstrap verifies index.html here, Caddy roots here.
    assert "{$PLUMERANK_ROOT:/opt/plumerank}/web/out" in caddy_text
    assert '"$APP_DIR/web/out/index.html"' in boot
    # The evidence crops: written by rank.draw_evidence, served without touching Python.
    assert "{$PLUMERANK_ROOT:/opt/plumerank}/web/evidence" in caddy_text
    assert "handle_path /evidence/*" in caddy_text, (
        "without handle_path the /evidence prefix is not stripped and every crop 404s"
    )
    # PLUMERANK_ROOT must reach Caddy, and the RPM's unit has no EnvironmentFile.
    assert "caddy.service.d" in boot
    assert "PLUMERANK_ROOT=${APP_DIR}" in boot


def test_the_unit_binds_loopback_and_one_worker() -> None:
    """Both are load-bearing, and both are the kind of flag somebody "fixes" later.

    ``--host 127.0.0.1``: Caddy owns every public byte and this process has no TLS
    configuration, so 0.0.0.0 would be an unencrypted second front door. ``--workers 1``:
    the replay loop is process-wide by design, and a second worker
    means a second wall, a second set of background models and double the CPU credits.
    """
    text = _read(UNIT)
    assert "--host 127.0.0.1" in text
    assert "--workers 1" in text
    assert "Restart=always" in text
    # Comments may name 0.0.0.0 to say why it is wrong; directives may not.
    directives = [
        line for line in text.splitlines() if line and not line.lstrip().startswith("#")
    ]
    assert not any("0.0.0.0" in line for line in directives), directives


# ---------------------------------------------------------------------------------
# launch.md -- the parts that are executable are executed
# ---------------------------------------------------------------------------------


def test_every_bash_block_in_launch_md_parses() -> None:
    blocks = _fenced_blocks(_read(LAUNCH), "bash")
    assert len(blocks) >= 10, f"only {len(blocks)} bash blocks -- did the file shrink?"
    for i, block in enumerate(blocks):
        result = subprocess.run(
            ["bash", "-n"], input=block, capture_output=True, text=True
        )
        assert result.returncode == 0, f"block {i} does not parse:\n{block}\n{result.stderr}"


def test_the_instance_policy_is_the_three_statements_a15_specifies() -> None:
    """the scope record (A15) is a scored responsible-operation claim, so it is asserted.

    Widening this policy is a one-word edit that nothing else in the repo would notice.
    """
    blocks = _fenced_blocks(_read(LAUNCH), "json")
    policies = [
        json.loads(b)
        for b in blocks
        if "PutMetricData" in b or "s3:GetObject" in b
    ]
    assert len(policies) == 1, "expected exactly one inline policy block"
    statements = policies[0]["Statement"]
    assert len(statements) == 3, [s.get("Sid") for s in statements]

    by_action = {json.dumps(s["Action"], sort_keys=True): s for s in statements}
    put = by_action['"cloudwatch:PutMetricData"']
    # PutMetricData has no resource-level ARN, so the namespace condition is the only
    # thing scoping it. Without it the box can write into any namespace in the account.
    assert put["Condition"]["StringEquals"]["cloudwatch:namespace"] == "PlumeRank"

    s3 = by_action['["s3:GetObject", "s3:ListBucket"]']
    assert all(r.startswith("arn:aws:s3:::") for r in s3["Resource"])
    assert any(r.endswith("/corpus/*") for r in s3["Resource"])
    # StringLikeIfExists, not StringLike: s3:prefix is absent from a GetObject request,
    # so a plain StringLike evaluates false on every object read.
    assert "StringLikeIfExists" in s3["Condition"]

    # Read-only on S3 and write-only on one metric namespace. Anything that mutates is
    # outside A15 no matter how convenient it looked at the time.
    flat = json.dumps(policies[0])
    for forbidden in ("s3:PutObject", "s3:Delete", "iam:", "ec2:", '"*"]'):
        assert forbidden not in flat, f"{forbidden} is outside the A15 policy"


def test_the_ami_pin_cannot_be_launched_on_until_it_is_resolved() -> None:
    """A well-formed AMI id nobody verified is worse than an obvious placeholder.

    /build-info echoing the AMI back is only evidence if the id it is compared against
    was actually resolved. An unverified but syntactically valid id launches a box and
    then invites a comparison against a number somebody typed.
    """
    ami_line = next(
        line for line in _read(LAUNCH).splitlines() if line.startswith("export PR_AMI=")
    )
    value = ami_line.split("=", 1)[1].strip()
    if value != "ami-RESOLVE-BEFORE-LAUNCH":
        # Once resolved it must be a real AMI id shape, and the ssm re-resolve command
        # has to stay in the file so the next re-pin is one command.
        assert re.fullmatch(r"ami-[0-9a-f]{8}([0-9a-f]{9})?", value), value
    assert "al2023-ami-kernel-6.1-arm64" in _read(LAUNCH)


def test_ssh_is_never_open_to_the_world() -> None:
    text = _read(LAUNCH)
    for block in _fenced_blocks(text, "bash"):
        if "--port 22" in block:
            assert "0.0.0.0/0" not in block, (
                "port 22 is authorised to 0.0.0.0/0 -- the box is public and unattended "
                "for the whole judging window"
            )
            # Opened and closed in one block, so the close cannot be lost between two
            # copy-pastes on a box nobody is watching.
            assert "revoke-security-group-ingress" in block, block


def test_launch_md_documents_the_wheel_install() -> None:
    # No source build exists, and launch.md is where that is recorded: the aarch64 wheel
    # is what makes `pip install -r requirements.lock` the whole OpenCV 5 install.
    assert not (DEPLOY / "build-opencv5.sh").exists()
    text = _read(LAUNCH)
    assert "no OpenCV source build" in text and "manylinux_2_28_aarch64" in text


def test_the_repo_url_placeholder_fails_loudly_rather_than_guessing() -> None:
    """Until the remote exists, bootstrap must die with the fix rather than clone a guess."""
    boot = _read(BOOTSTRAP)
    if "__SET_ME__" in boot:
        guard = boot[boot.index('if [ ! -d "$APP_DIR/.git" ]') :][:600]
        assert "die " in guard and "PLUMERANK_REPO" in guard
    else:
        default = next(
            line
            for line in boot.splitlines()
            if line.startswith("PLUMERANK_REPO_DEFAULT=")
        )
        assert "OWNER" not in default and "REPO.git" not in default, default


@pytest.mark.skipif(shutil.which("caddy") is None, reason="caddy is not installed here")
def test_caddyfile_validates() -> None:
    # Runs on the box and on any machine that happens to have Caddy; skipped otherwise
    # rather than dropped, because bootstrap.sh runs the same command before it restarts
    # anything and this is the one place the config can be checked off-instance.
    result = subprocess.run(
        ["caddy", "validate", "--adapter", "caddyfile", "--config", str(CADDYFILE)],
        capture_output=True,
        text=True,
        env={**os.environ, "PLUMERANK_DOMAIN": "", "PLUMERANK_ROOT": "/opt/plumerank"},
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------------
# Telemetry and hardening -- audit 2026-10-04 #9 and P2
# ---------------------------------------------------------------------------------


def test_alarms_script_parses_and_is_executable() -> None:
    subprocess.run(["bash", "-n", str(ALARMS)], check=True)
    assert os.access(ALARMS, os.X_OK)


def test_drift_alarm_thresholds_are_the_frozen_budget_band() -> None:
    from plumerank.constants import (
        BUDGET_ALERTS_PER_CAMERA_NIGHT,
        BUDGET_DRIFT_TOLERANCE,
        CLOUDWATCH_NAMESPACE,
    )

    text = _read(ALARMS)
    high = float(re.search(r'^DRIFT_HIGH="([0-9.]+)"', text, re.M).group(1))  # type: ignore[union-attr]
    low = float(re.search(r'^DRIFT_LOW="([0-9.]+)"', text, re.M).group(1))  # type: ignore[union-attr]
    assert high == pytest.approx(BUDGET_ALERTS_PER_CAMERA_NIGHT * (1 + BUDGET_DRIFT_TOLERANCE))
    assert low == pytest.approx(BUDGET_ALERTS_PER_CAMERA_NIGHT * (1 - BUDGET_DRIFT_TOLERANCE))
    assert f'NAMESPACE="{CLOUDWATCH_NAMESPACE}"' in text


def test_every_alarm_watches_a_metric_telemetry_publishes() -> None:
    from plumerank.telemetry import METRICS

    watched = set(re.findall(r"--metric-name (\w+)", _read(ALARMS)))
    assert watched and watched <= set(METRICS)
    assert "Heartbeat" in watched


def test_alarms_are_created_from_the_operator_side_not_the_box() -> None:
    """PutMetricAlarm is not in the instance profile, and must not need to be."""
    assert "put-metric-alarm" not in _read(BOOTSTRAP)
    assert "PutMetricAlarm" not in _read(LAUNCH).split("## 3.")[0]


def test_bootstrap_writes_the_region_into_the_env_file() -> None:
    """Without it boto3 raises NoRegionError and no metric ever lands (audit #9)."""
    text = _read(BOOTSTRAP)
    env_block = text.split('cat > "$ENV_FILE" <<ENVEOF', 1)[1].split("ENVEOF", 1)[0]
    assert "AWS_REGION=" in env_block and "AWS_DEFAULT_REGION=" in env_block


def test_bootstrap_hints_use_sudo_env() -> None:
    """The header says `sudo VAR=` can be dropped by env_reset; no hint may use it."""
    offenders = [
        line for line in _read(BOOTSTRAP).splitlines()
        if re.search(r"\bsudo [A-Z_]+=", line) and "never `sudo VAR=" not in line
    ]
    assert offenders == []


def test_pip_is_pinned_and_npm_never_runs_as_root() -> None:
    text = _read(BOOTSTRAP)
    assert not re.search(r"^[^#]*--upgrade pip", text, re.M)
    assert re.search(r'"pip==\$\{PIP_VERSION\}"', text)
    assert 'runuser -u "$APP_USER"' in text


def test_the_unit_caps_connections_above_the_viewer_cap() -> None:
    from api import main

    unit = _read(UNIT)
    limit = int(re.search(r"--limit-concurrency (\d+)", unit).group(1))  # type: ignore[union-attr]
    assert limit > main.MAX_VIEWERS, "the viewer cap must leave room for /health and /build-info"
    assert re.search(r"^LimitNOFILE=\d+", unit, re.M)


def test_caddy_sends_security_headers_and_hsts_only_over_tls() -> None:
    text = _read(CADDYFILE)
    for header in (
        "Content-Security-Policy",
        "X-Content-Type-Options nosniff",
        "X-Frame-Options DENY",
        "Referrer-Policy",
        "frame-ancestors 'none'",
    ):
        assert header in text, header
    assert "@tls protocol https" in text
    assert 'header @tls Strict-Transport-Security' in text


def test_the_checkout_is_root_owned_and_the_service_owns_only_evidence() -> None:
    """root runs git and pip in the tree on every deploy, so the service account must not
    be able to write it -- an npm install script running as that account could otherwise
    plant code root executes on the next deploy."""
    boot = _read(BOOTSTRAP)
    assert 'chown -R "$APP_USER":"$APP_USER" "$APP_DIR"\n' not in boot
    assert 'chown -R root:root "$APP_DIR"' in boot
    assert 'chown -R "$APP_USER":"$APP_USER" "$APP_DIR/web/evidence"' in boot
    # npm builds in a copy the service account owns, never in the checkout
    assert 'npm --prefix "$BUILD_ROOT/web" ci' in boot
    assert 'npm --prefix "$APP_DIR/web"' not in boot


def test_only_the_box_opts_into_telemetry() -> None:
    assert "PLUMERANK_TELEMETRY=1" in _read(BOOTSTRAP)


def test_the_unit_is_hardened_without_breaking_evidence_reads() -> None:
    unit = _read(UNIT)
    for directive in (
        "ProtectKernelLogs=yes", "ProtectClock=yes", "ProtectHostname=yes",
        "ProtectProc=invisible", "RestrictNamespaces=yes", "SystemCallArchitectures=native",
        "RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX", "CapabilityBoundingSet=",
    ):
        assert directive in unit, directive
    # Caddy (another user) reads the JPEGs the service writes: no restrictive UMask.
    assert "UMask=0027" not in unit and "UMask=0077" not in unit


def test_caddy_serves_the_designed_404_and_strips_trailing_slashes() -> None:
    text = _read(CADDYFILE)
    assert "handle_errors 404" in text and "rewrite * /404.html" in text
    assert "redir @trailing" in text

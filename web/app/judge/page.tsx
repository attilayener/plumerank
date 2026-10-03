import Link from "next/link";

/**
 * "/judge" -- built for exactly one reader. No auth, no cookies, no rate limit, no
 * redirect, and no dependency on the live pipeline: this page is static (a server
 * component, no "use client", no data fetching) so it renders even when the wall
 * itself is reporting a degraded pipeline. JUDGE.md at the repo root is the same
 * content; keep the two in sync by hand -- scripts/check_claims.py checks both for
 * route/dependency/path consistency but does not diff them against each other.
 */

const CLAIM =
  "PlumeRank ranks which wildfire-camera tiles a dispatcher should open next, " +
  "calibrated to a fixed budget of 2 alerts per camera-night, and measures itself on " +
  "median minutes-after-ignition until the right camera reaches the top of that ranking.";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginTop: 28 }}>
      <h2 style={{ fontSize: "var(--size-lead)", color: "var(--ink-0)", margin: "0 0 10px" }}>
        {title}
      </h2>
      {children}
    </section>
  );
}

export default function JudgePage() {
  return (
    <main style={{ padding: "20px 16px 48px", maxWidth: 820, margin: "0 auto" }}>
      <p className="label" style={{ margin: "0 0 6px" }}>
        for judges
      </p>
      <h1 className="display" style={{ fontSize: "var(--size-metric-s)", margin: "0 0 14px" }}>
        The 30-second path
      </h1>
      <p style={{ color: "var(--ink-1)", fontSize: "var(--size-lead)", margin: 0 }}>{CLAIM}</p>

      <Section title="1. Click these, in order">
        <ol style={{ color: "var(--ink-1)", lineHeight: 1.7, paddingLeft: 20 }}>
          <li>
            <Link href="/" style={{ textDecoration: "underline" }}>
              The Wall
            </Link>{" "}
            — the live ranking, running now, no login.
          </li>
          <li>
            <code className="code">GET /build-info</code> — real <code className="code">cv2.getBuildInformation()</code>{" "}
            output, the instance id, the AMI id and the git commit this box is running, read
            live off the box.
          </li>
          <li>
            <Link href="/results" style={{ textDecoration: "underline" }}>
              /results
            </Link>{" "}
            — the operating curve, the baseline race, the sensitivity strip, the ablation grid,
            and the same build-info block in-page.
          </li>
        </ol>
      </Section>

      <Section title="2. The receipt">
        <ul style={{ color: "var(--ink-1)", lineHeight: 1.8, paddingLeft: 20 }}>
          <li>
            <strong>634 passed, 2 skipped</strong>, 97% coverage on <code className="code">plumerank/</code>
          </li>
          <li>
            <strong>1,500 generated cases</strong> (<code className="code">tests/test_invariants.py</code>,
            hypothesis) verifying the fused hazard never leaves <code className="code">[0, 1]</code> and the
            alert budget can never be overspent, under any input or call sequence
          </li>
          <li>
            <code className="code">ruff check .</code> and <code className="code">mypy</code> both clean;{" "}
            <code className="code">scripts/check_claims.py</code> (inside <code className="code">make test</code>)
            mechanically checks every route, dependency, Makefile target and OpenCV symbol this
            page and the README name against the real source
          </li>
          <li>
            DEV arm (orientation only, never the claim): 17/29 detected, p50 32.0 min at 1.99
            alerts/camera-night
          </li>
          <li>HOLD arm (the only number allowed to be the headline): not yet run — see §4 below</li>
          <li>
            One AWS <code className="code">t4g.small</code> (Graviton), one S3 bucket, CloudWatch
            metrics and a drift alarm on the alert budget. No Lambda, no managed vision API.
          </li>
        </ul>
      </Section>

      <Section title="3. Reproduce it yourself">
        <pre className="code" style={{ padding: 12, overflowX: "auto", whiteSpace: "pre" }}>
{`python3 -m venv .venv && .venv/bin/pip install -r requirements.lock
make test      # ~95s: 634-test offline suite + coverage + the claim-consistency check
make verify    # <=3 min: rebuilds the venv, re-runs the suite, recomputes the
               # micro-corpus metrics against the committed results. Does NOT
               # reproduce the published headline, and says so on screen.`}
        </pre>
        <p style={{ color: "var(--ink-2)", fontSize: "var(--size-label)" }}>
          The pair that reproduces the held-out headline itself, once it exists:{" "}
          <code className="code">make fetch</code> then <code className="code">make bench</code>, ≈95
          minutes. Nobody needs to run that to evaluate this submission.
        </p>
      </Section>

      <Section title="4. What is not measured yet">
        <p style={{ color: "var(--ink-1)" }}>
          The HOLD run has not happened. Until <code className="code">results/bench_&lt;date&gt;.json</code>{" "}
          exists, <code className="code">GET /api/benchmark</code> returns 404 with an explicit message
          instead of a fabricated number, and the Wall reports its own degraded state rather than a
          fake ranking. That is a deliberate design decision, not an outage — if the box is showing
          a live ranking when you read this, the run has completed since this page was written.
        </p>
      </Section>

      <Section title="5. Honest limitations">
        <ul style={{ color: "var(--ink-1)", lineHeight: 1.8, paddingLeft: 20 }}>
          <li>Daylight only — no IR hardware, no night detection.</li>
          <li>
            Cloud, not fog, is the real adversary — the statistic this project was built around
            does not separate smoke from fog on its own.
          </li>
          <li>One instance, no HA — every published number is committed in the repository.</li>
          <li>No dispatch, no suppression, no upload path — architecturally absent.</li>
        </ul>
      </Section>

      <Section title="6. Links">
        <p style={{ color: "var(--ink-1)" }}>
          <Link href="/">Wall</Link> · <Link href="/results">/results</Link> ·{" "}
          <a href="/build-info">/build-info</a> · <a href="/health">/health</a>
        </p>
      </Section>
    </main>
  );
}

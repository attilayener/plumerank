"use client";

import { AblationTable } from "@/components/AblationTable";
import { BaselineRace } from "@/components/BaselineRace";
import { BuildInfoSection } from "@/components/BuildInfo";
import { DegradedStrip } from "@/components/DegradedStrip";
import { OperatingCurve } from "@/components/OperatingCurve";
import { ReproduceBlock } from "@/components/ReproduceBlock";
import { SensitivityStrip } from "@/components/SensitivityStrip";
import { TopBar } from "@/components/TopBar";
import { DEV_RUN, DEV_RUN_FILE, DEV_SUMMARY } from "@/lib/devRun";
import { ABSENT, count, minutes, percent, rate } from "@/lib/format";
import { useBenchmark } from "@/lib/useBenchmark";

/**
 * "/results" -- the technical-execution screen.
 *
 * /race and /build are SECTIONS of this page rather than routes of their own: a race
 * between systems IS a result, and a build string IS verification evidence. Six web
 * routes became two, which deleted four data paths and four loading states, and the
 * anchors #race and #build still address the same content.
 *
 * EVERY NUMBER ON THIS PAGE IS FETCHED FROM /api/benchmark AT RUNTIME. None is compiled
 * into the bundle, and there is no bundled fallback copy of the results either --
 * /api/benchmark serves the committed results/bench_<date>.json verbatim from disk, so
 * the page, the endpoint and the repository cannot disagree about the headline, and a
 * frontend that carried its own copy would be the one way to break that.
 */

function Section({
  id,
  title,
  note,
  children,
}: {
  id?: string;
  title: string;
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} style={{ marginTop: 34 }}>
      <h2 style={{ fontSize: "var(--size-lead)", color: "var(--ink-0)", margin: "0 0 4px" }}>{title}</h2>
      {note ? (
        <p style={{ color: "var(--ink-2)", fontSize: "var(--size-label)", margin: "0 0 12px" }}>{note}</p>
      ) : null}
      {children}
    </section>
  );
}

/** The development-set evidence, shown only while /api/benchmark answers 404. Every
 *  number here is read from the committed DEV payload (lib/devRun.ts) and the whole
 *  block is labelled in-sample: the constants were fit on these fires. */
function DevelopmentEvidence() {
  const d = DEV_SUMMARY;
  return (
    <>
      <p className="label" style={{ margin: "26px 0 6px", color: "var(--fog)" }}>
        development set — in-sample, not the headline
      </p>
      <p style={{ color: "var(--ink-1)", margin: "0 0 4px", maxWidth: 760 }}>
        {count(d.detected)} of {count(d.n)} development fires detected, median{" "}
        {minutes(d.p50)} min after ignition. The threshold was solved to{" "}
        {rate(d.calibrationRate)} alerts per camera-night on the fire-free confusion set;
        on the development fires PlumeRank realised {rate(d.devRate)} (uncapped), because a
        fire should alert. Source: <span className="mono">{DEV_RUN_FILE}</span>.
      </p>

      <Section
        id="race"
        title="Systems on one clock (development set)"
        note="PlumeRank against B1 and B2, first through the same per-camera baseline, then as deployed. In-sample."
      >
        <BaselineRace data={DEV_RUN} />
      </Section>

      <Section
        id="operating"
        title="Operating curve (development set)"
        note="Budget against median time-to-detection for every system, from the same development run."
      >
        <OperatingCurve data={DEV_RUN} />
      </Section>

      <Section
        title="Sensitivity to the ignition clock (development set)"
        note="Eleven offsets across ±5 min. The offset moves the ground-truth clock, not the detector."
      >
        <SensitivityStrip data={DEV_RUN} />
      </Section>

      <Section
        id="budget"
        title="Budget calibration"
        note="Solved on the fire-free confusion set, which is the only arm the threshold depends on — the held-out run will solve it on the same nights with the same code."
      >
        <div className="scroll-x card">
          <table>
            <tbody>
              <tr>
                <th scope="row">target</th>
                <td className="mono" style={{ fontWeight: 400 }}>
                  {rate(DEV_RUN.budget.target)} alerts / camera-night
                </td>
              </tr>
              <tr>
                <th scope="row">realised on the confusion set (uncapped)</th>
                <td className="mono" style={{ fontWeight: 400 }}>
                  {rate(DEV_RUN.budget.realised_uncapped)} alerts / camera-night
                </td>
              </tr>
              <tr>
                <th scope="row">solved threshold</th>
                <td className="mono" style={{ fontWeight: 400 }}>{rate(DEV_RUN.budget.threshold, 4)}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p style={{ color: "var(--ink-2)", fontSize: "var(--size-label)", marginTop: 8 }}>
          This is the threshold the live wall runs on until the held-out file exists.
        </p>
      </Section>

      <div style={{ marginTop: 34 }}>
        <ReproduceBlock data={null} />
      </div>
    </>
  );
}

export default function ResultsPage() {
  const { data, loading, error } = useBenchmark();
  // 404 is the designed "not run yet" answer from api/main.py, not an outage.
  const pending = error !== null && / 404\b/.test(error);

  // The denominator, said in full. run.n_sequences is what the bench divided by AFTER the
  // exclusion list was applied, and printing it alone under "read exactly once" reads as
  // sequences that went missing. The manifest's own count is the sum of the two figures
  // bench.py publishes against the same arm, added here rather than read as a third
  // field: a third field could disagree with the two it summarises, and this cannot.
  const measuredOn = data?.run.n_sequences ?? null;
  const excluded = data?.exclusions.count ?? null;
  const manifestWindows = measuredOn === null || excluded === null ? null : measuredOn + excluded;

  return (
    <>
      <TopBar />
      <main style={{ padding: "24px 16px 0", maxWidth: 1000, margin: "0 auto" }}>
        {error && pending ? (
          <section>
            <p className="label" style={{ margin: "0 0 6px" }}>held-out result</p>
            <h1 className="display" style={{ fontSize: "var(--size-metric-s)", margin: "0 0 10px" }}>
              Not run yet
            </h1>
            <p style={{ color: "var(--ink-1)", margin: "0 0 10px", maxWidth: 760 }}>
              The headline is measured once, on held-out fires the system has never seen,
              after every constant is frozen. That run has not happened, so this page shows
              no headline rather than a stand-in. Below is the development run the
              constants were fit on — real, committed, and in-sample.
            </p>
            <DevelopmentEvidence />
          </section>
        ) : error ? (
          <DegradedStrip
            title="results unavailable"
            commit={null}
            detail={`/api/benchmark did not answer (${error}). Nothing is rendered from a bundled copy.`}
          />
        ) : null}

        {loading && !data ? (
          <p style={{ color: "var(--ink-2)" }}>reading /api/benchmark ...</p>
        ) : null}

        {data ? (
          <>
            <header>
              <p className="label">
                median time-to-detection, {data.run.arm ?? ABSENT} arm,{" "}
                {count(measuredOn)} of the {count(manifestWindows)} sequences the manifest
                names, read exactly once after the constants were frozen —{" "}
                {count(excluded)} excluded by the pre-stated criteria below
              </p>
              <div style={{ display: "flex", gap: 34, alignItems: "flex-end", flexWrap: "wrap" }}>
                <span>
                  <span className="display" style={{ fontSize: "var(--size-metric-l)" }}>
                    {minutes(data.headline.p50_minutes)}
                  </span>
                  <span className="label" style={{ marginLeft: 8 }}>min p50</span>
                </span>
                <span>
                  <span className="display" style={{ fontSize: "var(--size-metric-m)", color: "var(--ink-1)" }}>
                    {minutes(data.headline.p95_minutes)}
                  </span>
                  <span className="label" style={{ marginLeft: 8 }}>min p95</span>
                </span>
                <span>
                  <span className="display" style={{ fontSize: "var(--size-metric-s)", color: "var(--ink-1)" }}>
                    {count(data.headline.n_missed)}
                  </span>
                  <span className="label" style={{ marginLeft: 8 }}>missed</span>
                </span>
              </div>
              <p style={{ color: "var(--ink-2)", fontSize: "var(--size-label)", marginTop: 8 }}>
                at {rate(data.budget.target, 1)} alerts per camera-night. A missed sequence
                counts as +infinity, never dropped and never imputed, so the miss count is
                published next to every percentile rather than under it.
              </p>
            </header>

            <Section
              title="Sensitivity to the ignition clock"
              note="Eleven offsets across ±5 min. The offset moves the ground-truth clock, not the detector."
            >
              <SensitivityStrip data={data} />
            </Section>

            <Section
              id="race"
              title="Systems on one clock"
              note="PlumeRank against B1 (the same CNN on every frame) and B2 (frame differencing), first through the same per-camera baseline, then as deployed. Every threshold is solved to the same budget on the same confusion set."
            >
              <BaselineRace data={data} />
            </Section>

            <Section
              id="operating"
              title="Operating curve"
              note={
                "Budget against median time-to-detection for every system. The operating " +
                "point the project is calibrated to is the only annotated point; its value " +
                "is drawn from the results file, not typed here."
              }
            >
              <OperatingCurve data={data} />
            </Section>

            <Section title="Per system">
              <div className="scroll-x card">
                <table>
                  <thead>
                    <tr>
                      <th>system</th>
                      <th>p50 (min)</th>
                      <th>p95 (min)</th>
                      <th>missed</th>
                      <th>alerts / camera-night (confusion set)</th>
                      <th>realised on this arm</th>
                      <th>model calls / camera-night</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.systems.length === 0 ? (
                      <tr>
                        <td colSpan={7} style={{ color: "var(--ink-3)" }}>
                          per-system results not published yet — it appears here after the held-out run
                        </td>
                      </tr>
                    ) : (
                      data.systems.map((s) => (
                        <tr key={s.key}>
                          <td>{s.label}</td>
                          <td className="mono" style={{ fontWeight: 400 }}>{minutes(s.p50_minutes)}</td>
                          <td className="mono" style={{ fontWeight: 400 }}>{minutes(s.p95_minutes)}</td>
                          <td className="mono" style={{ fontWeight: 400 }}>{count(s.n_missed)}</td>
                          <td className="mono" style={{ fontWeight: 400 }}>{rate(s.alerts_per_camera_night)}</td>
                          <td className="mono" style={{ fontWeight: 400 }}>{rate(s.arm_alerts_per_camera_night)}</td>
                          <td className="mono" style={{ fontWeight: 400 }}>
                            {rate(s.model_invocations_per_camera_night, 1)}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </Section>

            <Section
              id="budget"
              title="Budget calibration"
              note="The threshold is solved on the confusion set until the realised rate hits the target."
            >
              <div className="scroll-x card">
                <table>
                  <tbody>
                    <tr>
                      <th scope="row">target</th>
                      <td className="mono" style={{ fontWeight: 400 }}>
                        {rate(data.budget.target)} alerts / camera-night
                      </td>
                    </tr>
                    <tr>
                      <th scope="row">realised (uncapped)</th>
                      <td className="mono" style={{ fontWeight: 400 }}>
                        {rate(data.budget.realised_uncapped)} alerts / camera-night
                      </td>
                    </tr>
                    <tr>
                      <th scope="row">solved threshold</th>
                      <td className="mono" style={{ fontWeight: 400 }}>{rate(data.budget.threshold, 4)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p style={{ color: "var(--ink-2)", fontSize: "var(--size-label)", marginTop: 8 }}>
                Uncapped, and labelled: the realised rate counts threshold crossings without
                the per-camera budget cap, because that is the quantity the threshold was
                solved on — a capped count makes every threshold satisfy the target and the
                solver degenerates. The budget meter on the wall shows the capped one, and
                says so there too.
              </p>
            </Section>

            <Section
              title="Escalation and cost"
              note="One model, two uses. The learned component is identical in PlumeRank and in B1, which is what makes the comparison a controlled experiment rather than a contest between two systems that differ in several ways at once."
            >
              <div className="scroll-x card">
                <table>
                  <tbody>
                    <tr>
                      <th scope="row">escalation fraction</th>
                      <td className="mono" style={{ fontWeight: 400 }}>
                        {percent(data.escalation.fraction, 2)}
                        {data.escalation.max === null || data.escalation.fraction === null
                          ? ""
                          : data.escalation.fraction <= data.escalation.max
                            ? ` (within the ${percent(data.escalation.max, 0)} ceiling)`
                            : ` (above the ${percent(data.escalation.max, 0)} ceiling)`}
                      </td>
                    </tr>
                    <tr>
                      <th scope="row">tile classifier, val precision / recall</th>
                      <td className="mono" style={{ fontWeight: 400 }}>
                        {rate(data.classifier.precision)} / {rate(data.classifier.recall)}
                        {data.classifier.dataset ? ` on ${data.classifier.dataset}` : ""}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Section>

            <Section
              title="Ablation"
              note="{MOG2, LSBP, GSOC} × {S4 on/off} × {escalation on/off}, on the development arm — a separate run, quoted verbatim by the held-out file."
            >
              <AblationTable data={data} />
            </Section>

            <Section
              title="Exclusions"
              note="Windows removed before the run, with the reason for each. Published rather than quietly dropped."
            >
              <p style={{ margin: "0 0 8px" }}>
                <span className="mono num" style={{ fontSize: "var(--size-metric-s)", color: "var(--ink-0)" }}>
                  {count(data.exclusions.count)}
                </span>{" "}
                <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
                  windows excluded
                </span>
                {data.exclusions.contact_sheet_url ? (
                  <>
                    {" - "}
                    <a href={data.exclusions.contact_sheet_url}>the contact sheet they were eyeballed on</a>
                  </>
                ) : null}
              </p>
              {data.exclusions.items.length > 0 ? (
                <div className="scroll-x card">
                  <table>
                    <thead>
                      <tr>
                        <th>sequence</th>
                        <th>reason</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.exclusions.items.map((x) => (
                        <tr key={x.sequence_id}>
                          <td className="mono" style={{ fontWeight: 400 }}>{x.sequence_id}</td>
                          <td>{x.reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </Section>

            <div style={{ marginTop: 34 }}>
              <ReproduceBlock data={data} />
            </div>
          </>
        ) : null}

        {/* Mounted only once /api/benchmark has settled, either way. Drawn from first
            paint, these sections sat under a one-line "reading" caption and were shoved
            down by whatever replaced it -- the results, or the degraded strip -- which
            is a layout shift a reader sees (CLS 0.11-0.15 on this route). Content that
            appears is not a shift; content that moves is. */}
        {!loading ? (
          <>
            <div style={{ marginTop: 34 }}>
              <BuildInfoSection />
            </div>

            <div style={{ marginBottom: 12 }} />
          </>
        ) : null}
      </main>
    </>
  );
}

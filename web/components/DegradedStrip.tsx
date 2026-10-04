import { sha } from "@/lib/format";

/** api/main.py prefixes its reason with the same words this strip leads with, and its
 *  "no calibrated threshold" reason names an operator remedy (`make bench` "on the
 *  CONFUSE arm") that is wrong -- only the held-out run writes the results file the
 *  threshold is read from. Both are rewritten here for a judge, not an operator. */
export function humaneReason(detail: string | null | undefined): string | null {
  if (!detail) return null;
  const reason = detail.replace(/^live pipeline unavailable\s*-+\s*/i, "");
  if (/no calibrated threshold/i.test(reason)) {
    return (
      "the live ranking needs the alert threshold solved by the held-out benchmark run, " +
      "and that run has not been published yet. Nothing is shown in its place."
    );
  }
  return reason;
}

/**
 * State 7 of the seven designed states: the live pipeline is unavailable and the page
 * is showing the committed results instead.
 *
 * The strip names the file and carries the commit sha when a file is being shown, because the failure this
 * component exists to prevent is not "the page looks broken" -- it is "the page looks
 * fine and the numbers are from last week". Stale data is allowed on screen exactly
 * once it has said so.
 */
export function DegradedStrip({
  title = "live pipeline unavailable",
  commit,
  file,
  detail,
}: {
  title?: string;
  commit: string | null;
  /** The committed results file actually being shown, when there is one. */
  file?: string | null;
  detail?: string | null;
}) {
  return (
    <div className="degraded" role="status">
      {title}
      {file ? (
        <>
          {" "}-- showing the committed results from{" "}
          <span className="mono" style={{ fontWeight: 400 }}>results/{file}</span>
          {commit ? (
            <>
              {" "}at commit <span className="mono" style={{ fontWeight: 400 }}>{sha(commit)}</span>
            </>
          ) : null}
        </>
      ) : null}
      {detail ? <div style={{ color: "var(--ink-2)", marginTop: 4 }}>{detail}</div> : null}
    </div>
  );
}

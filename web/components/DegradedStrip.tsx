import { sha } from "@/lib/format";

/**
 * State 7 of the seven designed states: the live pipeline is unavailable and the page
 * is showing the committed results instead.
 *
 * The strip names the file and carries the commit sha, because the failure this
 * component exists to prevent is not "the page looks broken" -- it is "the page looks
 * fine and the numbers are from last week". Stale data is allowed on screen exactly
 * once it has said so.
 */
export function DegradedStrip({
  commit,
  detail,
}: {
  commit: string | null;
  detail?: string | null;
}) {
  return (
    <div className="degraded" role="status">
      live pipeline unavailable -- showing the committed results from
      {" "}
      <span className="mono" style={{ fontWeight: 400 }}>results/bench_&lt;date&gt;.json</span>
      {commit ? (
        <>
          {" "}at commit <span className="mono" style={{ fontWeight: 400 }}>{sha(commit)}</span>
        </>
      ) : null}
      {detail ? <div style={{ color: "var(--ink-2)", marginTop: 4 }}>{detail}</div> : null}
    </div>
  );
}

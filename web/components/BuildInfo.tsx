"use client";

import { useEffect, useState } from "react";
import { curlCommand, fetchJson, ROUTES } from "@/lib/api";
import { CopyButton } from "./CopyButton";
import { ABSENT, sha } from "@/lib/format";
import type { BuildInfo as BuildInfoDoc } from "@/lib/types";

/**
 * /build-info, rendered in the browser -- and the reason it is a SECTION of /results
 * rather than a route of its own is that a build string IS verification evidence, so it
 * belongs beside the numbers it verifies.
 *
 * This section turns an ASSERTED OpenCV 5 integration into one a judge checks. The
 * copy-the-curl button matters more than the rendering does: the honest version of
 * "we run OpenCV 5" is a reader hitting the endpoint themselves and reading
 * cv2.getBuildInformation() verbatim, not a version string we typed into a page. So the
 * page shows both, and the raw dump is the same bytes the API returned.
 *
 * The three key spellings below -- instance_id, ami_id, commit -- are fixed by the
 * cross-unit contract and are checked against deploy/launch.md's pinned AMI by hand.
 * Rendering `amiId` as absent because the server spelled it differently would break a
 * written verification step silently, so the names are not paraphrased anywhere.
 */
export function BuildInfoSection() {
  const [doc, setDoc] = useState<BuildInfoDoc | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [origin, setOrigin] = useState<string>("");

  useEffect(() => {
    setOrigin(window.location.origin);
    const controller = new AbortController();
    fetchJson<BuildInfoDoc>(ROUTES.buildInfo, controller.signal)
      .then(setDoc)
      .catch((e: unknown) => {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : String(e));
      });
    return () => controller.abort();
  }, []);

  const curl = curlCommand(origin);

  // While /build-info is in flight the table is drawn with its fixed keys and ABSENT
  // values, so the section has its final height from first paint: swapping a one-line
  // "reading" caption for a seven-row table pushed everything below it 360 px (CLS 0.11
  // on /results). The values are ABSENT, not guessed -- nothing here is a placeholder
  // that could be mistaken for a real build string.
  const fields: [string, string][] = doc
    ? [
        ["OpenCV", String(doc.opencv?.version ?? ABSENT)],
        [
          "DNN engine",
          doc.dnn?.engine_selected
            ? `${doc.dnn.engine_selected}${doc.dnn.loaded ? ", model loaded" : ", model NOT loaded"}`
            : String(doc.dnn?.detail ?? ABSENT),
        ],
        ["commit", sha(typeof doc.commit === "string" ? doc.commit : null, 40)],
        ["instance_id", String(doc.instance_id ?? "null (not on an EC2 instance)")],
        ["ami_id", String(doc.ami_id ?? "null (not on an EC2 instance)")],
        ["onnx sha256", sha(typeof doc.onnx_sha256 === "string" ? doc.onnx_sha256 : null, 16)],
        ["constants sha256", sha(typeof doc.constants?.sha256 === "string" ? doc.constants.sha256 : null, 16)],
      ]
    : ["OpenCV", "DNN engine", "commit", "instance_id", "ami_id", "onnx sha256", "constants sha256"].map(
        (k): [string, string] => [k, ABSENT],
      );

  return (
    <section id="build">
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
        <h2 style={{ fontSize: "var(--size-lead)", color: "var(--ink-0)", margin: "0 0 4px" }}>
          Build information
        </h2>
        <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
          {doc || error
            ? "verify it without leaving the page, or leave the page and verify it anyway"
            : `reading ${ROUTES.buildInfo} ...`}
        </span>
      </div>

      <div style={{ display: "flex", gap: 10, alignItems: "center", margin: "10px 0" }}>
        <code className="code" style={{ flex: 1, minWidth: 0 }}>{curl}</code>
        <CopyButton text={curl} label="copy the curl" />
      </div>

      {error ? (
        <div className="degraded">
          {ROUTES.buildInfo} did not answer ({error}). Nothing is rendered from cache:
          a build string is only evidence if it came from the box that is serving this page.
        </div>
      ) : (
        <>
          <div className="scroll-x card">
            <table>
              <tbody>
                {fields.map(([k, v]) => (
                  <tr key={k}>
                    <th scope="row" style={{ borderBottom: "1px solid var(--hairline)" }}>{k}</th>
                    <td className="mono" style={{ fontWeight: 400 }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* While loading, the badge line and the summary keep their slots (empty / the
              same summary text) so the page does not move when the document lands. */}
          {!doc ? (
            <p style={{ marginTop: 10, fontSize: "var(--size-label)" }}>
              <span className="badge" style={{ visibility: "hidden" }}>dnn present</span>
            </p>
          ) : null}
          {doc?.opencv?.modules ? (
            <p style={{ marginTop: 10, fontSize: "var(--size-label)" }}>
              {Object.entries(doc.opencv.modules).map(([mod, probe]) => (
                <span key={mod} className="badge" style={{ marginRight: 8, color: probe.ok ? "var(--ok)" : "var(--alert)" }}>
                  {mod} {probe.ok ? "present" : "MISSING"}
                </span>
              ))}
            </p>
          ) : null}

          <details style={{ marginTop: 12 }}>
            <summary style={{ cursor: "pointer", color: "var(--signal)", fontSize: "var(--size-label)" }}>
              cv2.getBuildInformation(), verbatim
            </summary>
            <pre
              className="code"
              style={{ maxHeight: 320, overflow: "auto", marginTop: 8, whiteSpace: "pre" }}
            >
              {doc
                ? String(doc.opencv?.build_information ?? "not published by /build-info")
                : `reading ${ROUTES.buildInfo} ...`}
            </pre>
          </details>
        </>
      )}
    </section>
  );
}

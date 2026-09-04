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

  const fields: [string, string][] = doc
    ? [
        ["OpenCV", String(doc.opencv_version ?? ABSENT)],
        ["DNN engine", String(doc.dnn_engine ?? ABSENT)],
        ["commit", sha(typeof doc.commit === "string" ? doc.commit : null, 40)],
        ["instance_id", String(doc.instance_id ?? "null (not on an EC2 instance)")],
        ["ami_id", String(doc.ami_id ?? "null (not on an EC2 instance)")],
        ["onnx sha256", sha(typeof doc.onnx_sha256 === "string" ? doc.onnx_sha256 : null, 16)],
        ["constants sha256", sha(typeof doc.constants_sha256 === "string" ? doc.constants_sha256 : null, 16)],
      ]
    : [];

  return (
    <section id="build">
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
        <h2 style={{ fontSize: "var(--size-lead)", color: "var(--ink-0)", margin: "0 0 4px" }}>
          Build information
        </h2>
        <span className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
          verify it without leaving the page, or leave the page and verify it anyway
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
      ) : !doc ? (
        <p style={{ color: "var(--ink-2)" }}>reading {ROUTES.buildInfo} ...</p>
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

          {doc.modules ? (
            <p style={{ marginTop: 10, fontSize: "var(--size-label)" }}>
              {Object.entries(doc.modules).map(([mod, ok]) => (
                <span key={mod} className="badge" style={{ marginRight: 8, color: ok ? "var(--ok)" : "var(--alert)" }}>
                  {mod} {ok ? "present" : "MISSING"}
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
              {String(doc.build_information ?? "not published by /build-info")}
            </pre>
          </details>
        </>
      )}
    </section>
  );
}

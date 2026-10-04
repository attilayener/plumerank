"use client";

import { useEffect, useRef } from "react";
import { apiUrl } from "@/lib/api";
import type { WatchRow } from "@/lib/types";

/**
 * The escalated crop at 2x, and the single most explanatory object in the product.
 *
 * The overlay -- a pinned foot line in --overlay-anchor solid, a climbing top line in
 * --overlay-growth dashed, and the labelled pixel gap between them -- is drawn
 * SERVER-SIDE into the JPEG by rank.draw_evidence (cv2.rectangle / cv2.line /
 * cv2.putText). This component lays out and captions an image and does no geometry at
 * all, which removes a whole class of source-pixel-to-layout-pixel bugs: there is no
 * scale factor here to get wrong, and the picture a judge sees is byte-identical to
 * the one on disk that a test can open.
 *
 * It is also the ONLY modal in the product.
 */
export function EvidenceCard({ row, onClose }: { row: WatchRow; onClose: () => void }) {
  const closeButton = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Move focus into the dialog and put it back on the thumbnail afterwards. Without
  // this, aria-modal is a promise the page does not keep: a keyboard user who opens the
  // crop has focus left on the row behind the scrim, so the next Tab walks the watch
  // list under an overlay they cannot see past. The evidence modal IS the product's
  // interactive surface, so it is the one place this cannot be approximated.
  useEffect(() => {
    const returnTo = document.activeElement as HTMLElement | null;
    closeButton.current?.focus();
    return () => returnTo?.focus?.();
  }, []);

  const src = row.evidence_url ?? row.frame_url ?? null;

  return (
    <div
      className="modal-scrim"
      role="dialog"
      aria-modal="true"
      aria-label={`evidence for ${row.camera_id}`}
      onClick={onClose}
    >
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            padding: "12px 16px",
            borderBottom: "1px solid var(--hairline)",
          }}
        >
          <span className="mono num" style={{ color: "var(--ink-0)", fontSize: "var(--size-metric-s)" }}>
            {row.rank}
          </span>
          <span>
            <div style={{ color: "var(--ink-0)", fontWeight: 600 }}>{row.camera_id}</div>
            <div className="label" style={{ textTransform: "none", letterSpacing: 0 }}>
              hazard {row.hazard.toFixed(3)}
              {row.escalated ? " — escalated to the tile classifier" : ""}
            </div>
          </span>
          <button
            ref={closeButton}
            className="btn"
            onClick={onClose}
            style={{ marginLeft: "auto" }}
            type="button"
          >
            close
          </button>
        </div>

        <div style={{ padding: 16 }}>
          {src ? (
            <img
              src={apiUrl(src)}
              alt={`evidence crop for ${row.camera_id} with the server-drawn anchoring overlay`}
              style={{
                display: "block",
                width: "100%",
                maxWidth: 860,
                imageRendering: "pixelated",   /* a 2x crop of a 480x270 frame: show the pixels rather than invent smooth ones */
                border: "1px solid var(--alert)",
                borderRadius: 2,
              }}
            />
          ) : (
            <div className="degraded">
              this row published no evidence crop. The overlay is drawn server-side into
              the JPEG, so there is nothing the browser could draw instead.
            </div>
          )}

          <dl
            style={{
              display: "grid",
              gridTemplateColumns: "auto 1fr",
              gap: "4px 16px",
              margin: "16px 0 0",
              fontSize: "var(--size-label)",
            }}
          >
            <dt style={{ color: "var(--overlay-anchor)" }}>solid line</dt>
            <dd style={{ margin: 0, color: "var(--ink-1)" }}>
              S1 — base anchoring: how tied the component&apos;s foot is to the ground.
            </dd>
            <dt style={{ color: "var(--overlay-growth)" }}>dashed line</dt>
            <dd style={{ margin: 0, color: "var(--ink-1)" }}>
              S2 — growth anisotropy: whether the top climbs faster than the sides spread.
            </dd>
            <dt style={{ color: "var(--ink-3)" }}>S3</dt>
            <dd style={{ margin: 0, color: "var(--ink-1)" }}>
              diffuse-boundary contrast, {row.s3.toFixed(3)}. Measured on the development
              set, none of the three separates fire on its own (AUC 0.53, 0.50 and 0.44);
              the ranking comes from fusing them and scoring each camera against its own
              recent history.
            </dd>
          </dl>
        </div>
      </div>
    </div>
  );
}

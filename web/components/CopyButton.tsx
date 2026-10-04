"use client";

import { useState } from "react";

/**
 * Copy one exact string, and fail visibly if it cannot.
 *
 * navigator.clipboard only exists on a secure origin, and the box serves plain HTTP until
 * a domain and TLS exist. So the async API is tried first, then the legacy
 * execCommand("copy") on a hidden textarea -- which still works on plain HTTP inside a
 * user gesture -- and only if both fail does the button say so, with the command text
 * selected for the judge to copy by hand. A silent no-op would be the worst outcome: a
 * judge pastes yesterday's clipboard into a terminal and blames the command.
 */

function legacyCopy(text: string): boolean {
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.top = "-1000px";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  document.body.removeChild(area);
  return ok;
}

export function CopyButton({
  text,
  label = "copy",
  targetId,
}: {
  text: string;
  label?: string;
  /** Element whose text is selected when every copy route is blocked. */
  targetId?: string;
}) {
  const [note, setNote] = useState<string | null>(null);

  async function copy() {
    let ok = false;
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        ok = true;
      } catch {
        ok = false;
      }
    }
    if (!ok) ok = legacyCopy(text);
    if (!ok && targetId) {
      const el = document.getElementById(targetId);
      const sel = window.getSelection();
      if (el && sel) {
        const range = document.createRange();
        range.selectNodeContents(el);
        sel.removeAllRanges();
        sel.addRange(range);
      }
    }
    setNote(ok ? "copied" : "selected — press Ctrl/⌘+C");
    window.setTimeout(() => setNote(null), 2500);
  }

  return (
    <button className="btn" onClick={copy} type="button" aria-live="polite">
      {note ?? label}
    </button>
  );
}

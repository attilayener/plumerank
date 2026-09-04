"use client";

import { useState } from "react";

/**
 * Copy one exact string. The entire interactive surface of this product is the
 * evidence modal, these buttons and scrolling, so the few surviving interactions have
 * to be flawless -- which here means failing visibly.
 *
 * navigator.clipboard is unavailable on a plain-HTTP origin, and the box serves plain
 * HTTP until the A record exists and bootstrap.sh is re-run with a domain. A silent
 * no-op there would be the worst outcome: a judge presses copy, pastes yesterday's
 * clipboard into a terminal and blames the command. So the failure is stated on the
 * button itself and the command stays selectable above it.
 */
export function CopyButton({ text, label = "copy" }: { text: string; label?: string }) {
  const [note, setNote] = useState<string | null>(null);

  async function copy() {
    try {
      if (!navigator.clipboard) throw new Error("clipboard unavailable on this origin");
      await navigator.clipboard.writeText(text);
      setNote("copied");
    } catch {
      setNote("copy blocked -- select the text");
    }
    window.setTimeout(() => setNote(null), 2500);
  }

  return (
    <button className="btn" onClick={copy} type="button">
      {note ?? label}
    </button>
  );
}

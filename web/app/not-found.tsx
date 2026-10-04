import type { Metadata } from "next";
import Link from "next/link";
import { TopBar } from "@/components/TopBar";

export const metadata: Metadata = { title: "Not found" };

/** The export's 404.html, served by Caddy for unknown paths. Same chrome as every other
 *  page, so a mistyped URL does not look like a broken site. */
export default function NotFound() {
  return (
    <>
      <TopBar />
      <main style={{ padding: "48px 16px", maxWidth: 720, margin: "0 auto" }}>
        <p className="label" style={{ margin: "0 0 6px" }}>404</p>
        <h1 className="display" style={{ fontSize: "var(--size-metric-s)", margin: "0 0 12px" }}>
          Nothing at this address
        </h1>
        <p style={{ color: "var(--ink-1)", margin: "0 0 16px" }}>
          This box serves three pages: <Link href="/">the wall</Link>,{" "}
          <Link href="/results">the results</Link> and <Link href="/judge">the path for judges</Link>.
          The build it is running is at <a href="/build-info">/build-info</a>.
        </p>
      </main>
    </>
  );
}

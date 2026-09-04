import type { Metadata, Viewport } from "next";
import { LimitsBanner } from "@/components/LimitsBanner";
import "../styles/globals.css";

/**
 * Global chrome. The LimitsBanner lives HERE and not in the pages, which is what makes
 * "persistent and non-dismissible on every screen" a structural fact rather than a
 * thing two pages both happen to remember to do.
 */

export const metadata: Metadata = {
  title: "PlumeRank -- wildfire camera attention ranking",
  // No figures in the description. This string is compiled into the export and cannot
  // be refetched, so any number in it is a number that outlives the run it came from --
  // and the one place the headline is allowed to live is results/bench_<date>.json,
  // served verbatim by /api/benchmark and rendered at runtime.
  description:
    "Ranks a wildfire camera network by an anchored plume-growth statistic under a hard " +
    "alert budget. Median time-to-detection measured on held-out HPWREN sequences.",
};

export const viewport: Viewport = {
  // The palette is dark in both branches; declaring it stops the browser painting a
  // light scrollbar and light form controls over a dark page before the CSS lands.
  colorScheme: "dark",
  themeColor: "#0A0E13",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* The one font file, preloaded. No external font request anywhere in this
            product: the endpoint has to work behind a strict network and must not leak
            judge traffic to a third party. */}
        <link
          rel="preload"
          href="/fonts/inter-latin-wght-normal.woff2"
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />
      </head>
      <body>
        {children}
        <LimitsBanner />
      </body>
    </html>
  );
}

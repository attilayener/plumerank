import type { Metadata, Viewport } from "next";
import { LimitsBanner } from "@/components/LimitsBanner";
import "../styles/globals.css";

/**
 * Global chrome. The LimitsBanner lives HERE and not in the pages, which is what makes
 * "persistent and non-dismissible on every screen" a structural fact rather than a
 * thing two pages both happen to remember to do.
 */

const TITLE = "PlumeRank — which wildfire camera to look at first";
// No measured figures in the description. This string is compiled into the export and
// cannot be refetched, so any result in it would outlive the run it came from -- and the
// one place the headline is allowed to live is results/bench_<date>.json, served
// verbatim by /api/benchmark. The budget is a design constant, not a result.
// <=125 chars: og:description clips on mobile earlier than a plain meta description does,
// so one string serves both constraints. It says what the page headline says.
const DESCRIPTION =
  "Two alerts per camera per 81-minute window. PlumeRank ranks wildfire cameras so the " +
  "likeliest smoke comes first.";
const SITE_URL = "http://44.242.91.201";
const AUTHOR = "Edy Cu";

export const metadata: Metadata = {
  // Needed to resolve the relative OG/Twitter image URLs below into absolute ones for
  // crawlers that cannot do it themselves. Points at the live box (no domain/TLS yet);
  // update when the real domain exists -- see deploy/bootstrap.sh's PLUMERANK_DOMAIN.
  metadataBase: new URL(SITE_URL),
  // Per-route titles ("Results — PlumeRank") come from each route's own layout; the
  // home page and anything without one keeps the full title.
  title: { default: TITLE, template: "%s — PlumeRank" },
  description: DESCRIPTION,
  authors: [{ name: AUTHOR }],
  creator: AUTHOR,
  icons: {
    icon: "/icon.svg",
  },
  openGraph: {
    title: TITLE,
    description: DESCRIPTION,
    url: SITE_URL,
    siteName: "PlumeRank",
    images: [{ url: "/og-image.png?v=3", width: 1200, height: 630, alt: TITLE }],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
    // No X/Twitter account exists for this project -- the real author's name, not a
    // fabricated handle (a string here does not have to resolve to an account, but it
    // must be honest about what it names).
    creator: AUTHOR,
    images: ["/og-image.png?v=3"],
  },
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

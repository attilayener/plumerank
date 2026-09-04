/**
 * PlumeRank web -- Next.js 15 STATIC EXPORT.
 *
 * `output: "export"` is the whole deployment story: there is no Node process on the
 * box at request time. Caddy serves web/out as files and reverse-proxies /api/*,
 * /health and /build-info to Uvicorn (deploy/Caddyfile). A t4g.small holding 40 warm
 * background models has no spare vCPU to also render React, and a server-rendered
 * frontend would put page loads in contention with the pipeline producing the frames.
 *
 * Consequences that are load-bearing rather than incidental:
 *  - every page is a client component: all data arrives at runtime from the API, so
 *    there is nothing for a server render to do except emit an empty shell twice.
 *  - `images.unoptimized` because the Next image optimiser is a server. Evidence crops
 *    are already the right size -- rank.draw_evidence writes the crop with the overlay
 *    burnt in -- so there is nothing to optimise.
 *  - no rewrites/redirects/headers: none of them exist in an export, and writing them
 *    here would produce a config that works in `next dev` and silently does nothing in
 *    production, which is the worst of the available failure modes.
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  reactStrictMode: true,
  images: { unoptimized: true },
  // The endpoint must not leak judge traffic to a third party, and that includes us.
  devIndicators: false,
};

export default nextConfig;

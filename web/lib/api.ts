/**
 * Where the frontend gets its numbers, and the one place that decides it.
 *
 * IN PRODUCTION THE BASE URL IS THE EMPTY STRING, and that is the deployment design
 * rather than a default nobody thought about: Caddy serves this static export and
 * reverse-proxies /api/*, /health and /build-info to Uvicorn on 127.0.0.1:8000
 * (deploy/Caddyfile), so the page and the API are the same origin. No CORS, no
 * preflight, no second hostname to get wrong, and nothing to configure on the box.
 *
 * DEV-TIME BASE URL
 * -----------------
 * `next dev` serves the page on :3000 while the API runs on :8000, so the two are NOT
 * the same origin. Point the frontend at the API with:
 *
 *     # terminal 1 -- the API
 *     .venv/bin/python -m uvicorn api.main:app --reload --port 8000
 *
 *     # terminal 2 -- the frontend
 *     cd web && NEXT_PUBLIC_PLUMERANK_API=http://127.0.0.1:8000 npm run dev
 *
 * NEXT_PUBLIC_* is inlined at BUILD time, which is the correct behaviour here: the
 * production build is made by deploy/bootstrap.sh with the variable unset, so the
 * shipped bundle can only ever talk to its own origin. A runtime-configurable base URL
 * would be one more way for the deployed page to end up pointed somewhere other than
 * the box it is served from.
 */

const BASE = (process.env.NEXT_PUBLIC_PLUMERANK_API ?? "").replace(/\/+$/, "");

export function apiUrl(path: string): string {
  return `${BASE}${path}`;
}

/** The four routes of api/main.py. Named once so a typo is a build error, not a 404. */
export const ROUTES = {
  buildInfo: "/build-info",
  health: "/health",
  watchlist: "/api/watchlist",
  benchmark: "/api/benchmark",
} as const;

/** The curl a judge copies off the BuildInfo section. Absolute, because it is pasted
 *  into a terminal that has no notion of the page's origin. */
export function curlCommand(origin: string): string {
  return `curl -s ${origin}${ROUTES.buildInfo} | head -40`;
}

/**
 * Fetch JSON, or throw. There is deliberately no fallback value anywhere in this file:
 * a component that receives `null` renders its designed "not published" state, and a
 * component that receives a plausible-looking default renders a lie. The whole point
 * of /api/benchmark serving results/bench_<date>.json verbatim is that the page, the
 * endpoint and the repo cannot disagree, which only holds if the page refuses to
 * invent the bytes it did not receive.
 */
export async function fetchJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(apiUrl(path), { signal, cache: "no-store" });
  if (!res.ok) {
    throw new Error(`${path} responded ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

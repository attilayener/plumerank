import { defineConfig, devices } from "@playwright/test";

/**
 * CI starts the real FastAPI backend itself (see .github/workflows/ci.yml, Stage 4)
 * and builds this app against it before `playwright test` runs, so `webServer` here
 * only needs to serve the already-built app -- it does not start a mock API, because
 * there is no mock API in this project (api/main.py degrades honestly instead).
 *
 * This is a static export (next.config.mjs: `output: "export"`), so there is no Next
 * server to start -- `next start` errors out on this config (confirmed by running it).
 * `npm run start` (web/package.json) instead serves `out/` with `serve`, the same shape
 * Caddy serves in production.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? "html" : "list",
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile-chrome", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: "npm run start",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});

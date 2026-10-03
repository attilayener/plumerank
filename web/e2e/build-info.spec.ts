import { test, expect } from "@playwright/test";

/**
 * /results#build renders what the REAL backend's /build-info returns. The section once
 * read a flat shape (opencv_version, dnn_engine, modules) the server never sent, so the
 * live page showed "--" for every value while curl showed OpenCV 5.0.0. This runs
 * against the actual FastAPI app, so a renamed field fails here, not in front of a judge.
 */
test("the build-information table shows the server's OpenCV version and module probes", async ({
  page,
}) => {
  await page.goto("/results");
  const table = page.locator("#build table");
  const opencvRow = table.locator("tr", { has: page.locator("th", { hasText: /^OpenCV$/ }) });
  await expect(opencvRow.locator("td")).toHaveText(/^\d+\.\d+\.\d+/, { timeout: 15_000 });
  await expect(page.locator("#build .badge", { hasText: /^video / })).toBeVisible();
});

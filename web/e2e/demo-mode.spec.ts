import { test, expect } from "@playwright/test";

/**
 * Smoke test: the Wall loads with no API keys, no accounts, and no login -- and when
 * the backend's live pipeline is not yet calibrated (the honest, current state of this
 * repo -- see JUDGE.md section 4), the page says so instead of crashing or faking data.
 */

test.describe("demo mode", () => {
  test("home page loads without API keys and carries correct metadata", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));

    const response = await page.goto("/");
    expect(response?.ok()).toBeTruthy();

    await expect(page).toHaveTitle(/PlumeRank/);
    expect(
      await page.locator('meta[name="description"]').getAttribute("content")
    ).toMatch(/two alerts per camera per 81-minute window/i);

    expect(errors).toEqual([]);
  });

  test("the non-dismissible limits banner is present on every load", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText(/daylight/i).first()).toBeVisible();
  });

  test("a disconnected or uncalibrated live pipeline is shown honestly, not faked", async ({
    page,
  }) => {
    await page.goto("/");
    // Either a live ranking is streaming (role="status" is reserved for the honest
    // degraded/empty states -- DegradedStrip and WatchList's own empty message), or one
    // of those honest states is visible. There is no third state where fabricated rows
    // are drawn from nothing.
    // Visible matches only: on narrow viewports the wall's area is display:none, and its
    // own inline notice comes first in DOM order, ahead of the DegradedStrip below it.
    const honestState = page
      .getByText(/unavailable|not delivering/i)
      .filter({ visible: true })
      .first();
    const rankedRow = page.locator(".watchlist-scroll .row").first();
    await expect(honestState.or(rankedRow).first()).toBeVisible({ timeout: 15_000 });
  });
});

test("a running wall names the run its threshold came from", async ({ page }) => {
  await page.goto("/");
  const ranked = page.locator(".watchlist-scroll .row").first();
  const honest = page.getByText(/unavailable|not delivering/i).filter({ visible: true }).first();
  await expect(honest.or(ranked).first()).toBeVisible({ timeout: 20_000 });
  if (await ranked.isVisible()) {
    // Whenever the wall is ranking, the provenance of its threshold is on screen.
    await expect(
      page.getByText(/Threshold solved on the fire-free confusion set/).first(),
    ).toBeAttached();
  }
});

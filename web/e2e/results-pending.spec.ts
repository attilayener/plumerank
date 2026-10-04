import { test, expect } from "@playwright/test";

/**
 * Before the held-out run, /results shows the committed development run -- labelled
 * in-sample -- and every anchor the rest of the site links to still lands somewhere.
 */

test("/results before the held-out run: development evidence, labelled, with live anchors", async ({
  page,
}) => {
  await page.goto("/results");
  const pending = page.getByText(/Not run yet/);
  const headline = page.getByText(/min p50/);
  await expect(pending.or(headline).first()).toBeVisible({ timeout: 15_000 });
  if (!(await pending.isVisible())) test.skip(true, "a held-out result is published");

  await expect(page.getByText(/development set — in-sample, not the headline/i)).toBeVisible();
  await expect(page.getByText(/like-for-like/i).first()).toBeVisible();
  await expect(page.locator("#reproduce")).toBeAttached();
  await expect(page.locator("#budget")).toBeAttached();
  // Both rates, each named: the calibration rate is never passed off as the arm's.
  await expect(page.getByText(/on the fire-free confusion set/i).first()).toBeVisible();
  await expect(page.getByText(/realised/i).first()).toBeVisible();
});

test("the copy button copies or selects -- it never just fails", async ({ page }) => {
  await page.goto("/results");
  const button = page.getByRole("button", { name: /copy/ }).first();
  await expect(button).toBeVisible({ timeout: 15_000 });
  await button.click();
  await expect(page.getByRole("button", { name: /copied|selected/ }).first()).toBeVisible();
});

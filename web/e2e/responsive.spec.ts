import { test, expect } from "@playwright/test";

/**
 * Layout at the three breakpoints the Wall's grid (web/styles/globals.css) actually
 * designs for. No horizontal scrollbar at any of them, and the non-dismissible limits
 * banner (present on every screen by construction -- see LimitsBanner.tsx) stays
 * reachable.
 */

const SIZES = [
  { name: "mobile", width: 375, height: 812 },
  { name: "tablet", width: 768, height: 1024 },
  { name: "desktop", width: 1440, height: 900 },
] as const;

for (const { name, width, height } of SIZES) {
  test(`no horizontal overflow at ${name} (${width}x${height})`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await page.goto("/");

    const [scrollWidth, clientWidth] = await page.evaluate(() => [
      document.documentElement.scrollWidth,
      document.documentElement.clientWidth,
    ]);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 1); // +1: sub-pixel rounding

    await expect(page.locator(".limits-banner")).toBeVisible();
  });
}

test("/judge has no horizontal overflow on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto("/judge");
  const [scrollWidth, clientWidth] = await page.evaluate(() => [
    document.documentElement.scrollWidth,
    document.documentElement.clientWidth,
  ]);
  expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 1);
});

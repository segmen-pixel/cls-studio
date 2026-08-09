// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/**
 * Cancelling an inspection actually cancels it.
 *
 * Reported as 「中止ボタンが効かない」. Three separate defects made it true:
 *   - the cancel flag was read only at the top of the pump loop, and `shift()`
 *     empties the queue before the await, so for the last (or only) image the
 *     loop exited on the `while` condition without ever re-reading it — a
 *     literal 100% no-op;
 *   - the in-flight request was never aborted (no AbortController anywhere),
 *     so the wait lasted a full score either way;
 *   - the click wrote a ref and nothing re-rendered, so the UI was
 *     pixel-identical afterwards and the button read as dead.
 *
 * The single-image case below is the one that was permanently broken, so it is
 * the one worth pinning. The route is throttled because the defect only shows
 * while a score is in flight.
 */
import { expect, test } from "@playwright/test";
import {
  createProject, deleteProject, selectBank, teachImage, tinyPng, uniqueName,
} from "../helpers/api";

test.describe.configure({ timeout: 300_000 });

test("cancel during the only queued image drops it and leaves no result", async ({ page, request }) => {
  const proj = await createProject(request, uniqueName("cancel"));
  try {
    await selectBank(request, proj.id);
    await teachImage(request, "normal", "e2e-cancel-seed-1.png");
    await teachImage(request, "normal", "e2e-cancel-seed-2.png");

    await page.goto("/");
    await page.getByText(proj.name).first().click();
    await page.getByRole("button", { name: /^(検査|Inspect)$/ }).first().click();

    // Hold /score open so the cancel lands while the request is in flight —
    // exactly the window in which it used to do nothing.
    await page.route("**/score?**", async (route) => {
      await new Promise((r) => setTimeout(r, 10_000));
      await route.continue();
    });

    const input = page.locator('input[type="file"][multiple]').last();
    await input.setInputFiles({
      name: "e2e-cancel-me.png", mimeType: "image/png", buffer: tinyPng(),
    });
    await expect(page.getByText("e2e-cancel-me.png").first()).toBeVisible({ timeout: 30_000 });

    const cancel = page.getByRole("button", { name: /^(キャンセル|Cancel)$/ }).first();
    await expect(cancel).toBeVisible({ timeout: 30_000 });

    const t0 = Date.now();
    await cancel.click();

    // The row must go, and it must go WITHOUT waiting out the in-flight score.
    // That elapsed-time bound is the whole contract: the flag alone could only
    // be honoured after the request returned, and for the last (or only) image
    // it was never honoured at all. The 10 s route delay above is the thing
    // being outrun — the transient 「キャンセル中…」 label is deliberately not
    // asserted, because with the abort in place the row disappears too fast
    // for it to be reliably observable.
    await expect(page.getByText("e2e-cancel-me.png")).toHaveCount(0, { timeout: 8_000 });
    expect(Date.now() - t0, "cancel waited for the in-flight score").toBeLessThan(5_000);
    await expect(page.getByRole("img", { name: "anomaly heatmap" })).toHaveCount(0);

    // And it must not come back as a committed result.
    await page.unroute("**/score?**");
    await page.reload();
    await page.getByRole("button", { name: /^(検査|Inspect)$/ }).first().click();
    await expect(page.getByText("e2e-cancel-me.png")).toHaveCount(0, { timeout: 30_000 });
  } finally {
    await deleteProject(request, proj.id);
  }
});

test("files dropped right after a cancel are not swallowed", async ({ page, request }) => {
  const proj = await createProject(request, uniqueName("cancelre"));
  try {
    await selectBank(request, proj.id);
    await teachImage(request, "normal", "e2e-cancel2-seed.png");

    await page.goto("/");
    await page.getByText(proj.name).first().click();
    await page.getByRole("button", { name: /^(検査|Inspect)$/ }).first().click();

    let slow = true;
    await page.route("**/score?**", async (route) => {
      if (slow) await new Promise((r) => setTimeout(r, 8_000));
      await route.continue();
    });

    const input = page.locator('input[type="file"][multiple]').last();
    await input.setInputFiles({
      name: "e2e-first.png", mimeType: "image/png", buffer: tinyPng(),
    });
    await expect(page.getByText("e2e-first.png").first()).toBeVisible({ timeout: 30_000 });
    await page.getByRole("button", { name: /^(キャンセル|Cancel)$/ }).first().click();
    await expect(page.getByText("e2e-first.png")).toHaveCount(0, { timeout: 60_000 });

    // The cancel flag used to stay latched after the loop exited, so the next
    // drop was spliced away the moment the pump resumed — the files vanished
    // with no toast and no explanation.
    slow = false;
    await input.setInputFiles({
      name: "e2e-second.png", mimeType: "image/png", buffer: tinyPng(),
    });
    await expect(page.getByText("e2e-second.png").first()).toBeVisible({ timeout: 30_000 });
    await expect(
      page.getByRole("img", { name: "anomaly heatmap" }).first(),
    ).toBeVisible({ timeout: 240_000 });
  } finally {
    await deleteProject(request, proj.id);
  }
});

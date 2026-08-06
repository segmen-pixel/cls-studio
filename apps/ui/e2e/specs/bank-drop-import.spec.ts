// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/**
 * Dropping image files on the Bank tab imports them.
 *
 * The tab had no drop handler at all — importing meant the hidden file input
 * behind the Import button — while the Inspect tab took the same gesture
 * happily. A gesture that works one tab over reads as broken rather than
 * missing, so this asserts the overlay as well as the import: telling the user
 * the drop will land is half of the fix.
 */
import { expect, test, type Page } from "@playwright/test";
import {
  createProject, deleteProject, selectBank, TINY_PNG_B64, uniqueName,
} from "../helpers/api";

test.describe.configure({ timeout: 180_000 });

const OVERLAY = /^(ドロップで取り込み|Drop to import)$/;
const ALL_CHIP = /^(すべて|All)\s+\d+$/;

/** A DataTransfer carrying real Files, the way a desktop drop delivers them. */
function fileDataTransfer(page: Page, names: string[]) {
  return page.evaluateHandle(async ({ b64, names }) => {
    const dt = new DataTransfer();
    for (const name of names) {
      const blob = await (await fetch(`data:image/png;base64,${b64}`)).blob();
      dt.items.add(new File([blob], name, { type: "image/png" }));
    }
    return dt;
  }, { b64: TINY_PNG_B64, names });
}

test("dropping files anywhere on the bank imports them", async ({ page, request }) => {
  const proj = await createProject(request, uniqueName("drop"));
  try {
    await selectBank(request, proj.id);

    await page.goto("/");
    // Clicking the tile only selects the project — that is what binds the
    // client to it — so the Bank tab still has to be opened by hand.
    await page.getByText(proj.name).first().click();
    await page.getByRole("button", { name: /^(バンク|Bank)$/ }).first().click();

    const bank = page.locator(".bank-tab");
    await expect(bank).toBeVisible({ timeout: 30_000 });

    const allChip = page.getByRole("button", { name: ALL_CHIP });
    await expect(allChip).toHaveText(/\b0$/, { timeout: 15_000 });

    const dt = await fileDataTransfer(page, ["e2e-drop-1.png", "e2e-drop-2.png"]);

    // Dragging over the tab has to say the drop will land, before it does.
    await bank.dispatchEvent("dragenter", { dataTransfer: dt });
    await expect(page.getByText(OVERLAY)).toBeVisible({ timeout: 5_000 });

    await bank.dispatchEvent("drop", { dataTransfer: dt });

    // Overlay clears, and both images are in the bank.
    await expect(page.getByText(OVERLAY)).toHaveCount(0, { timeout: 30_000 });
    await expect(allChip).toHaveText(/\b2$/, { timeout: 120_000 });
  } finally {
    await deleteProject(request, proj.id);
  }
});

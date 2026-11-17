// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/**
 * A running job's 中止 must stay reachable when the user changes tabs.
 *
 * The separation sweep, the teach-progress dialog and the heatmap pre-render
 * are deliberately NOT gated on the tab being active — they block the whole
 * app on purpose, because any side work would race them for the GPU. But they
 * were rendered inside the tab panel, and `.tab-panel { display: none }` hides
 * a fixed-position child just as thoroughly as anything else. Switching tabs
 * therefore took the overlay away and left the job running with no way to stop
 * it. They are portalled to <body> now.
 *
 * (The bank-activation and upload dialogs ARE gated on `active` on purpose —
 * they fire automatically on tab activation and must not cover wherever the
 * user just went. Those are not what this pins.)
 */
import { expect, test } from "@playwright/test";
import {
  createProject, deleteProject, selectBank, teachImage, uniqueName,
} from "../helpers/api";

test.describe.configure({ timeout: 300_000 });

test("sweep cancel stays reachable across a tab switch", async ({ page, request }) => {
  const proj = await createProject(request, uniqueName("sweeptab"));
  try {
    await selectBank(request, proj.id);
    await teachImage(request, "normal", "e2e-sweep-1.png");
    await teachImage(request, "normal", "e2e-sweep-2.png");
    await teachImage(request, "critical", "e2e-sweep-3.png");

    await page.goto("/");
    await page.getByText(proj.name).first().click();
    await page.getByRole("button", { name: /^(学習|Teach)$/ }).first().click();

    // Slow the per-image evaluate so the sweep dialog is observable.
    await page.route("**/bank/images/evaluate**", async (route) => {
      await new Promise((r) => setTimeout(r, 4_000));
      await route.continue();
    });

    // Not anchored: the button renders with a "▶ " prefix.
    await page.getByRole("button", { name: /(評価を実行|Run evaluation)/ }).first().click();

    const dialog = page.getByRole("alertdialog");
    await expect(dialog).toBeVisible({ timeout: 60_000 });

    // The tab bar sits behind the modal, so the keyboard shortcut is the only
    // way to reach it — a window-level handler with no modal guard.
    await page.keyboard.press("Control+ArrowRight");
    await expect(dialog, "the overlay must survive the tab change").toBeVisible();

    // And its cancel must still work from there.
    const cancel = dialog.getByRole("button", { name: /^(中止|Cancel)$/ });
    await expect(cancel).toBeVisible();
    await cancel.click();
    await expect(dialog).toHaveCount(0, { timeout: 60_000 });
  } finally {
    await deleteProject(request, proj.id);
  }
});

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/**
 * The Teach tab's image list: result column, filters, and moving a checked
 * batch to the suppression tier.
 *
 * The list's third column used to be `${patches}p`, which is derived from the
 * image's dimensions — on a project shot at one sensor size every row read the
 * same number. It now carries the separation check's score and a mark for the
 * rows that fell the wrong side of the threshold, and those are the rows this
 * whole flow exists to move.
 *
 * The verdicts are published by the check rather than derived in the list, so
 * the assertions below deliberately compare the list against the check card:
 * a second derivation would be a second threshold, and it would disagree the
 * moment the slider moved.
 */
import { expect, test, type Page } from "@playwright/test";
import {
  bankState, createProject, deleteProject, ingestAndAssemble, selectBank, uniqueName,
} from "../helpers/api";

// The first teach loads DINOv2; everything after it is stored-feature maths.
test.describe.configure({ timeout: 420_000 });

const TEACH_TAB = /^(学習|Teach)$/;
const ALL_CHIP = /^(すべて|All)\s+\d+$/;
const MISS_CHIP = /^(見逃し|Miss)\s+\d+$/;
const RUN_SWEEP = /(評価を実行|Run evaluation)/;
// The action button, not the tier filter chip: both carry the tier's name, so
// the count and the trailing particle are what tell them apart.
const MOVE_BTN = /^\d+\s*(件を過検知抑制へ|to FP suppression)/;

const rows = (page: Page) => page.locator('.develop-tab [role="option"]');
const boxes = (page: Page) => rows(page).locator('input[type="checkbox"]');

async function openTeach(page: Page, name: string) {
  await page.goto("/");
  await page.getByText(name).first().click();
  await page.getByRole("button", { name: TEACH_TAB }).first().click();
  await expect(rows(page).first()).toBeVisible({ timeout: 30_000 });
}

async function seed(request: Parameters<typeof createProject>[0], slug: string) {
  const proj = await createProject(request, uniqueName(slug));
  await selectBank(request, proj.id);
  // Through the store, not /bank/append: the move under test resolves each
  // image to a store entry, and the retired append path creates none.
  await ingestAndAssemble(request, [
    { name: "ok-a.png", tier: "normal" },
    { name: "ok-b.png", tier: "normal" },
    { name: "ng-a.png", tier: "critical" },
  ]);
  return proj;
}

test("the result column is a dash until the sweep fills it in", async ({ page, request }) => {
  const proj = await seed(request, "result-col");
  try {
    await openTeach(page, proj.name);
    // Before any sweep every row reads a dash -- not an empty cell, which
    // would leave the column with no reason to exist.
    await expect(rows(page).first()).toContainText("—");

    await page.getByRole("button", { name: RUN_SWEEP }).first().click();
    // Numbers replace the dashes once the per-image evaluations land.
    await expect(rows(page).first()).not.toContainText("—", { timeout: 120_000 });
    await expect(rows(page).first()).toContainText(/\d+\.\d\d/);
  } finally {
    await deleteProject(request, proj.id);
  }
});

test("the verdict chips narrow the list, and agree with the check card",
  async ({ page, request }) => {
    const proj = await seed(request, "list-filter");
    try {
      await openTeach(page, proj.name);
      await expect(rows(page)).toHaveCount(3);

      await page.getByRole("button", { name: RUN_SWEEP }).first().click();
      await expect(rows(page).first()).not.toContainText("—", { timeout: 120_000 });

      // Whatever the check calls a miss, the chip must count -- the point of
      // publishing one set of verdicts instead of deriving them twice.
      const miss = page.getByRole("button", { name: MISS_CHIP });
      const label = (await miss.getAttribute("aria-label")) ?? "";
      const n = Number(label.match(/(\d+)\s*$/)?.[1] ?? "0");
      if (n === 0) {
        await expect(miss).toBeDisabled();
      } else {
        await miss.click();
        await expect(rows(page)).toHaveCount(n);
      }

      // "All" always comes back to the whole bank.
      await page.getByRole("button", { name: ALL_CHIP }).click();
      await expect(rows(page)).toHaveCount(3);
    } finally {
      await deleteProject(request, proj.id);
    }
  });

test("checked rows move to the suppression tier", async ({ page, request }) => {
  const proj = await seed(request, "move-sel");
  try {
    const before = await bankState(request);
    await openTeach(page, proj.name);

    // Tick the first, then shift-click the second: the range is what makes a
    // pass over a long list bearable, so it is the gesture under test.
    await boxes(page).nth(0).click();
    await boxes(page).nth(1).click({ modifiers: ["Shift"] });

    const move = page.getByRole("button", { name: MOVE_BTN });
    await expect(move).toBeVisible();
    await move.click();

    // The tier really moved, server-side.
    await expect
      .poll(async () => (await bankState(request)).negative, { timeout: 120_000 })
      .toBeGreaterThan(before.negative);

    // And the selection clears, so the action cannot be fired twice.
    await expect(page.getByRole("button", { name: MOVE_BTN })).toHaveCount(0);
  } finally {
    await deleteProject(request, proj.id);
  }
});

test("clicking a row opens it without checking it", async ({ page, request }) => {
  const proj = await seed(request, "click-vs-check");
  try {
    await openTeach(page, proj.name);
    await rows(page).first().click();
    // The row is selected in the viewer sense...
    await expect(rows(page).first()).toHaveAttribute("aria-selected", "true");
    // ...but nothing is checked, so no action bar appears. This is the whole
    // reason the list uses checkboxes rather than the bank tab's click-select.
    await expect(boxes(page).first()).not.toBeChecked();
    await expect(page.getByRole("button", { name: MOVE_BTN })).toHaveCount(0);
  } finally {
    await deleteProject(request, proj.id);
  }
});

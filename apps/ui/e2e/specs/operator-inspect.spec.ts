// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/**
 * The real Operator flow through the UI: with a seeded bank, drop an image
 * on the Operator tab, watch it score, and confirm the result survives a
 * reload (server-side inspection log).
 */
import { expect, test } from "@playwright/test";
import {
  createProject, deleteProject, selectBank, teachImage, tinyPng, uniqueName,
} from "../helpers/api";

test.describe.configure({ timeout: 300_000 });

test("drop on Operator -> scored result -> survives reload", async ({ page, request }) => {
  const proj = await createProject(request, uniqueName("inspect"));
  try {
    await selectBank(request, proj.id);
    await teachImage(request, "normal", "e2e-seed-1.png");
    await teachImage(request, "normal", "e2e-seed-2.png");

    await page.goto("/");
    // Open the seeded project so the Operator tab binds to its bank.
    await page.getByText(proj.name).first().click();
    await page.getByRole("button", { name: /^(検査|Inspect)$/ }).first().click();

    // The drop zone is backed by a hidden multi-file input.
    const input = page.locator('input[type="file"][multiple]').last();
    await input.setInputFiles({
      name: "e2e-probe.png", mimeType: "image/png", buffer: tinyPng(),
    });

    // The queued file appears and eventually gets scored (first score can
    // be slow: model load + bank tensor build). A rendered heatmap is the
    // completion signal — without a saved verdict recipe there is no OK/NG
    // verdict to assert on.
    await expect(page.getByText("e2e-probe.png").first()).toBeVisible({ timeout: 30_000 });
    await expect(
      page.getByRole("img", { name: "anomaly heatmap" }).first(),
    ).toBeVisible({ timeout: 240_000 });

    // Reload: the inspection log restores the entry server-side.
    await page.reload();
    await page.getByRole("button", { name: /^(検査|Inspect)$/ }).first().click();
    await expect(page.getByText("e2e-probe.png").first()).toBeVisible({ timeout: 30_000 });
  } finally {
    await deleteProject(request, proj.id);
  }
});

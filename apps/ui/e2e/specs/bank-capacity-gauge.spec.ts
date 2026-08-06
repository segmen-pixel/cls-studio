// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/**
 * The capacity gauge must account for the labelled tiers.
 *
 * The bar and its "N / ceiling patches" line only ever described the normal
 * tier, because that is all the ceiling governs — but the critical and
 * negative tiers are resident too and are not capped at all. On a real project
 * on the dev box that read "11.7% · ~240 MB" while the bank actually held
 * 1,248,528 labelled patches, about 2 GB. Someone watching that gauge to stay
 * off the pagefile was being told the opposite of the truth.
 */
import { expect, test } from "@playwright/test";
import { createProject, deleteProject, selectBank, teachImage, uniqueName } from "../helpers/api";

test.describe.configure({ timeout: 300_000 });

test("settings gauge reports labelled patches, which no ceiling covers", async ({ page, request }) => {
  const proj = await createProject(request, uniqueName("gauge"));
  try {
    await selectBank(request, proj.id);
    // Two normal images against one critical, so the two counts differ — with
    // one each they are both a single window's patches and the assertion below
    // cannot tell which line it matched.
    await teachImage(request, "normal", "e2e-gauge-ok-1.png");
    await teachImage(request, "normal", "e2e-gauge-ok-2.png");
    await teachImage(request, "critical", "e2e-gauge-ng.png", "scratch");

    // The API is the contract; assert it before trusting the rendering.
    const cap = await (await request.get("http://localhost:8791/api/v1/bank/capacity")).json();
    expect(cap.normal, "normal tier seeded").toBeGreaterThan(0);
    expect(cap.labeled, "critical tier must be counted").toBeGreaterThan(0);
    expect(cap.labeled).not.toBe(cap.normal);
    expect(cap.est_vram_total_mb).toBeGreaterThanOrEqual(cap.est_vram_mb);

    await page.goto("/");
    await page.getByText(proj.name).first().click();
    await page.getByRole("button", { name: /^(設定|Settings)$/ }).first().click();

    // The labelled-tier line must be on screen carrying the real count — and
    // it must be that line, not the normal-tier one that was always there.
    const labeledLine = page.getByText(/(NG・除外タグのパッチ|NG \/ excluded patches)/);
    await expect(labeledLine).toBeVisible({ timeout: 30_000 });
    await expect(labeledLine).toContainText(cap.labeled.toLocaleString("en-US"));
  } finally {
    await deleteProject(request, proj.id);
  }
});

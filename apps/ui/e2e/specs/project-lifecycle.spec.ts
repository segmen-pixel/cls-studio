// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/** Project appears on the Projects grid after creation and vanishes after delete. */
import { expect, test } from "@playwright/test";
import { createProject, deleteProject, uniqueName } from "../helpers/api";

test("created project shows on the grid; deleted project disappears", async ({ page, request }) => {
  const name = uniqueName("lifecycle");
  const proj = await createProject(request, name);
  try {
    await page.goto("/");
    await expect(page.getByText(name).first()).toBeVisible({ timeout: 15_000 });
  } finally {
    await deleteProject(request, proj.id);
  }
  await page.reload();
  await expect(page.getByText(name)).toHaveCount(0, { timeout: 15_000 });
});

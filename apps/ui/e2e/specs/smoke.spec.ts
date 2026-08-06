// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/** App boots: UI loads, every tab exists, health endpoint answers. */
import { expect, test } from "@playwright/test";
import { API } from "../helpers/api";

test("UI loads with every tab present (either language)", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: /^(プロジェクト|Projects)$/ }).first(),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /^(バンク|Bank)$/ }).first(),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /^(学習|Teach)$/ }).first(),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /^(検査|Inspect)$/ }).first(),
  ).toBeVisible();
});

test("health endpoint reports ok", async ({ request }) => {
  const res = await request.get(`${API}/health`);
  expect(res.ok()).toBeTruthy();
  const body = (await res.json()) as { status: string; version: string };
  expect(body.status).toBe("ok");
  expect(body.version).toBeTruthy();
});

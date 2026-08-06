// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/** API helpers for the e2e suite — seed state fast, assert through the UI. */
import { expect, type APIRequestContext } from "@playwright/test";

export const BASE = process.env.CLS_E2E_BASE ?? "http://localhost:8791";
export const API = `${BASE}/api/v1`;

/** 64x64 PNG (solid mid-grey with a darker corner square), base64. */
export const TINY_PNG_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAdUlEQVR4nO3PQQ3AMBADwYNTmFGQ" +
  "F0P72UgZywB25vm4tddRHwAAAAAAAAAAAACA34C8AKAuAKgLAOoCgLoAoC4AqAsA6gKAugCgLgCo" +
  "CwDqAoC6AKAuAKgLAOoCgLoAoC4AqAsA6gKAugCgLgCoCwDqgtsBL7XmF1qJPGTYAAAAAElFTkSu" +
  "QmCC";

export function tinyPng(): Buffer {
  return Buffer.from(TINY_PNG_B64, "base64");
}

export function uniqueName(slug: string): string {
  return `zz-e2e-${slug}-${Date.now().toString(36)}`;
}

export async function createProject(
  ctx: APIRequestContext, name: string,
): Promise<{ id: string; name: string }> {
  const res = await ctx.post(`${API}/projects`, { data: { name } });
  expect(res.ok(), `create project -> ${res.status()}`).toBeTruthy();
  return (await res.json()) as { id: string; name: string };
}

export async function deleteProject(ctx: APIRequestContext, id: string): Promise<void> {
  await ctx.delete(`${API}/projects/${id}`); // best-effort cleanup
}

/** Bind the server's active bank to this project (default bank). */
export async function selectBank(ctx: APIRequestContext, projectId: string): Promise<void> {
  const res = await ctx.post(`${API}/bank/select`, { data: { project_id: projectId } });
  expect(res.ok(), `bank/select -> ${res.status()}`).toBeTruthy();
}

/** Teach one tiny image into the given tier of the ACTIVE bank. */
export async function teachImage(
  ctx: APIRequestContext, tier: "normal" | "critical", name: string, label = "",
): Promise<void> {
  const res = await ctx.post(`${API}/bank/append/${tier}`, {
    multipart: {
      image: { name, mimeType: "image/png", buffer: tinyPng() },
      label,
    },
    timeout: 120_000, // first call loads DINOv2
  });
  expect(res.ok(), `append/${tier} -> ${res.status()}`).toBeTruthy();
}

export async function bankState(
  ctx: APIRequestContext,
): Promise<{ normal: number; critical: number; negative: number }> {
  const res = await ctx.get(`${API}/bank`);
  expect(res.ok(), `GET /bank -> ${res.status()}`).toBeTruthy();
  return (await res.json()) as { normal: number; critical: number; negative: number };
}

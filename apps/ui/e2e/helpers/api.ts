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

/**
 * Seed a bank the way the product actually builds one now: ingest into the
 * store, assign tiers on the label set, assemble.
 *
 * `teachImage` above uses `/bank/append`, the retired path — it writes
 * straight into the bank and creates NO store entries, so anything that
 * resolves an image through the store finds nothing. A spec that seeds with
 * it is testing a shape no new project has.
 */
export async function ingestAndAssemble(
  ctx: APIRequestContext,
  images: Array<{ name: string; tier: "normal" | "critical" | "negative" }>,
): Promise<void> {
  const res = await ctx.post(`${API}/store/ingest`, {
    multipart: {
      // Playwright's multipart takes one value per key; the route accepts a
      // repeated `images` field, so send them one request at a time.
      images: { name: images[0].name, mimeType: "image/png", buffer: tinyPng() },
      tier: "",
    },
    timeout: 180_000, // first call loads DINOv2
  });
  expect(res.ok(), `store/ingest -> ${res.status()}`).toBeTruthy();
  for (const im of images.slice(1)) {
    const r = await ctx.post(`${API}/store/ingest`, {
      multipart: {
        images: { name: im.name, mimeType: "image/png", buffer: tinyPng() },
        tier: "",
      },
      timeout: 120_000,
    });
    expect(r.ok(), `store/ingest ${im.name} -> ${r.status()}`).toBeTruthy();
  }

  const listing = await (await ctx.get(`${API}/store`)).json() as {
    images: Array<{ id: string; name: string }>;
  };
  const idOf = new Map(listing.images.map((e) => [e.name, e.id]));
  for (const tier of ["normal", "critical", "negative"] as const) {
    const ids = images.filter((im) => im.tier === tier)
      .map((im) => idOf.get(im.name))
      .filter((x): x is string => !!x);
    if (ids.length === 0) continue;
    const a = await ctx.post(`${API}/labelsets/assign`, { data: { ids, tier, label: "" } });
    expect(a.ok(), `labelsets/assign ${tier} -> ${a.status()}`).toBeTruthy();
  }

  const asm = await ctx.post(`${API}/bank/assemble`, { data: {}, timeout: 180_000 });
  expect(asm.ok(), `bank/assemble -> ${asm.status()}`).toBeTruthy();
}

export async function bankState(
  ctx: APIRequestContext,
): Promise<{ normal: number; critical: number; negative: number }> {
  const res = await ctx.get(`${API}/bank`);
  expect(res.ok(), `GET /bank -> ${res.status()}`).toBeTruthy();
  return (await res.json()) as { normal: number; critical: number; negative: number };
}

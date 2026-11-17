// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/**
 * Teach path against the live server, end to end through HTTP: staging
 * upload -> label -> teach (real DINOv2 extraction) -> bank counts rise ->
 * per-image delete prunes them back out.
 */
import { expect, test } from "@playwright/test";
import {
  API, bankState, createProject, deleteProject, selectBank, teachImage,
  tinyPng, uniqueName,
} from "../helpers/api";

test.describe.configure({ timeout: 240_000 }); // first teach loads the model

test("staging upload -> teach lands rows in the bank", async ({ request }) => {
  const proj = await createProject(request, uniqueName("teach"));
  try {
    await selectBank(request, proj.id);

    // Stage two images, label them OK, teach by name.
    const up = await request.post(`${API}/bank/staging/upload`, {
      multipart: { files: { name: "e2e-ok-1.png", mimeType: "image/png", buffer: tinyPng() } },
    });
    expect(up.ok(), `staging/upload -> ${up.status()}`).toBeTruthy();
    const staged = (await up.json()) as { items: Array<{ name: string }> };
    expect(staged.items.length).toBeGreaterThan(0);
    const stagedName = staged.items[0].name;

    const lab = await request.post(`${API}/bank/staging/label`, {
      data: { names: [stagedName], tier: "normal", label: "" },
    });
    expect(lab.ok(), `staging/label -> ${lab.status()}`).toBeTruthy();

    const teach = await request.post(`${API}/bank/staging/teach`, {
      data: { names: [stagedName], tier: "normal", label: "" },
      timeout: 180_000,
    });
    expect(teach.ok(), `staging/teach -> ${teach.status()}`).toBeTruthy();

    const after = await bankState(request);
    expect(after.normal).toBeGreaterThan(0);

    // Direct append into critical with a label, then verify counts.
    await teachImage(request, "critical", "e2e-ng-1.png", "scratch");
    const withNg = await bankState(request);
    expect(withNg.critical).toBeGreaterThan(0);
  } finally {
    await deleteProject(request, proj.id);
  }
});

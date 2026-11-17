// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/**
 * Global setup — verifies a live server before any spec runs.
 *
 * The suite is additive-only: every spec creates its own uniquely-named
 * `zz-e2e-*` project, works inside it, and deletes it afterwards. Nothing
 * here seeds shared state. Note that specs DO re-bind the server's active
 * bank (it is process-global); other browser tabs re-bind themselves, but
 * prefer a scratch CLS_PROJECTS_DIR instance for CI-like runs.
 */
import { request, type FullConfig } from "@playwright/test";

const API = `${process.env.CLS_E2E_BASE ?? "http://localhost:8791"}/api/v1`;

export default async function globalSetup(_config: FullConfig): Promise<void> {
  const ctx = await request.newContext();
  try {
    const res = await ctx.get(`${API}/health`);
    if (!res.ok()) {
      throw new Error(`API health returned ${res.status()}`);
    }
  } catch (err) {
    throw new Error(
      `cls-studio API is not reachable at ${API} — start the server ` +
      `(serving the built UI) before running e2e. (${String(err)})`,
    );
  } finally {
    await ctx.dispose();
  }
}

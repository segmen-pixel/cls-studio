// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import { readFileSync } from "node:fs";

import { defineConfig } from "@playwright/test";

// global-setup asks for "a scratch CLS_PROJECTS_DIR instance for CI-like
// runs", which was impossible while this address was hardcoded in three
// files. Set CLS_E2E_BASE to point the suite at one.
const BASE = process.env.CLS_E2E_BASE ?? "http://localhost:8791";

// The fixture records its localStorage against one origin, and a
// storageState origin that does not match the page is simply ignored --
// which brings the tutorial overlay back and swallows every click. Retarget
// it at whatever BASE says instead of keeping a second copy per port.
function seededState() {
  const state = JSON.parse(
    readFileSync("./e2e/fixtures/storage-state.json", "utf8"),
  ) as { cookies: unknown[]; origins: Array<{ origin: string }> };
  for (const o of state.origins) o.origin = BASE;
  return state;
}

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",
  testIgnore: ["_tools/**", "_archive/**"],
  outputDir: "./e2e/test-results",
  timeout: 60_000,
  retries: 0,
  workers: 1,
  use: {
    baseURL: `${BASE}/ui`,
    // Pre-seed localStorage so the first-launch tutorial overlay (which
    // intercepts pointer events) never auto-opens during e2e runs.
    storageState: seededState(),
    viewport: { width: 1920, height: 1080 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  // NOTE: run WITHOUT --reporter CLI flag — it would override this list and
  // silently drop the skip-budget gate.
  reporter: [
    ["dot"],
    ["html", { outputFolder: "./e2e/html-report", open: "never" }],
    ["./e2e/skip-budget-reporter.ts"],
  ],
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
});

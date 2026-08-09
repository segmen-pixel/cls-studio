// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/**
 * Switching projects must not leave the previous project's data on screen.
 *
 * Every workspace panel is mounted for the whole session and hidden with CSS,
 * so each one used to keep its own state across a switch — the Bank tab kept
 * its whole listing, its selection and its preview image. That is not only a
 * cosmetic problem: store entry ids are a per-store zero-padded counter, so
 * "000000" exists in EVERY project, which meant a selection carried over from
 * project A pointed at real, different entries in project B and the server
 * accepted writes against them.
 *
 * Both projects below are seeded with disjoint filenames and both banks hold
 * an entry "000000", which is the collision the fix has to survive.
 *
 * The first assertion is deliberately written against a MutationObserver
 * recording rather than as a plain expect: Playwright's auto-retry would
 * happily wait out a frame that showed the old project and then pass.
 */
import { expect, test, type Page } from "@playwright/test";
import {
  createProject, deleteProject, ingestAndAssemble, selectBank, uniqueName,
} from "../helpers/api";

// The first ingest loads DINOv2.
test.describe.configure({ timeout: 420_000 });

const PROJECTS_TAB = /^(プロジェクト|Projects)$/;
const BANK_TAB = /^(バンク|Bank)$/;
const A_FILE = "aaa-only-in-a.png";
const B_FILE = "bbb-only-in-b.png";

const bankRows = (page: Page) => page.locator(".bank-tab [data-row]");

/** Open a project the way a person does: through the Projects tab, inside the
 *  running app.
 *
 *  Deliberately NOT page.goto("/") — a reload remounts everything and clears
 *  every cache on its own, so a test that navigates between projects that way
 *  passes no matter what the code does. It is the in-app switch that used to
 *  leave the previous project's data on screen. (The app also restores the last
 *  project from sessionStorage on load, so a goto lands back where it was.) */
async function switchToProject(page: Page, name: string) {
  await page.getByRole("button", { name: PROJECTS_TAB }).first().click();
  // DOUBLE click: a tile's first click only SELECTS, the double click OPENS
  // (ProjectTile). Selecting and then reaching for the tab bar skips
  // openProjectWorkspace entirely -- which is how an earlier version of this
  // test ran green against a build whose project-switch guard deadlocked the
  // app on every real open.
  await page.getByText(name).first().dblclick();
  await page.getByRole("button", { name: BANK_TAB }).first().click();
}

test("the bank tab never paints the previous project's rows", async ({ page, request }) => {
  const a = await createProject(request, uniqueName("switch-a"));
  const b = await createProject(request, uniqueName("switch-b"));
  try {
    await selectBank(request, a.id);
    await ingestAndAssemble(request, [{ name: A_FILE, tier: "normal" }]);
    await selectBank(request, b.id);
    await ingestAndAssemble(request, [{ name: B_FILE, tier: "normal" }]);

    await page.goto("/");
    await switchToProject(page, a.name);
    await expect(bankRows(page).first()).toContainText(A_FILE);

    // Step onto the Projects tab FIRST, then start recording. The Bank panel
    // stays mounted (it is hidden with CSS, not unmounted), so its DOM still
    // holds project A's rows at this point -- recording from before the switch
    // would capture content that is legitimately still A's and fail on it. The
    // window that matters starts the instant project B is chosen.
    await page.getByRole("button", { name: PROJECTS_TAB }).first().click();
    await page.evaluate(() => {
      (window as unknown as { __seen: string[] }).__seen = [];
      new MutationObserver(() => {
        (window as unknown as { __seen: string[] }).__seen.push(
          document.querySelector(".bank-tab")?.textContent ?? "",
        );
      }).observe(document.body, { childList: true, subtree: true, characterData: true });
    });

    await page.getByText(b.name).first().dblclick();
    await page.getByRole("button", { name: BANK_TAB }).first().click();
    await expect(bankRows(page).first()).toContainText(B_FILE);

    const seen = await page.evaluate(
      () => (window as unknown as { __seen: string[] }).__seen.join("\n"),
    );
    expect(seen, "project A's filename was painted while project B was open").not.toContain(A_FILE);
  } finally {
    await deleteProject(request, a.id);
    await deleteProject(request, b.id);
  }
});

test("the preview image is re-requested, not reused, across a switch", async ({ page, request }) => {
  const a = await createProject(request, uniqueName("switch-img-a"));
  const b = await createProject(request, uniqueName("switch-img-b"));
  try {
    await selectBank(request, a.id);
    await ingestAndAssemble(request, [{ name: A_FILE, tier: "normal" }]);
    await selectBank(request, b.id);
    await ingestAndAssemble(request, [{ name: B_FILE, tier: "normal" }]);

    await page.goto("/");
    await switchToProject(page, a.name);
    await bankRows(page).first().click();
    await expect(page.locator(".bank-tab img").first()).toBeVisible();

    const imageRequests: string[] = [];
    page.on("request", (r) => {
      if (r.url().includes("/store/image/")) imageRequests.push(r.url());
    });

    await switchToProject(page, b.name);
    await bankRows(page).first().click();
    await expect(page.locator(".bank-tab img").first()).toBeVisible();

    // Entry ids collide across projects, so without the project in the URL the
    // src string is identical and the browser never asks for the new picture.
    expect(imageRequests.length, "no image request after the switch").toBeGreaterThan(0);
    const src = await page.locator(".bank-tab img").first().getAttribute("src");
    expect(src, "the image URL does not identify the project").toContain(b.id);
  } finally {
    await deleteProject(request, a.id);
    await deleteProject(request, b.id);
  }
});

test("switching projects leaves the app able to write to the new one", async ({ page, request }) => {
  // The regression this exists for: the binding guard was routed through the
  // same helper /bank/select uses, so the ONE call that resolves a mismatch was
  // refused BECAUSE of the mismatch. Every switch ended in "bank select failed"
  // and the bank never loaded. The two tests above did not catch it, because
  // they navigated by selecting a tile rather than opening it.
  const a = await createProject(request, uniqueName("switch-write-a"));
  const b = await createProject(request, uniqueName("switch-write-b"));
  try {
    await selectBank(request, a.id);
    await ingestAndAssemble(request, [{ name: A_FILE, tier: "normal" }]);
    await selectBank(request, b.id);
    await ingestAndAssemble(request, [{ name: B_FILE, tier: "normal" }]);

    const toasts: string[] = [];
    page.on("console", (m) => { if (m.type() === "error") toasts.push(m.text()); });

    await page.goto("/");
    await switchToProject(page, a.name);
    await expect(bankRows(page).first()).toContainText(A_FILE);
    await switchToProject(page, b.name);
    await expect(bankRows(page).first()).toContainText(B_FILE);

    // The bank loaded, so /bank/select was not refused. Now prove a MUTATION
    // reaches the new project: select the row and assign it a tier (key "2").
    await bankRows(page).first().click();
    await page.keyboard.press("2");
    await expect(page.locator(".bank-tab").getByText(/(不良|Defect)/).first()).toBeVisible();
    expect(toasts.join("\n")).not.toContain("switching project");
  } finally {
    await deleteProject(request, a.id);
    await deleteProject(request, b.id);
  }
});

#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/**
 * Capture the documentation screenshots listed in
 * docs/contributing/screenshots.md against a SCRATCH server.
 *
 * Never point this at a working instance. The Projects shot captures the
 * whole grid, so every project name on that server ends up in a public
 * repository -- which is why this seeds its own world and why the server it
 * talks to must have its own CLS_PROJECTS_DIR:
 *
 *   set CLS_PROJECTS_DIR=C:\somewhere\demo_projects
 *   .venv-windows\Scripts\python.exe -m uvicorn apps.api.app.main:app --port 8792
 *
 * Usage:
 *   node e2e/_tools/docs-screenshots.mjs --api http://localhost:8792 \
 *       --images C:\somewhere\demo_imgs [--lang en|ja]
 *
 * The images directory is read by filename prefix, any extension:
 *   ok_*      taught as normal
 *   ng_*      taught as critical, defect label taken from the name
 *             (ng_<label>_<n> -> "<label>")
 *   probe_*   dropped on the Inspect tab, never taught
 *
 * Output: e2e/screenshots/docs/<lang>/*.png
 */
import { chromium } from "playwright-core";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function parseArgs() {
  const a = process.argv.slice(2);
  const o = { api: "http://localhost:8792", images: null, lang: "en" };
  for (let i = 0; i < a.length; i++) {
    if (a[i] === "--api" && a[i + 1]) o.api = a[++i];
    else if (a[i] === "--images" && a[i + 1]) o.images = a[++i];
    else if (a[i] === "--lang" && a[i + 1]) o.lang = a[++i];
  }
  if (!o.images) throw new Error("--images <dir of ok_* / ng_* / probe_* files> is required");
  return o;
}

async function api(base, method, p, body) {
  const res = await fetch(`${base}/api/v1${p}`, {
    method,
    headers: body && !(body instanceof FormData) ? { "Content-Type": "application/json" } : undefined,
    body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${method} ${p} -> ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return res.json();
}

const MIME = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp" };

function ingestForm(files) {
  const fd = new FormData();
  for (const fp of files) {
    const type = MIME[path.extname(fp).toLowerCase()] || "application/octet-stream";
    fd.append("images", new Blob([fs.readFileSync(fp)], { type }), path.basename(fp));
  }
  fd.append("tier", "");
  return fd;
}

/** ng_scratch_2.jpg -> "scratch"; ng_3.jpg -> "" */
function defectLabel(name) {
  const m = /^ng_(.+)_\d+\.[^.]+$/.exec(name);
  return m ? m[1] : "";
}

async function main() {
  const opts = parseArgs();
  const outDir = path.join(__dirname, "..", "screenshots", "docs", opts.lang);
  fs.mkdirSync(outDir, { recursive: true });

  const all = fs.readdirSync(opts.images).sort();
  const pick = (p) => all.filter((n) => n.startsWith(p)).map((n) => path.join(opts.images, n));
  const okFiles = pick("ok_"), ngFiles = pick("ng_"), probes = pick("probe_");
  if (!okFiles.length || !ngFiles.length) throw new Error("need at least one ok_* and one ng_*");

  // ---- seed a demo world over the API ------------------------------------
  // Same three moves the bank tab makes: get the images in, label them, fold
  // the labels into the bank. Nothing here writes to the bank directly --
  // that second writer is exactly what the UI stopped having.
  console.log("[seed] projects");
  const proj = await api(opts.api, "POST", "/projects", { name: "hash-brown-line-A" });
  await api(opts.api, "POST", "/projects", { name: "hash-brown-line-B" }); // the grid needs company
  await api(opts.api, "POST", "/bank/select", { project_id: proj.id });

  console.log(`[seed] ingest ${okFiles.length} ok + ${ngFiles.length} ng`);
  await api(opts.api, "POST", "/store/ingest", ingestForm([...okFiles, ...ngFiles]));

  const store = await api(opts.api, "GET", "/store");
  const idOf = (name) => {
    const e = store.images.find((x) => x.name === name);
    if (!e) throw new Error(`store has no entry for ${name}`);
    return e.id;
  };

  console.log("[seed] label");
  await api(opts.api, "POST", "/labelsets/assign",
            { ids: okFiles.map((f) => idOf(path.basename(f))), tier: "normal", label: "" });
  // One assign call per defect label, so the bank grows the labels a real one
  // would have rather than one anonymous "NG" heap.
  const byLabel = new Map();
  for (const f of ngFiles) {
    const n = path.basename(f);
    const l = defectLabel(n);
    if (!byLabel.has(l)) byLabel.set(l, []);
    byLabel.get(l).push(idOf(n));
  }
  for (const [label, ids] of byLabel) {
    await api(opts.api, "POST", "/labelsets/assign", { ids, tier: "critical", label, severity: 2 });
  }

  console.log("[seed] defect marks");
  const markedName = path.basename(ngFiles[0]);
  await api(opts.api, "POST", "/labelsets/mark", {
    id: idOf(markedName),
    rects: [{ x: 0.28, y: 0.30, w: 0.30, h: 0.28 }, { x: 0.52, y: 0.55, w: 0.20, h: 0.18 }],
  });

  console.log("[seed] assemble");
  await api(opts.api, "POST", "/bank/assemble", {});

  // ---- drive the UI ------------------------------------------------------
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.setDefaultTimeout(120_000);
  await page.addInitScript(() => {
    // Same state the e2e storage fixture seeds — the first-launch tutorial
    // overlay intercepts every pointer event otherwise.
    localStorage.setItem(
      "seg-tutorial-state",
      JSON.stringify({ completed: true, skipped: true, lastStep: 0, mode: null }),
    );
  });
  await page.goto(`${opts.api}/ui/`);

  // Language: the header toggle flips ja <-> en; land on the requested one.
  const bankTab = page.getByRole("button", { name: /^(バンク|Bank)$/ });
  await bankTab.waitFor();
  const isJa = (await bankTab.textContent()) === "バンク";
  if ((opts.lang === "en") === isJa) {
    await page.getByRole("button", { name: /^(EN|JA|日本語|English)$/ }).first().click();
    await page.waitForTimeout(500);
  }

  const tab = (re) => page.getByRole("button", { name: re }).first();
  const TAB = {
    bank: /^(バンク|Bank)$/,
    teach: /^(学習|Teach)$/,
    inspect: /^(検査|Inspect)$/,
  };

  const shot = async (name) => {
    // Park the pointer in dead space first — a hover tooltip burnt into a
    // published screenshot reads as UI noise.
    await page.mouse.move(8, 560);
    await page.waitForTimeout(700); // let transitions + tooltip fade settle
    await page.screenshot({ path: path.join(outDir, name) });
    console.log("[shot]", name);
  };
  // Wait until NO blocking dialog has been on screen for a little while —
  // the sweep dialog is chained straight into the heatmap pre-render one,
  // so a single waitFor(hidden) lands in the gap between them.
  const settleDialogs = async () => {
    for (let quiet = 0; quiet < 3; ) {
      if (await page.getByRole("alertdialog").count()) { quiet = 0; await page.waitForTimeout(1000); }
      else { quiet += 1; await page.waitForTimeout(400); }
    }
  };

  await shot("projects.png");

  await page.getByText(proj.name).first().click();
  await page.waitForTimeout(1200);

  // The bank tab: the numbered steps down the right, the labelled list on the
  // left. This is where images come in and where the judgement is made.
  await tab(TAB.bank).click();
  await page.waitForTimeout(2000);
  await shot("bank_label.png");

  // Defect marks live here too, on the image they belong to.
  for (const row of await page.getByText(markedName, { exact: true }).all()) {
    if (await row.isVisible()) { await row.click(); break; }
  }
  await page.waitForTimeout(1500);
  await shot("bank_marks.png");

  // The teach tab: sweep the bank and read how far OK and NG separate.
  await tab(TAB.teach).click();
  await page.waitForTimeout(1200);
  await page.getByRole("button", { name: /(評価を実行|Run evaluation)/ }).first().click();
  await settleDialogs();
  // Put an image in the viewer: an empty pane reads "select an image from the
  // list", which is not what this shot is meant to show.
  await page.locator("[data-rowkey]").first().click().catch(() => {});
  await page.waitForTimeout(2000);
  await shot("check_histogram.png");

  // Same tab, the feature map: what the bank looks like from above.
  const mapBtn = page.getByRole("button", { name: /^(特徴分離マップ|Feature separation map)$/ }).first();
  if (await mapBtn.count()) {
    await mapBtn.click();
    await page.waitForTimeout(2500);
    await shot("check_map.png");
  }

  // The inspect tab: probes that were never taught, scored and heat-mapped.
  await tab(TAB.inspect).click();
  await page.waitForTimeout(800);
  const input = page.locator('input[type="file"][multiple]').last();
  await input.setInputFiles(probes);
  await page.getByRole("img", { name: "anomaly heatmap" }).first().waitFor({ timeout: 300_000 });
  await page.waitForTimeout(1000);
  const ngProbe = probes.map((p) => path.basename(p)).find((n) => n.startsWith("probe_ng"));
  if (ngProbe) await page.getByText(ngProbe).first().click();
  // The heatmap toggle chip states its own polarity; only press H when off.
  if (await page.getByText(/Heatmap OFF|ヒートマップ OFF/i).count()) await page.keyboard.press("h");
  await page.waitForTimeout(900);
  await shot("inspect_queue.png");
  await shot("hero.png");

  await browser.close();
  console.log("done ->", outDir);
}

main().catch((e) => { console.error(e); process.exit(1); });

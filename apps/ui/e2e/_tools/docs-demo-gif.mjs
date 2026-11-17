#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/**
 * Capture the frames for docs/images/hero.gif — the README's moving demo:
 * the assembled bank → the separation check → an OK probe → an NG probe →
 * the NG probe with the heatmap on the defect. Five stills, assembled into a
 * stepped GIF by assemble-gif.py.
 *
 * Same rules as docs-screenshots.mjs: SCRATCH server only, demo images only.
 * See docs/contributing/screenshots.md.
 *
 * Usage:
 *   node e2e/_tools/docs-demo-gif.mjs --api http://localhost:8792 \
 *       --images <demo images dir> [--lang en|ja]
 *
 * Images are read by filename prefix, any extension: ok_* / ng_<kind>_<n> /
 * probe_*.
 *
 * Output: e2e/screenshots/docs/gif/<lang>/frame_*.png (1280x720)
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
  if (!o.images) throw new Error("--images <dir> is required");
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
  const outDir = path.join(__dirname, "..", "screenshots", "docs", "gif", opts.lang);
  fs.mkdirSync(outDir, { recursive: true });
  for (const f of fs.readdirSync(outDir)) fs.unlinkSync(path.join(outDir, f));

  const all = fs.readdirSync(opts.images).sort();
  const pick = (p) => all.filter((n) => n.startsWith(p)).map((n) => path.join(opts.images, n));
  const okFiles = pick("ok_"), ngFiles = pick("ng_");
  const probeOk = pick("probe_ok")[0], probeNg = pick("probe_ng")[0];
  if (!okFiles.length || !ngFiles.length || !probeOk || !probeNg) {
    throw new Error("need ok_*, ng_*, probe_ok* and probe_ng* in the images dir");
  }

  // Seeded the way the Bank tab does it: ingest, label, assemble.
  console.log("[seed] ingest");
  const proj = await api(opts.api, "POST", "/projects", { name: "hash-brown-line-A" });
  await api(opts.api, "POST", "/bank/select", { project_id: proj.id });
  await api(opts.api, "POST", "/store/ingest", ingestForm([...okFiles, ...ngFiles]));

  const store = await api(opts.api, "GET", "/store");
  const idOf = (n) => {
    const e = store.images.find((x) => x.name === n);
    if (!e) throw new Error(`store has no entry for ${n}`);
    return e.id;
  };

  console.log("[seed] label + assemble");
  await api(opts.api, "POST", "/labelsets/assign",
            { ids: okFiles.map((f) => idOf(path.basename(f))), tier: "normal", label: "" });
  const byLabel = new Map();
  for (const f of ngFiles) {
    const n = path.basename(f), l = defectLabel(n);
    if (!byLabel.has(l)) byLabel.set(l, []);
    byLabel.get(l).push(idOf(n));
  }
  for (const [label, ids] of byLabel) {
    await api(opts.api, "POST", "/labelsets/assign", { ids, tier: "critical", label, severity: 2 });
  }
  await api(opts.api, "POST", "/bank/assemble", {});

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.setDefaultTimeout(120_000);
  await page.addInitScript(() => {
    localStorage.setItem(
      "seg-tutorial-state",
      JSON.stringify({ completed: true, skipped: true, lastStep: 0, mode: null }),
    );
  });
  await page.goto(`${opts.api}/ui/`);

  const bankTab = page.getByRole("button", { name: /^(バンク|Bank)$/ });
  await bankTab.waitFor();
  const isJa = (await bankTab.textContent()) === "バンク";
  if ((opts.lang === "en") === isJa) {
    await page.getByRole("button", { name: /^(EN|JA|日本語|English)$/ }).first().click();
    await page.waitForTimeout(500);
  }
  const tab = (re) => page.getByRole("button", { name: re }).first();
  const TAB = { bank: /^(バンク|Bank)$/, teach: /^(学習|Teach)$/, inspect: /^(検査|Inspect)$/ };

  let n = 0;
  const frame = async () => {
    await page.mouse.move(8, 400);
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(outDir, `frame_${n}.png`) });
    console.log("[frame]", n++);
  };
  const settleDialogs = async () => {
    for (let quiet = 0; quiet < 3; ) {
      if (await page.getByRole("alertdialog").count()) { quiet = 0; await page.waitForTimeout(1000); }
      else { quiet += 1; await page.waitForTimeout(400); }
    }
  };

  // 0: the labelled, assembled bank — what "teaching" looks like now
  await page.getByText(proj.name).first().click();
  await page.waitForTimeout(1200);
  await tab(TAB.bank).click();
  await page.waitForTimeout(2000);
  await frame();

  // 1: the separation check after the sweep
  await tab(TAB.teach).click();
  await page.waitForTimeout(1200);
  await page.getByRole("button", { name: /(評価を実行|Run evaluation)/ }).first().click();
  await settleDialogs();
  await page.locator("[data-rowkey]").first().click().catch(() => {});
  await page.waitForTimeout(1500);
  await frame();

  // 2: an OK probe, plain
  await tab(TAB.inspect).click();
  await page.waitForTimeout(800);
  const input = page.locator('input[type="file"][multiple]').last();
  await input.setInputFiles([probeOk]);
  await page.getByRole("img", { name: "anomaly heatmap" }).first().waitFor({ timeout: 300_000 });
  await page.waitForTimeout(600);
  if (await page.getByText(/Heatmap ON|ヒートマップ ON/i).count()) await page.keyboard.press("h");
  await page.waitForTimeout(400);
  await frame();

  // 3: the NG probe, plain
  await input.setInputFiles([probeNg]);
  await page.waitForTimeout(3000);
  await page.getByText(path.basename(probeNg)).first().click();
  await page.waitForTimeout(600);
  await frame();

  // 4: the NG probe with the heatmap on the defect
  if (await page.getByText(/Heatmap OFF|ヒートマップ OFF/i).count()) await page.keyboard.press("h");
  await page.waitForTimeout(800);
  await frame();

  await browser.close();
  console.log("done ->", outDir);
}

main().catch((e) => { console.error(e); process.exit(1); });

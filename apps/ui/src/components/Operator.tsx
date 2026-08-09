// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// Inline icons in this file use SVG path data from Feather Icons
// (https://github.com/feathericons/feather), MIT licensed,
// Copyright (c) 2013-2023 Cole Bemis. Some paths are adapted.
//
// Operator tab — inspect images against the active project's bank and get a
// live anomaly heatmap + OK/NG verdict per image. Any number of images can be
// dropped anywhere on the tab; they are scored sequentially (the GPU does one
// cdist batch at a time) into a Develop-style list + viewer. Finished results
// are persisted server-side (/inspections) and restored on mount, so a
// browser reload doesn't erase what the backend already computed; only
// not-yet-scored uploads die with the page (beforeunload warns while the
// queue is running). The verdict compares each image's top-k mean (k=10) —
// the same statistic as the Teach tab's separation check — against a
// threshold seeded from that check's suggestion, so tuning done on the
// Teach tab carries over to inspection unchanged.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n";
import {
  ALPHA_MAX,
  clearInspections,
  deleteInspection,
  fetchCachedEvaluations,
  fetchEvalCmin,
  fetchInspections,
  fetchRuntimeConfig,
  inspectionFileUrl,
  okPatchFloor,
  saveRuntimeConfig,
  scoreImage,
  selectBank,
  topkComposite,
  type StoredImageEval,
} from "../api/cls";
import { isAbortError } from "../api/shared";
import BankStamp from "./BankStamp";
import ImageViewer from "../ui/ImageViewer";
import { useZoomPan } from "../ui/useZoomPan";
import { ACCENT, BORDER, DANGER, INK, MUTED, OK, PANEL_2, RULE, TYPE, VERM } from "../ui/tokens";

type Props = {
  projectId: string | null;
  active: boolean;
  showToast: (msg: string) => void;
};

// k of the image-level top-k mean; matches the separation check's default.
const TOPK = 10;

type InspectStatus = "pending" | "scoring" | "done" | "error";
type InspectScores = {
  topk: number;
  max: number;
  p99: number;
  nExemplar: number;
  serverMs?: number;
};
type InspectItem = {
  id: number;
  name: string;
  url?: string; // object URL of the upload — heatmap-OFF fallback for old servers
  status: InspectStatus;
  scores?: InspectScores;
  heatSrc?: string; // data: URI (fresh score) or server URL (restored)
  origSrc?: string;
  serverId?: string; // persisted inspection-log entry id (per-item delete)
  ms?: number; // client round trip for fresh scores; server ms for restored
  err?: string;
};

export default function Operator({ projectId, active, showToast }: Props) {
  const { t } = useI18n();
  const [items, setItems] = useState<InspectItem[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [device, setDevice] = useState<string>("");
  // α is shared with the Teach tab's slider via localStorage so the value
  // proven on the separation check is what inspection runs with.
  const [alpha, setAlpha] = useState<number>(() => {
    const v = Number(localStorage.getItem("anom.alpha"));
    // Clamped rather than discarded — see the same read in Develop.tsx.
    return Number.isFinite(v) && v > 0 ? Math.min(v, ALPHA_MAX) : 0;
  });
  // β subtracts by proximity to the FP-suppression tier. It was hardcoded to
  // 0 everywhere, so teaching an image into that tier did nothing at all at
  // inspection time.
  const [beta, setBeta] = useState<number>(() => {
    const v = Number(localStorage.getItem("anom.beta"));
    return Number.isFinite(v) && v > 0 ? Math.min(v, ALPHA_MAX) : 0;
  });
  const [thr, setThr] = useState<number | null>(null);
  const thrEdited = useRef(false);
  const [evals, setEvals] = useState<StoredImageEval[]>([]);
  const [cminMap, setCminMap] = useState<Map<string, number[]>>(new Map());
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => { localStorage.setItem("anom.alpha", String(alpha)); }, [alpha]);
  useEffect(() => { localStorage.setItem("anom.beta", String(beta)); }, [beta]);

  // True when the current threshold/alpha came from the bank's saved
  // runtime config (the deployable recipe) rather than the live suggestion.
  const [fromConfig, setFromConfig] = useState(false);
  // The bank changed (teach / delete / mark) after the recipe was saved.
  const [cfgStale, setCfgStale] = useState(false);

  const idSeq = useRef(0);
  // Mirror for unmount cleanup + restore race checks (state isn't readable
  // from an unmount cleanup, and updaters must stay side-effect-free).
  const itemsRef = useRef<InspectItem[]>([]);
  useEffect(() => { itemsRef.current = items; }, [items]);
  useEffect(() => () => {
    itemsRef.current.forEach((it) => { if (it.url) URL.revokeObjectURL(it.url); });
  }, []);

  // Which project the current items belong to — switching projects drops
  // them (they'd be another bank's verdicts) before the new restore lands.
  const itemsProjectRef = useRef<string | null>(null);

  useEffect(() => {
    if (!active || !projectId) return;
    let cancelled = false;
    // α and β are shared with the Teach tab through localStorage, and both
    // panels stay mounted for the session — so each only ever read the value
    // that existed when it first rendered. Tuning α on the check and coming
    // here left this tab inspecting at its mount value, usually 0, with the
    // two sliders visibly disagreeing and "Save recipe" committing the wrong
    // number. Re-read on every activation.
    const savedAlpha = Number(localStorage.getItem("anom.alpha"));
    if (Number.isFinite(savedAlpha)) setAlpha(Math.min(Math.max(savedAlpha, 0), ALPHA_MAX));
    const savedBeta = Number(localStorage.getItem("anom.beta"));
    if (Number.isFinite(savedBeta)) setBeta(Math.min(Math.max(savedBeta, 0), ALPHA_MAX));

    setEvals([]); setFromConfig(false); setCfgStale(false);
    if (itemsProjectRef.current !== projectId) {
      itemsProjectRef.current = projectId;
      // thr resets with the project, NOT with every activation: it used to sit
      // above this guard, so switching to Teach and back threw away a
      // hand-typed threshold and silently flipped every scored row's verdict
      // back. A project with no saved config still gets a clean null here,
      // which is what stops it judging against the previous project's number.
      setThr(null);
      thrEdited.current = false;
      // Stop the queue before wiping the list. Otherwise the pump keeps
      // scoring the old project's files against the newly bound bank, with
      // its rows — and its cancel button — already gone from the UI, so
      // there is no way left to stop it. Written out rather than calling
      // requestCancel(): that callback is declared further down, so naming it
      // in this effect's dependency array would read it before initialisation.
      cancelRef.current = true;
      abortRef.current?.abort();
      queueRef.current = [];
      setItems((prev) => {
        prev.forEach((it) => { if (it.url) URL.revokeObjectURL(it.url); });
        return [];
      });
      setSelectedId(null);
    }
    selectBank(projectId)
      .then(async (r) => {
        if (cancelled) return;
        setDevice(r.device);
        // A saved runtime config is the bank's committed recipe — it wins
        // over the live suggestion until the operator edits by hand.
        try {
          const cfg = await fetchRuntimeConfig();
          if (!cancelled && cfg) {
            // Clamped like the localStorage read above. Recipes written before
            // the slider's range was lowered carry values well past the
            // ceiling (the package tests round-trip 500 and 750); unclamped,
            // the thumb pins at one end while the readout shows the old
            // number, and the first nudge silently discards it.
            setAlpha(Math.min(Math.max(cfg.alpha, 0), ALPHA_MAX));
            setBeta(Math.min(Math.max(cfg.beta ?? 0, 0), ALPHA_MAX));
            setCfgStale(!!cfg.stale);
            if (cfg.threshold != null) {
              setThr(cfg.threshold);
              thrEdited.current = true;
              setFromConfig(true);
            }
          }
        } catch { /* no saved config */ }
        try {
          const rows = await fetchCachedEvaluations();
          if (!cancelled) setEvals(rows);
        } catch { /* no cached evals — threshold stays manual */ }
        // Restore persisted results — only into an idle, empty list (a live
        // queue or an already-restored session must not be clobbered).
        try {
          const { entries } = await fetchInspections();
          if (!cancelled && entries.length && itemsRef.current.length === 0) {
            const restored = entries.map((e) => ({
              id: ++idSeq.current,
              name: e.name,
              status: "done" as InspectStatus,
              scores: {
                topk: e.topk_score, max: e.max_score, p99: e.p99_score,
                nExemplar: e.n_exemplar_rows, serverMs: e.server_ms,
              },
              heatSrc: e.heat ? inspectionFileUrl(e.heat) : undefined,
              origSrc: e.orig ? inspectionFileUrl(e.orig) : undefined,
              serverId: e.id,
              ms: Math.round(e.server_ms),
            }));
            setItems(restored);
            setSelectedId(restored[restored.length - 1]?.id ?? null);
          }
        } catch { /* no log yet */ }
      })
      .catch((e) => showToast(`bank select failed: ${e.message}`));
    return () => { cancelled = true; };
  }, [active, projectId, showToast]);

  const saveConfig = useCallback(async () => {
    try {
      await saveRuntimeConfig({
        topk: TOPK, k: 5, alpha, beta, exemplar_alpha: true, threshold: thr,
      });
      setFromConfig(true);
      setCfgStale(false);
      showToast(t("operator.configSaved"));
    } catch (e) {
      showToast(`config save failed: ${(e as Error).message}`);
    }
  }, [alpha, beta, thr, showToast, t]);

  // cmin is only needed while α > 0; fetched once per bank / α on-off flip.
  const alphaOn = alpha > 0;
  useEffect(() => {
    if (!alphaOn || evals.length === 0) { setCminMap(new Map()); return; }
    let cancelled = false;
    fetchEvalCmin()
      .then((rows) => {
        if (cancelled) return;
        setCminMap(new Map(rows.map((r) => [`${r.tier}/${r.label}/${r.name}`, r.top_cmin])));
      })
      .catch(() => { if (!cancelled) setCminMap(new Map()); });
    return () => { cancelled = true; };
  }, [alphaOn, evals]);

  // Suggested threshold: Youden's J over the cached evals under the current
  // α — the same rule the separation check uses for its dashed line.
  const suggested = useMemo(() => {
    if (!evals.length) return null;
    const val = (r: StoredImageEval) =>
      topkComposite(r, TOPK, alpha, cminMap.get(`${r.tier}/${r.label}/${r.name}`));
    const ok = evals.filter((r) => r.tier === "normal").map(val);
    const ng = evals.filter((r) => r.tier === "critical").map(val);
    if (!ok.length || !ng.length) return null;
    const cand = [...ok, ...ng].sort((a, b) => a - b);
    let best = cand[0], bestJ = -Infinity;
    for (let c = 0; c < cand.length - 1; c++) {
      const m = (cand[c] + cand[c + 1]) / 2;
      const tpr = ng.filter((v) => v > m).length / ng.length;
      const fpr = ok.filter((v) => v > m).length / ok.length;
      if (tpr - fpr > bestJ) { bestJ = tpr - fpr; best = m; }
    }
    return best;
  }, [evals, cminMap, alpha]);

  // Follow the suggestion until the operator overrides it by hand.
  useEffect(() => {
    if (suggested != null && !thrEdited.current) setThr(suggested);
  }, [suggested]);

  // ---- multi-image inspection queue ---------------------------------------
  // Files land in queueRef and are scored one at a time by pump(): the GPU
  // runs one cdist batch at a time anyway, and per-image completion keeps the
  // list live. α mid-queue edits apply to not-yet-scored items (ref).
  const queueRef = useRef<{ id: number; file: File }[]>([]);
  const pumping = useRef(false);
  const cancelRef = useRef(false);
  // Aborts the score that is in flight right now. Without it a cancel can only
  // be honoured after the current request returns, which on a large image is
  // tens of seconds of a UI that looks like the button did nothing.
  const abortRef = useRef<AbortController | null>(null);
  // Rendered state, unlike cancelRef: the click has to change something on
  // screen immediately, or a cancel that is merely *fast* still reads as dead.
  const [cancelling, setCancelling] = useState(false);

  // Stop the queue when this panel goes away. It is remounted on a project
  // switch (App.tsx keys the workspace panels on the project id), and without
  // this the pump kept scoring the OLD project's files -- against the NEW
  // project's bank, since the binding is global and BankTab's own select had
  // already moved it -- and persisted them into the new project's inspection
  // log as real rows that survive a reload.
  //
  // Deliberately keyed on nothing, so it fires on unmount only: a TAB switch
  // must NOT stop the queue (the panel stays mounted for that), which is the
  // behaviour the comment above pump() relies on.
  //
  // Placed here rather than beside the object-URL cleanup further up because
  // these two refs are declared below that one; a cleanup closure would read
  // them fine, but the file has already been bitten once by declaration order
  // (see the note in the project effect).
  useEffect(() => () => {
    cancelRef.current = true;
    abortRef.current?.abort();
    queueRef.current = [];
  }, []);
  const alphaRef = useRef(alpha);
  useEffect(() => { alphaRef.current = alpha; }, [alpha]);
  const betaRef = useRef(beta);
  useEffect(() => { betaRef.current = beta; }, [beta]);

  // Heatmap anchors for the inspection view. Without them /score falls back to
  // renormalising each image against its OWN 5th..99th percentile, which by
  // construction paints the hottest 1% of EVERY image full red -- a
  // defect-free part included. With them the server draws the absolute
  // blue → white → vermilion map, where blue really means "OK level".
  //
  // Null rather than approximate, in three cases where the anchors would not
  // describe the field the server is about to compose:
  //   * β > 0 — /score subtracts a per-patch β term the cached evals do not
  //     carry, so anchors derived from them would sit above the field.
  //   * α > 0 with no cmin yet — cmin is fetched after the evals, and the
  //     composite silently degrades to the raw scale without it, which would
  //     put the blue end too low for one render and then move it.
  //   * no threshold, or a threshold under the floor.
  // In each case the previous rendering is kept: a wrong absolute scale is
  // worse than an honest relative one.
  const heatAnchors = useMemo(() => {
    if (beta !== 0 || thr == null || !evals.length) return null;
    if (alpha > 0 && cminMap.size === 0) return null;
    const lo = okPatchFloor(evals, alpha, (r) => cminMap.get(`${r.tier}/${r.label}/${r.name}`));
    if (lo == null || !(thr > lo)) return null;
    return { lo, hi: thr };
  }, [evals, cminMap, alpha, beta, thr]);
  // pump() reads this after an await, for the same reason α and β are refs.
  const heatAnchorsRef = useRef(heatAnchors);
  useEffect(() => { heatAnchorsRef.current = heatAnchors; }, [heatAnchors]);

  const patchItem = (id: number, patch: Partial<InspectItem>) =>
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)));

  // Cancelled work did not happen, so its rows leave the list — a half-scored
  // row left behind is indistinguishable from a real inspection.
  const dropItems = useCallback((ids: number[]) => {
    setItems((prev) => prev.filter((it) => {
      if (!ids.includes(it.id)) return true;
      if (it.url) URL.revokeObjectURL(it.url); // double-revoke under StrictMode is a no-op
      return false;
    }));
  }, []);

  const requestCancel = useCallback(() => {
    cancelRef.current = true;
    setCancelling(true);
    abortRef.current?.abort();
  }, []);

  const pump = useCallback(async () => {
    if (pumping.current) return;
    pumping.current = true;
    cancelRef.current = false;
    setCancelling(false);
    try {
      while (queueRef.current.length) {
        if (cancelRef.current) break;
        const job = queueRef.current.shift()!;
        patchItem(job.id, { status: "scoring" });
        const t0 = performance.now();
        const ac = new AbortController();
        abortRef.current = ac;
        try {
          const anchors = heatAnchorsRef.current;
          const r = await scoreImage(job.file, {
            k: 5, topk: TOPK, alpha: alphaRef.current, beta: betaRef.current,
            hmLo: anchors?.lo, hmHi: anchors?.hi, signal: ac.signal,
          });
          // Re-check AFTER the await. `shift()` above already emptied the queue
          // for the last (or only) image, so the `while` test would exit here
          // without ever reading the flag again — which made cancelling a
          // single-image inspection a literal no-op: the result was committed
          // and persisted exactly as if the button had never been pressed
          // (2026-07-31).
          if (cancelRef.current) { dropItems([job.id]); break; }
          patchItem(job.id, {
            status: "done",
            scores: {
              topk: r.topk_score, max: r.max_score, p99: r.p99_score,
              nExemplar: r.n_exemplar_rows, serverMs: r.timings?.total_server_ms,
            },
            heatSrc: `data:image/png;base64,${r.heatmap_png_base64}`,
            origSrc: r.original_jpeg_base64 ? `data:image/jpeg;base64,${r.original_jpeg_base64}` : undefined,
            serverId: r.inspection_id || undefined,
            ms: Math.round(performance.now() - t0),
          });
          setSelectedId(job.id); // follow the newest result while the queue drains
        } catch (e) {
          // An abort is the user getting what they asked for, not a failure:
          // no toast, no error row.
          if (isAbortError(e) || cancelRef.current) { dropItems([job.id]); break; }
          patchItem(job.id, { status: "error", err: (e as Error).message });
          showToast(`score failed (${job.file.name}): ${(e as Error).message}`);
          // 409 = the active bank/project changed (deleted or re-bound by
          // another client). Every remaining job would fail the same way —
          // stop hammering the server and drain the queue as cancelled.
          if ((e as { status?: number }).status === 409) {
            cancelRef.current = true;
          }
        } finally {
          abortRef.current = null;
        }
      }
      if (cancelRef.current) dropItems(queueRef.current.splice(0).map((d) => d.id));
    } finally {
      pumping.current = false;
      abortRef.current = null;
      // Clear on the way OUT, not only on the way in. enqueue()'s pump() call
      // short-circuits while `pumping` is true, so a flag left latched here
      // makes the next drop land in a queue that the resumed loop immediately
      // splices away — the user's files vanish with no explanation.
      cancelRef.current = false;
      setCancelling(false);
    }
  }, [dropItems, showToast]);

  const enqueue = useCallback((files: FileList | File[]) => {
    if (!projectId) { showToast(t("operator.selectFirst")); return; }
    const list = Array.from(files).filter(
      (f) => f.type.startsWith("image/") || /\.(png|jpe?g|bmp|webp|tiff?)$/i.test(f.name),
    );
    if (!list.length) return;
    const fresh = list.map((f) => {
      const id = ++idSeq.current;
      queueRef.current.push({ id, file: f });
      return { id, name: f.name, url: URL.createObjectURL(f), status: "pending" as InspectStatus };
    });
    setItems((prev) => [...prev, ...fresh]);
    void pump();
  }, [projectId, pump, showToast, t]);

  const clearAll = useCallback(() => {
    // Abort, don't just flag: an in-flight score that is allowed to finish
    // appends its inspection *after* the DELETE below and the image comes
    // straight back on the next mount.
    requestCancel();
    queueRef.current = [];
    setItems((prev) => { prev.forEach((it) => { if (it.url) URL.revokeObjectURL(it.url); }); return []; });
    setSelectedId(null);
    // Drop the server-side log too — clear means clear, not "hide until the
    // next reload resurrects everything". Best-effort.
    void clearInspections().catch(() => {});
  }, [requestCancel]);

  // Not-yet-scored uploads exist only in this page's memory: warn before an
  // accidental reload/close while the queue is still draining.
  useEffect(() => {
    const busy = items.some((it) => it.status === "pending" || it.status === "scoring");
    if (!busy) return;
    const h = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ""; };
    window.addEventListener("beforeunload", h);
    return () => window.removeEventListener("beforeunload", h);
  }, [items]);

  const selectedIdRef = useRef<number | null>(null);
  useEffect(() => { selectedIdRef.current = selectedId; }, [selectedId]);

  // Remove one inspection: list row + object URL + the persisted log entry.
  // Selection falls to the neighbour, matching the Develop list's delete.
  const removeSelected = useCallback(() => {
    const list = itemsRef.current;
    const idx = list.findIndex((it) => it.id === selectedIdRef.current);
    if (idx < 0) return;
    const victim = list[idx];
    if (victim.status === "scoring") return; // let the in-flight request finish
    queueRef.current = queueRef.current.filter((q) => q.id !== victim.id);
    if (victim.url) URL.revokeObjectURL(victim.url);
    if (victim.serverId) void deleteInspection(victim.serverId).catch(() => {});
    const neighbour = list[idx + 1] ?? list[idx - 1] ?? null;
    setItems((prev) => prev.filter((it) => it.id !== victim.id));
    setSelectedId(neighbour ? neighbour.id : null);
  }, []);

  // ---- keyboard: same grammar as the Teach tab ---------------------------
  // H = heatmap on/off, ↑/↓ = walk the list, Delete/Backspace = remove the
  // selected image, Esc = deselect. Window-level (no focus dance), ignored
  // while typing in a field so sliders / number inputs keep native keys.
  const [heatOn, setHeatOn] = useState(true);
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      if (e.key.toLowerCase() === "h") {
        setHeatOn((v) => !v);
        return;
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        if (selectedIdRef.current != null) { e.preventDefault(); removeSelected(); }
        return;
      }
      if (e.key === "Escape") {
        setSelectedId(null);
        return;
      }
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        const list = itemsRef.current;
        if (!list.length) return;
        e.preventDefault(); // keep the page from scrolling under the list
        const idx = list.findIndex((it) => it.id === selectedIdRef.current);
        const next = e.key === "ArrowDown"
          ? Math.min(list.length - 1, idx < 0 ? 0 : idx + 1)
          : Math.max(0, idx < 0 ? 0 : idx - 1);
        const id = list[next].id;
        setSelectedId(id);
        document.getElementById(`insp-row-${id}`)?.scrollIntoView({ block: "nearest" });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, removeSelected]);

  // Wheel = zoom around the cursor, drag / space+drag / middle-drag = pan
  // while zoomed, double-click = fit. Zoom resets per image.
  const zoom = useZoomPan(selectedId, active);

  // ---- whole-tab drop target ----------------------------------------------
  // dragenter/leave fire per child element; a counter keeps the overlay
  // stable while the pointer crosses inner borders.
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);

  if (!projectId) return <div className="muted" style={{ padding: 32 }}>{t("operator.selectFirst")}</div>;

  const selected = items.find((it) => it.id === selectedId) ?? null;
  const scores = selected?.scores ?? null;
  const ng = scores !== null && thr !== null && scores.topk > thr;
  const verdict = scores === null || thr === null ? null : ng ? "NG" : "OK";
  const border = RULE;

  const doneCount = items.filter((it) => it.status === "done").length;
  const errCount = items.filter((it) => it.status === "error").length;
  const running = items.some((it) => it.status === "pending" || it.status === "scoring");
  const okCount = thr == null ? 0 : items.filter((it) => it.scores && it.scores.topk <= thr).length;
  const ngCount = thr == null ? 0 : items.filter((it) => it.scores && it.scores.topk > thr).length;

  const verdictOf = (it: InspectItem): "OK" | "NG" | null =>
    it.scores && thr != null ? (it.scores.topk > thr ? "NG" : "OK") : null;

  const badge = (it: InspectItem) => {
    const v = verdictOf(it);
    if (it.status === "scoring") return <span className="train-spinner" style={{ flex: "none" }} />;
    if (it.status === "error") return <b style={{ flex: "none", fontSize: TYPE.base, padding: "0 6px", borderRadius: 5, background: MUTED, color: "#fff" }}>ERR</b>;
    if (v) return (
      <b style={{ flex: "none", fontSize: TYPE.base, padding: "0 6px", borderRadius: 5, background: v === "NG" ? DANGER : OK, color: "#fff" }}>{v}</b>
    );
    return <span className="muted" style={{ flex: "none", fontSize: TYPE.base, width: 26, textAlign: "center" }}>…</span>;
  };

  const viewerSrc = selected?.heatSrc
    ? (heatOn ? selected.heatSrc : (selected.origSrc ?? selected.url ?? selected.heatSrc))
    : null;

  return (
    <div
      className="operator-tab"
      style={{ position: "relative", display: "grid", gridTemplateColumns: "minmax(240px, 300px) minmax(0, 1fr) 320px", gap: 14, alignItems: "stretch", height: "min(76vh, 820px)" }}
      onDragEnter={(e) => { e.preventDefault(); dragDepth.current += 1; setDragging(true); }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => { dragDepth.current = Math.max(0, dragDepth.current - 1); if (dragDepth.current === 0) setDragging(false); }}
      onDrop={(e) => {
        e.preventDefault();
        dragDepth.current = 0; setDragging(false);
        if (e.dataTransfer.files.length) enqueue(e.dataTransfer.files);
      }}
    >
      {/* Whole-tab drop overlay: anywhere below the tab bar accepts files. */}
      {dragging && (
        <div style={{
          position: "absolute", inset: 0, zIndex: 60, pointerEvents: "none",
          border: `2px dashed ${ACCENT}`, borderRadius: 12,
          background: "rgba(34, 199, 219, 0.07)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <span style={{ fontSize: TYPE.hero, fontWeight: 700, color: ACCENT, textShadow: "0 1px 8px rgba(0,0,0,.25)" }}>
            {t("operator.dropOverlay")}
          </span>
        </div>
      )}

      {/* Column 1: inspection list — Develop-style card, ALWAYS rendered so
          the layout doesn't jump when the first image lands. */}
      <div style={{ border, borderRadius: 10, padding: 10, display: "flex", flexDirection: "column", gap: 8, minWidth: 0, minHeight: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <button
            data-tutorial-step="operator-drop"
            onClick={() => inputRef.current?.click()}
            style={{ padding: "3px 12px", fontSize: TYPE.base, borderRadius: 6, border, background: ACCENT, color: "#fff", cursor: "pointer", flex: "none" }}
          >+ {t("operator.addMore")}</button>
          {running && (
            <button
              onClick={requestCancel}
              disabled={cancelling}
              aria-busy={cancelling}
              style={{ padding: "2px 10px", fontSize: TYPE.base, borderRadius: 6, border, background: "transparent", color: INK, cursor: cancelling ? "default" : "pointer", opacity: cancelling ? 0.6 : 1, flex: "none" }}
            >{cancelling ? t("operator.cancelling") : t("operator.cancel")}</button>
          )}
          {items.length > 0 && (
            <button
              onClick={clearAll}
              style={{ marginLeft: "auto", padding: "2px 10px", fontSize: TYPE.base, borderRadius: 6, border, background: "transparent", color: INK, cursor: "pointer", flex: "none" }}
            >{t("operator.clear")}</button>
          )}
        </div>
        {(running || (thr != null && doneCount > 0)) && (
          <div className="muted" style={{ display: "flex", alignItems: "center", gap: 10, fontSize: TYPE.base, fontVariantNumeric: "tabular-nums", flexWrap: "wrap" }}>
            {running && (
              <span>{t("operator.progress").replace("{n}", String(doneCount + errCount)).replace("{total}", String(items.length))}</span>
            )}
            {thr != null && doneCount > 0 && (
              <span style={{ display: "inline-flex", gap: 10 }}>
                <span><b style={{ color: OK }}>OK</b> {okCount}</span>
                <span><b style={{ color: DANGER }}>NG</b> {ngCount}</span>
                {errCount > 0 && <span>{t("operator.errors").replace("{n}", String(errCount))}</span>}
              </span>
            )}
          </div>
        )}
        <div style={{ flex: 1, minHeight: 0, overflowY: "auto", display: "flex", flexDirection: "column", gap: 2 }}>
          {items.length === 0 && (
            <span className="muted" style={{ fontSize: TYPE.base, padding: "8px 4px" }}>{t("operator.listEmpty")}</span>
          )}
          {items.map((it) => {
            const isSel = it.id === selectedId;
            return (
              <button
                key={it.id}
                id={`insp-row-${it.id}`}
                onClick={() => setSelectedId(it.id)}
                title={it.name}
                style={{
                  display: "flex", alignItems: "center", gap: 8, padding: "4px 6px", borderRadius: 6,
                  border: "none", width: "100%", textAlign: "left", cursor: "pointer",
                  background: isSel ? PANEL_2 : "transparent",
                  color: INK,
                }}
              >
                {badge(it)}
                <span style={{ flex: 1, minWidth: 0, fontSize: TYPE.md, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{it.name}</span>
                <b style={{ flex: "none", fontSize: TYPE.md, fontVariantNumeric: "tabular-nums" }}>
                  {it.scores ? it.scores.topk.toFixed(2) : ""}
                </b>
                {it.ms != null && (
                  <span className="muted" style={{ flex: "none", fontSize: TYPE.base, fontVariantNumeric: "tabular-nums", minWidth: 42, textAlign: "right" }}>{it.ms}ms</span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Column 2: heatmap viewer — fixed pane, image fits inside (the raw
          heatmap is huge; scrolling a giant image was unusable). Zoom/pan
          grammar matches the Develop viewer. With no images yet, the pane IS
          the drop CTA. */}
      <div style={{ border, borderRadius: 10, padding: 10, display: "flex", flexDirection: "column", gap: 6, minWidth: 0, minHeight: 0 }}>
        {/* Which bank is passing judgement. Same stamp, same place as the
            check tab: an operator who sees OK here and NG there needs to be
            able to tell in one look whether the two ran on the same bank. */}
        <BankStamp projectId={projectId} active={active} border={border} />
        {items.length === 0 ? (
          <div
            className="anom-drop"
            onClick={() => inputRef.current?.click()}
            style={{
              flex: 1, minWidth: 0, minHeight: 0,
              border: `2px dashed ${BORDER}`, borderRadius: 8,
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10,
              cursor: "pointer", textAlign: "center", padding: 24,
            }}
          >
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.55 }} aria-hidden>
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            <div style={{ fontSize: TYPE.xl, fontWeight: 600 }}>{t("operator.dropTitle")}</div>
            <div className="muted" style={{ fontSize: TYPE.base }}>{t("operator.dropSub")}</div>
          </div>
        ) : (
          <>
          <ImageViewer
            zoom={zoom}
            style={{ minWidth: 0 }}
            overlay={selected?.heatSrc ? (
              <button
                onClick={() => setHeatOn((v) => !v)}
                aria-pressed={heatOn}
                title={`${t("operator.heatToggle")} (H)`}
                style={{
                  position: "absolute", top: 8, right: 8, padding: "3px 10px", fontSize: TYPE.base, borderRadius: 6,
                  border: "1px solid rgba(255,255,255,.3)", cursor: "pointer",
                  background: heatOn ? ACCENT : "rgba(0,0,0,.55)", color: "#fff",
                }}
              >{t("operator.heatToggle")} {heatOn ? "ON" : "OFF"} <span style={{ opacity: 0.7 }}>(H)</span></button>
            ) : null}
          >
            {viewerSrc && selected ? (
                /* Fills the pane rather than capping at its own pixel size --
                   same as the check tab's viewer. A 512px source in a 940px
                   pane used to sit in the middle of a field of black. */
                <img
                  src={viewerSrc}
                  draggable={false}
                  alt={heatOn ? "anomaly heatmap" : selected.name}
                  style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
                />
            ) : selected == null || selected.status === "scoring" || selected.status === "pending" ? (
              <span className="muted" style={{ fontSize: TYPE.md }}>{t("operator.scoring")}</span>
            ) : selected.status === "error" ? (
              <span className="muted" style={{ fontSize: TYPE.md, padding: 16, textAlign: "center" }}>{selected.err}</span>
            ) : (
              <span className="muted" style={{ fontSize: TYPE.md }}>—</span>
            )}
          </ImageViewer>
          {/* What the colours mean. The Teach viewer has always printed this;
              this one printed nothing, so it switched between an absolute and
              a per-image relative scale with no way to tell which you were
              looking at -- and the relative one paints the hottest 1% of any
              image red, defect-free parts included. */}
          {heatOn && (
            <div className="muted" style={{ fontSize: TYPE.sm, textAlign: "center", flex: "none" }}>
              {heatAnchors
                ? <span style={{ color: VERM }}>{t("develop.heatmap.absScale").replace("{thr}", heatAnchors.hi.toFixed(1))}</span>
                : t("operator.heatmap.relScale")}
            </div>
          )}
          {/* Same wording as the Develop viewer so the muscle memory carries. */}
          <div className="muted" style={{ fontSize: TYPE.sm, textAlign: "center", flex: "none" }}>{t("develop.viewer.hint")}</div>
          </>
        )}
      </div>
      <input ref={inputRef} type="file" accept="image/*" multiple hidden
        onChange={(e) => { if (e.target.files?.length) enqueue(e.target.files); e.target.value = ""; }} />

      <aside className="panel" data-tutorial-step="operator-save" style={{ padding: 16, border, borderRadius: 10, height: "fit-content", minWidth: 0 }}>
        <div style={{ fontSize: TYPE.verdict, fontWeight: 800, letterSpacing: 1, textAlign: "center",
          color: verdict === "NG" ? DANGER : verdict === "OK" ? OK : MUTED }}>
          {verdict ?? "—"}
        </div>
        <div className="muted" style={{ textAlign: "center", fontSize: TYPE.base, marginBottom: 14, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {selected ? selected.name : device ? `device ${device}` : ""}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: TYPE.base }}>
          {scores && (
            <>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <span>{t("operator.metricTopk")}</span>
                <b style={{ fontVariantNumeric: "tabular-nums", fontSize: TYPE.xxl }}>{scores.topk.toFixed(2)}</b>
              </div>
              {([["max", scores.max], ["p99", scores.p99]] as const).map(([k_, v]) => (
                <div key={k_} style={{ display: "flex", justifyContent: "space-between" }}>
                  <span className="muted">{k_}</span><b style={{ fontVariantNumeric: "tabular-nums" }}>{v.toFixed(2)}</b>
                </div>
              ))}
            </>
          )}

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, borderTop: border, paddingTop: 8 }}>
            <span className="muted" style={{ flex: "none" }}>{t("operator.threshold")}</span>
            <input
              type="number" step={0.1}
              value={thr != null ? Number(thr.toFixed(2)) : ""}
              placeholder="—"
              onChange={(e) => {
                thrEdited.current = true;
                setFromConfig(false);
                const v = Number(e.target.value);
                setThr(Number.isFinite(v) ? v : null);
              }}
              style={{ width: 90, textAlign: "right", fontVariantNumeric: "tabular-nums" }}
            />
          </div>
          {suggested != null && (
            <div className="muted" style={{ fontSize: TYPE.sm, textAlign: "right" }}>
              {t("operator.thresholdAuto")}: {suggested.toFixed(2)}
              {thrEdited.current && (
                <button
                  onClick={() => { thrEdited.current = false; setFromConfig(false); setThr(suggested); }}
                  style={{ marginLeft: 6, padding: "0 8px", fontSize: TYPE.sm, borderRadius: 6, border, background: "transparent", color: INK, cursor: "pointer" }}
                >↺</button>
              )}
            </div>
          )}
          {thr === null && (
            <div className="muted" style={{ fontSize: TYPE.sm }}>{t("operator.noThreshold")}</div>
          )}

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
            <span className="muted" style={{ flex: "none" }} title={t("develop.eval.alphaHint")}>{t("develop.eval.alpha")}</span>
            <input
              type="range" min={0} max={ALPHA_MAX} step={5} value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))}
              aria-label={t("develop.eval.alpha")}
              style={{ flex: 1, minWidth: 0 }}
            />
            <b style={{ fontVariantNumeric: "tabular-nums", minWidth: 34, textAlign: "right" }}>{alpha}</b>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
            <span className="muted" style={{ flex: "none" }} title={t("develop.eval.betaHint")}>{t("develop.eval.beta")}</span>
            <input
              type="range" min={0} max={ALPHA_MAX} step={5} value={beta}
              onChange={(e) => setBeta(Number(e.target.value))}
              aria-label={t("develop.eval.beta")}
              style={{ flex: 1, minWidth: 0 }}
            />
            <b style={{ fontVariantNumeric: "tabular-nums", minWidth: 34, textAlign: "right" }}>{beta}</b>
          </div>
          {beta > 0 && (
            <div className="muted" style={{ fontSize: TYPE.sm, textAlign: "right" }}>
              {t("develop.eval.betaNotInCheck")}
            </div>
          )}
          {scores && alpha > 0 && (
            <div className="muted" style={{ fontSize: TYPE.sm, textAlign: "right" }}>
              {t("operator.exemplarRows")}: {scores.nExemplar.toLocaleString()}
            </div>
          )}

          {cfgStale && (
            <div style={{ fontSize: TYPE.sm, color: DANGER }}>
              {t("operator.configStale")}
            </div>
          )}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, borderTop: border, paddingTop: 8 }}>
            <span className="muted" style={{ fontSize: TYPE.sm, minWidth: 0 }}>
              {fromConfig ? `✓ ${t("operator.configActive")}` : ""}
            </span>
            <button
              onClick={() => void saveConfig()}
              style={{ flex: "none", padding: "4px 12px", fontSize: TYPE.base, borderRadius: 6, border, background: "transparent", color: INK, cursor: "pointer", whiteSpace: "nowrap" }}
            >{t("operator.saveConfig")}</button>
          </div>

          {scores?.serverMs != null && (
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8, borderTop: border, paddingTop: 8 }}>
              <span className="muted">{t("operator.timing")}</span>
              <span style={{ fontVariantNumeric: "tabular-nums" }}>
                <b>{Math.round(scores.serverMs)} ms</b>
                {selected?.ms != null && (
                  <span className="muted" style={{ fontSize: TYPE.sm }}> ({t("operator.roundtrip")} {selected.ms} ms)</span>
                )}
              </span>
            </div>
          )}
        </div>
        <p className="muted" style={{ fontSize: TYPE.sm, marginTop: 14 }}>{t("operator.verdictNote")}</p>
      </aside>
    </div>
  );
}

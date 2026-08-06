// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// Inline icons in this file use SVG path data from Feather Icons
// (https://github.com/feathericons/feather), MIT licensed,
// Copyright (c) 2013-2023 Cole Bemis. Some paths are adapted.
//
// Develop (学習) tab — check the bank the bank tab built: score every taught
// image from its stored features, read how far the OK and NG distributions
// separate, and settle on a threshold the inspection tab will run with.
//
// It does not teach anymore. Dropping files here staged them into a second
// writer that wrote to the bank directly, while the bank tab assembles the
// bank from the STORE — so assemble rebuilt the bank without them and no
// warning fired, because the staleness fingerprint only watches the store and
// the label set. One place puts images in: the bank tab.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useI18n, formatError } from "../i18n";
import {
  evaluateBankImage,
  fetchBank,
  fetchBankImages,
  imageUrl,
  scoreImage,
  selectBank,
  type BankImage,
  type BankState,
  type ScoreResult,
  type SelectResult,
  type StoredImageEval,
  type Tier,
} from "../api/cls";
import { isAbortError } from "../api/shared";
import BankStamp from "./BankStamp";
import Glyph from "./Glyph";
import ImageViewer from "../ui/ImageViewer";
import { useZoomPan } from "../ui/useZoomPan";
import { ACCENT, ACCENT_SOFT, BORDER, INK, RULE, TYPE, VERM } from "../ui/tokens";
import BankProjection from "./develop/FeatureMap";
import BusyDialog from "./develop/BusyDialog";
import ImageInfo from "./develop/ImageInfo";
import { VerdictAnchors } from "./develop/types";

type Props = {
  projectId: string | null;
  active: boolean;
  showToast: (msg: string) => void;
};

// Column widths. The gutters are the drag handles, so they are part of the
// arithmetic: the centre never gets squeezed below MIN_VIEWER no matter which
// side is being dragged.
const SPLIT_W = 14;
const MIN_LIST = 200;
const MIN_RAIL = 260;
const MIN_VIEWER = 320;
const DEF_LIST = 300;
const DEF_RAIL = 340;

/** A gutter you can drag. It IS the gap between two columns rather than a bar
 *  inside it -- a 4px hit target between two panels is a thing you hunt for,
 *  and the 14px of air was already there. Arrow keys move it too: pointer-only
 *  resize is unusable without a mouse. */
function Splitter({ label, onDrag }: { label: string; onDrag: (dx: number) => void }) {
  const last = useRef(0);
  const [live, setLive] = useState<"idle" | "hover" | "drag">("idle");
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      title={label}
      tabIndex={0}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        last.current = e.clientX;
        setLive("drag");
      }}
      onPointerMove={(e) => {
        if (live !== "drag") return;
        const dx = e.clientX - last.current;
        if (dx) { last.current = e.clientX; onDrag(dx); }
      }}
      onPointerUp={(e) => { e.currentTarget.releasePointerCapture(e.pointerId); setLive("idle"); }}
      onPointerCancel={() => setLive("idle")}
      onPointerEnter={() => setLive((v) => (v === "drag" ? v : "hover"))}
      onPointerLeave={() => setLive((v) => (v === "drag" ? v : "idle"))}
      onKeyDown={(e) => {
        const step = e.shiftKey ? 48 : 16;
        if (e.key === "ArrowLeft") onDrag(-step);
        else if (e.key === "ArrowRight") onDrag(step);
        else return;
        e.preventDefault();
      }}
      style={{
        width: SPLIT_W, flex: "none", alignSelf: "stretch", cursor: "col-resize",
        display: "flex", alignItems: "center", justifyContent: "center",
        touchAction: "none",
      }}
    >
      <span style={{
        width: 2, height: live === "idle" ? 24 : "60%", borderRadius: 2,
        background: live === "drag" ? ACCENT : BORDER,
        transition: "height .12s",
      }} />
    </div>
  );
}

// Stable key for a taught bank image, used for list selection and the
// active-row lookup.
const taughtKey = (im: BankImage) => `T:${im.tier}/${im.label}/${im.name}`;

// One row of the image list. Every row is a taught bank image: the staged
// (dropped-but-not-taught) rows that used to sit above them belonged to a
// writer this tab no longer has.
type ListRow = { key: string; tier: Tier; name: string; im: BankImage };

const TIER_RANK: Record<Tier, number> = { normal: 0, negative: 1, critical: 2 };

type SortMode = "tier" | "name-asc" | "name-desc";

export default function Develop({ projectId, active, showToast }: Props) {
  const { t, lang } = useI18n();
  // Column widths, dragged and remembered. Different banks want different
  // splits -- a project of long filenames wants the list wide, a 4:3 sensor
  // wants the middle -- and re-dragging it every session is the kind of small
  // tax people stop noticing they are paying.
  const tabRef = useRef<HTMLDivElement | null>(null);
  const [cols, setCols] = useState(() => {
    const read = (k: string, d: number, min: number) => {
      const v = Number(localStorage.getItem(k));
      return Number.isFinite(v) && v >= min ? v : d;
    };
    return { list: read("anom.listW", DEF_LIST, MIN_LIST), rail: read("anom.railW", DEF_RAIL, MIN_RAIL) };
  });
  useEffect(() => {
    localStorage.setItem("anom.listW", String(cols.list));
    localStorage.setItem("anom.railW", String(cols.rail));
  }, [cols]);
  // Clamped against the OTHER column's current width, so dragging one side
  // wide cannot push the image out of the middle.
  const resize = useCallback((which: "list" | "rail", dx: number) => {
    setCols((c) => {
      const total = tabRef.current?.clientWidth ?? 0;
      const min = which === "list" ? MIN_LIST : MIN_RAIL;
      const other = which === "list" ? c.rail : c.list;
      const room = total ? total - other - 2 * SPLIT_W - MIN_VIEWER : Infinity;
      const next = Math.round(Math.min(Math.max(min, room), Math.max(min, (which === "list" ? c.list + dx : c.rail - dx))));
      return which === "list" ? { ...c, list: next } : { ...c, rail: next };
    });
  }, []);
  const [bank, setBank] = useState<BankState | null>(null);
  const [activeBankId, setActiveBankId] = useState<string>("");
  const [images, setImages] = useState<BankImage[]>([]);
  // The image shown in the right-hand viewer.
  const [preview, setPreview] = useState<BankImage | null>(null);

  const applySelect = useCallback((r: SelectResult) => {
    setBank(r.bank);
    setActiveBankId(r.bank_id);
  }, []);

  // Everything that points INTO a specific bank's contents — the viewer, the
  // list selection — must be dropped whenever the project or bank changes.
  // Leaving it renders the PREVIOUS project's image in the viewer of the new
  // one (2026-07-18 report; the classic stale-view bug class). Deliberately
  // NOT part of applySelect: the silent re-select on tab activation must keep
  // the view.
  const resetBankViewState = useCallback(() => {
    setPreview(null);
  }, []);

  const refreshImages = useCallback(() => {
    // On error keep the last known list: blanking it turns a transient
    // fetch failure (or a 409 from a raced project delete) into a fake
    // "empty bank" display that is indistinguishable from real emptiness.
    fetchBankImages().then((r) => setImages(r.images)).catch(() => { /* keep last */ });
  }, []);

  // Last successfully selected project — lets re-activation of the tab keep
  // the current view, while an actual project switch drops the previous
  // bank's state immediately (a big bank can take seconds to load server-side
  // and everything shown meanwhile would belong to the old project).
  const selectedProjectRef = useRef<string | null>(null);
  // Server-side bank activation reads the whole bank into RAM — tens of
  // seconds for a multi-GB bank — and the tab meanwhile shows nothing.
  // Surface that wait as an explicit blocking dialog (silent blank = "did it
  // crash?"). Only a real project/bank change raises it; the silent
  // re-select on tab activation keeps the current view without flashing.
  const [bankLoading, setBankLoading] = useState(false);
  useEffect(() => {
    if (!active || !projectId) return;
    let cancelled = false;
    if (selectedProjectRef.current !== projectId) {
      setBank(null); setActiveBankId(""); setImages([]);
      resetBankViewState();
      setBankLoading(true);
    }
    selectBank(projectId)
      .then((r) => { if (!cancelled) { selectedProjectRef.current = projectId; applySelect(r); refreshImages(); } })
      .catch((e) => showToast(`${t("develop.bankSelectFailed")}: ${formatError(e, lang)}`))
      .finally(() => { if (!cancelled) setBankLoading(false); });
    return () => { cancelled = true; };
  }, [active, projectId, applySelect, refreshImages, showToast, t, lang]);

  // Pre-render every taught image's heatmap into the client cache so the
  // preview's H toggle is instant. Sequential, cancellable, skips cached.
  const [heatPrefetch, setHeatPrefetch] = useState<{ done: number; total: number } | null>(null);
  const heatPrefetchCancel = useRef(false);
  const heatPrefetchAbort = useRef<AbortController | null>(null);
  const [heatPrefetchCancelling, setHeatPrefetchCancelling] = useState(false);
  const requestHeatPrefetchCancel = useCallback(() => {
    heatPrefetchCancel.current = true;
    setHeatPrefetchCancelling(true);
    heatPrefetchAbort.current?.abort();
  }, []);
  const heatAnchorsRef = useRef<VerdictAnchors | null>(null);
  const prefetchHeatmaps = useCallback(async () => {
    const bankAtStart = activeBankId;
    let list: BankImage[] = [];
    try { list = (await fetchBankImages()).images; } catch { return; }
    if (!list.length) return;
    heatPrefetchCancel.current = false;
    setHeatPrefetchCancelling(false);
    setHeatPrefetch({ done: 0, total: list.length });
    try {
      for (let i = 0; i < list.length; i++) {
        if (heatPrefetchCancel.current) break;
        const im = list[i];
        const key = `${im.tier}/${im.label}/${im.name}`;
        if (!heatCache.current.has(key)) {
          const ac = new AbortController();
          heatPrefetchAbort.current = ac;
          try {
            const res = await fetch(`${imageUrl(im.url)}?b=${bankAtStart}`, { signal: ac.signal });
            if (!res.ok) throw new Error(String(res.status));
            const blob = await res.blob();
            const file = new File([blob], im.name, { type: blob.type || "image/jpeg" });
            // Anchors are re-read per image: the sweep that precedes the
            // prefetch lifts fresh (debounced) anchors while this loop is
            // already running.
            const anchors = heatAnchorsRef.current;
            const r = await scoreImage(file, { k: 5, alpha: anchors?.alpha ?? 0, hmLo: anchors?.lo, hmHi: anchors?.hi, recordHits: false, signal: ac.signal });
            heatCache.current.set(key, r);
          } catch (e) {
            // An abort means the user cancelled: stop the loop rather than
            // treating it as one skippable image and scoring the next N.
            if (isAbortError(e) || heatPrefetchCancel.current) break;
            // 409 = active bank/project changed — abort the whole pre-render
            // instead of silently firing N more failing requests.
            if ((e as { status?: number }).status === 409) {
              heatPrefetchCancel.current = true;
            }
            /* otherwise skip this image */
          } finally {
            heatPrefetchAbort.current = null;
          }
        }
        setHeatPrefetch({ done: i + 1, total: list.length });
      }
    } finally {
      heatPrefetchAbort.current = null;
      setHeatPrefetch(null);
      setHeatPrefetchCancelling(false);
    }
  }, [activeBankId]);

  // 評価を実行: sweep the separation check (which re-projects the map), then
  // pre-render heatmaps in the background. The sweep resolves to `true` when
  // the user cancelled it, so the chain stops instead of opening the next
  // near-identical blocking modal a moment later — which is indistinguishable
  // from the button having done nothing at all (2026-07-31).
  const evalRunRef = useRef<(() => Promise<boolean>) | null>(null);
  const [orchestrating, setOrchestrating] = useState(false);
  const runAll = useCallback(async () => {
    if (orchestrating) return;
    setOrchestrating(true);
    try {
      if (await evalRunRef.current?.()) return;
      void prefetchHeatmaps();
    } finally { setOrchestrating(false); }
  }, [orchestrating, prefetchHeatmaps]);

  const openPreview = useCallback((im: BankImage) => { setPreview(im); }, []);

  // Heatmap overlay inside the preview dialog, toggled with "H". Scores are
  // cached per image so re-toggling and ←/→ navigation don't re-score.
  const [heatOn, setHeatOn] = useState(false);
  const [heatResult, setHeatResult] = useState<ScoreResult | null>(null);
  const [heatLoading, setHeatLoading] = useState(false);
  const heatCache = useRef<Map<string, ScoreResult>>(new Map());
  // Verdict-scale heatmap anchors from the separation check (OK median →
  // operative threshold, α-composite), reported up by BankProjection. With
  // anchors the server renders the anomaly-focus overlay — colour only where
  // the score approaches the verdict level — instead of per-image
  // percentiles that paint something red on every image.
  const [heatAnchors, setHeatAnchors] = useState<VerdictAnchors | null>(null);
  const onRawAnchors = useCallback((a: VerdictAnchors | null) => {
    heatAnchorsRef.current = a;
    setHeatAnchors(a);
  }, []);

  // Any bank mutation (teach/delete/switch) — or an anchor/α change after an
  // eval run — invalidates cached heatmaps.
  const heatFingerprint = `${activeBankId}/${bank ? `${bank.normal}/${bank.critical}/${bank.negative}` : ""}/${heatAnchors ? `${heatAnchors.lo.toFixed(4)}-${heatAnchors.hi.toFixed(4)}-a${heatAnchors.alpha}` : "rel"}`;
  useEffect(() => { heatCache.current.clear(); setHeatResult(null); }, [heatFingerprint]);

  useEffect(() => {
    if (!preview || !heatOn) { setHeatResult(null); return; }
    const key = `${preview.tier}/${preview.label}/${preview.name}`;
    const hit = heatCache.current.get(key);
    if (hit) { setHeatResult(hit); return; }
    let cancelled = false;
    setHeatLoading(true); setHeatResult(null);
    (async () => {
      try {
        const res = await fetch(`${imageUrl(preview.url)}?b=${activeBankId}`);
        if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
        const blob = await res.blob();
        const file = new File([blob], preview.name, { type: blob.type || "image/jpeg" });
        const r = await scoreImage(file, { k: 5, alpha: heatAnchors?.alpha ?? 0, hmLo: heatAnchors?.lo, hmHi: heatAnchors?.hi, recordHits: false });
        if (!cancelled) { heatCache.current.set(key, r); setHeatResult(r); }
      } catch (e) {
        if (!cancelled) showToast(`heatmap failed: ${(e as Error).message}`);
      } finally {
        if (!cancelled) setHeatLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [preview, heatOn, activeBankId, heatAnchors, showToast]);

  // A stable identity for the previewed image: effects key off this rather
  // than the object, whose identity changes on every refresh even when the
  // image has not.
  const previewKey = preview ? `${preview.tier}/${preview.label}/${preview.name}` : "";

  // Cross-client sync (read-only): the server holds ONE global active bank
  // shared by every LAN client, and this tab otherwise only re-fetches on its
  // own actions — so a second PC would never see the editor's uploads. Poll
  // bank counts + images on window focus and a slow interval; the projection
  // map (keyed on the counts) and the separation check (keyed on the images
  // array) both re-fetch off these, so the whole tab follows. Deliberately
  // does NOT call selectBank: that would mutate the global active bank and
  // yank it away from whoever is editing. Skipped while this client is mid-
  // operation so a poll can't land on top of a local eval.
  const syncFromServer = useCallback(() => {
    if (!active || !projectId || !activeBankId) return;
    if (orchestrating) return;
    void fetchBank().then((b) => setBank(b)).catch((e) => {
      // 409 = the server lost its active project (restart, or a delete
      // re-bound it). Re-select this tab's project once instead of
      // polling 409s forever behind a blank-looking UI (2026-07-18: an
      // open tab kept its dead session across an API restart and every
      // poll failed until a manual hard reload).
      if ((e as { status?: number }).status === 409) {
        selectBank(projectId)
          .then((r) => { applySelect(r); refreshImages(); })
          .catch(() => { /* project gone for real — the toast on the next
                            user action explains it */ });
      }
    });
    refreshImages();
  }, [active, projectId, activeBankId, orchestrating, refreshImages, applySelect]);

  useEffect(() => {
    if (!active) return;
    const onFocus = () => syncFromServer();
    window.addEventListener("focus", onFocus);
    const id = window.setInterval(() => {
      if (document.visibilityState === "visible") syncFromServer();
    }, 8000);
    return () => { window.removeEventListener("focus", onFocus); window.clearInterval(id); };
  }, [active, syncFromServer]);

  // Wheel = zoom around the cursor, left-drag = pan while zoomed,
  // space+drag / middle-drag = pan, double-click = fit. Shared with inspect.
  const zoom = useZoomPan(previewKey, active);

  // Per-image stored-feature eval for the info panel. Server-cached after a
  // sweep, so this is a fast lookup; an uncached image costs one small cdist.
  const [previewEval, setPreviewEval] = useState<StoredImageEval | null>(null);
  useEffect(() => {
    if (!preview) { setPreviewEval(null); return; }
    let cancelled = false;
    setPreviewEval(null);
    evaluateBankImage(preview.tier, preview.name, preview.label)
      .then((r) => { if (!cancelled) setPreviewEval(r); })
      .catch(() => { if (!cancelled) setPreviewEval(null); });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewKey]);

  // Global H = heatmap toggle while a taught image is in the viewer. List
  // navigation (arrows / Delete / Esc) is handled on the list container so it
  // doesn't fire while typing elsewhere.
  useEffect(() => {
    if (!preview) return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if ((e.key === "h" || e.key === "H") && !e.metaKey && !e.ctrlKey) { e.preventDefault(); setHeatOn((v) => !v); }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [preview]);

  // ---- The taught images, as a list -------------------------------------
  const [sortMode, setSortMode] = useState<SortMode>(() => {
    const v = localStorage.getItem("anom.listSort");
    return v === "name-asc" || v === "name-desc" || v === "tier" ? v : "tier";
  });
  useEffect(() => { localStorage.setItem("anom.listSort", sortMode); }, [sortMode]);
  const rows = useMemo<ListRow[]>(() => {
    const list: ListRow[] = images.map((im) => ({ key: taughtKey(im), tier: im.tier, name: im.name, im }));
    // Natural order so OK_2 sorts before OK_10 (plain locale compare doesn't).
    const byName = (a: ListRow, b: ListRow) => a.name.localeCompare(b.name, undefined, { numeric: true });
    return list.sort(
      sortMode === "name-asc" ? byName
      : sortMode === "name-desc" ? (a, b) => byName(b, a)
      : (a, b) => TIER_RANK[a.tier] - TIER_RANK[b.tier] || byName(a, b),
    );
  }, [images, sortMode]);
  const rowsRef = useRef(rows);
  rowsRef.current = rows;

  const activeKey = preview ? taughtKey(preview) : null;
  useEffect(() => { activeKeyRef.current = activeKey; }, [activeKey]);
  const activeIdx = activeKey ? rows.findIndex((r) => r.key === activeKey) : -1;

  const activateRow = useCallback((row: ListRow) => { setPreview(row.im); }, []);

  // Ref mirror of activeKey, updated synchronously in moveActive so a rapid
  // burst of arrow keys steps once per keystroke instead of skipping (the
  // classic pending-arrow-index problem).
  const activeKeyRef = useRef<string | null>(null);
  const listBoxRef = useRef<HTMLDivElement | null>(null);

  // Move the active row through the unified list. Global ↑/↓, the list's own
  // keydown and the ‹ › header buttons all funnel here; shift extends the
  // selection from the anchor (standard list behaviour).
  const moveActive = useCallback((dir: -1 | 1) => {
    const list = rowsRef.current;
    if (!list.length) return;
    const key = activeKeyRef.current;
    const idx = key ? list.findIndex((r) => r.key === key) : -1;
    const nextIdx = idx < 0 ? 0 : Math.max(0, Math.min(list.length - 1, idx + dir));
    if (nextIdx === idx) return;
    const next = list[nextIdx]!;
    activeKeyRef.current = next.key;
    activateRow(next);
    // Keep the active row visible in the scrolling list.
    listBoxRef.current?.querySelector(`[data-rowkey="${CSS.escape(next.key)}"]`)?.scrollIntoView({ block: "nearest" });
  }, [activateRow]);

  // Global ↑/↓ = previous/next image, annotation-tool style:
  // skipped while typing, and skipped when the focused listbox already
  // handled the key (defaultPrevented guard avoids a double step).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.defaultPrevented) return;
      e.preventDefault();
      moveActive(e.key === "ArrowUp" ? -1 : 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [moveActive]);

  const onListKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      moveActive(e.key === "ArrowUp" ? -1 : 1);
    }
  }, [moveActive]);

  if (!projectId) return <div className="muted" style={{ padding: 32 }}>{t("develop.selectFirst")}</div>;

  const border = RULE;

  return (
    <div ref={tabRef} className="develop-tab" style={{ display: "flex", height: "100%", minHeight: 0 }}>
      {/* Blocking progress dialog while the server activates a bank. Fixed
          overlay escapes the hidden tab-panel, so gate on `active` — bank
          select auto-fires on tab activation and must not cover other tabs. */}
      {active && bankLoading && (
        <BusyDialog border={border} title={t("develop.bankLoading")} hint={t("develop.bankLoadingHint")} />
      )}
      {/* The list runs the whole height of the tab, like the bank tab's.
          It used to start below the card row, which cost it half its length
          for no reason -- and made the two charts share the full width with
          the bank card instead of having the right side to themselves. */}
      <div style={{ width: cols.list, flex: "none", border, borderRadius: 10, padding: 10, display: "flex", flexDirection: "column", gap: 8, minWidth: 0, minHeight: 0 }}>
          <div className="muted" style={{ fontSize: TYPE.sm, fontVariantNumeric: "tabular-nums", display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ flex: 1, minWidth: 0 }}>
              {t("develop.list.count").replace("{n}", String(rows.length))}
            </span>
            <span aria-hidden style={{ flex: "none" }}>{t("develop.list.sort")}</span>
            <select
              value={sortMode}
              onChange={(e) => setSortMode(e.target.value as SortMode)}
              aria-label={t("develop.list.sort")}
              style={{ flex: "none", fontSize: TYPE.sm, padding: "1px 4px", borderRadius: 4 }}
            >
              <option value="tier">{t("develop.list.sort.tier")}</option>
              <option value="name-asc">{t("develop.list.sort.nameAsc")}</option>
              <option value="name-desc">{t("develop.list.sort.nameDesc")}</option>
            </select>
          </div>
          {/* Read-only: it used to be a whole-area dropzone, and a drop here
              staged images into a writer whose work assemble discarded. */}
          <div
            ref={listBoxRef}
            tabIndex={0}
            role="listbox"
            aria-label={t("develop.list.label")}
            onKeyDown={onListKeyDown}
            style={{ flex: 1, minHeight: 0, overflowY: "auto", borderRadius: 8, border, outline: "none" }}
          >
            {rows.length === 0 && (
              <div className="muted" style={{ padding: 20, textAlign: "center", fontSize: TYPE.base }}>{t("develop.list.noImages")}</div>
            )}
            {rows.map((row) => {
              const isActive = activeKey === row.key;
              const tierName = t(`develop.tier.${row.tier}` as "develop.tier.normal");
              return (
                <div
                  key={row.key}
                  data-rowkey={row.key}
                  role="option"
                  aria-selected={isActive}
                  aria-label={`${row.name} · ${tierName}`}
                  title={`${row.name}\n${row.im.patches} p`}
                  onClick={() => activateRow(row)}
                  style={{
                    display: "flex", alignItems: "center", gap: 9, padding: "8px 10px",
                    fontSize: TYPE.md, cursor: "pointer", userSelect: "none",
                    borderLeft: isActive ? `4px solid ${ACCENT}` : "4px solid transparent",
                    background: isActive ? ACCENT_SOFT : "transparent",
                  }}
                >
                  <Glyph tier={row.tier} size={15} />
                  <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{row.name}</span>
                  <span className="muted" style={{ fontSize: TYPE.base, flex: "none", fontVariantNumeric: "tabular-nums" }}>{row.im.patches}p</span>
                </div>
              );
            })}
          </div>
        </div>

      <Splitter label={t("develop.resizeList")} onDrag={(dx) => resize("list", dx)} />

      {/* Centre: the image, floor to ceiling. The check used to sit above it
          in a 264px band, and that was 264px the image never got -- it reads
          just as well down the right rail, beside the numbers for the very
          image it is judging. */}
        <div data-tutorial-step="develop-viewer" style={{ border, borderRadius: 10, padding: 12, display: "flex", flexDirection: "column", gap: 12, minWidth: 0, minHeight: 0, flex: 1 }}>
          {/* What the scores below were produced BY. Over the image rather
              than in the rail: the rail is what the numbers mean, this is
              which bank produced them. */}
          <BankStamp projectId={projectId} active={active} border={border} />
          {preview ? (
            <>
              <ImageViewer
                zoom={zoom}
                title={t("develop.viewer.hint")}
                overlay={<>
                  {heatOn && heatLoading && (
                    <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: TYPE.md, color: "#fff", textShadow: "0 1px 4px rgba(0,0,0,.8)" }}>{t("develop.heatmap.loading")}</div>
                  )}
                  {/* Name and controls ride ON the image. As a row above it
                      and a hint row below it they cost the viewer ~50px of
                      height for two lines of chrome. */}
                  <span style={{ position: "absolute", top: 6, left: 8, right: 8, display: "flex", alignItems: "center", gap: 8, pointerEvents: "none", zIndex: 2 }}>
                    <span style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: TYPE.base, padding: "2px 8px", borderRadius: 4, background: "rgba(0,0,0,.55)", color: "#fff" }}>
                      {preview.name}
                      <span style={{ opacity: .75 }}> · {t(`develop.tier.${preview.tier}` as "develop.tier.normal")}</span>
                      {rows.length > 1 && activeIdx >= 0 && <span style={{ opacity: .75 }}> · {activeIdx + 1}/{rows.length}</span>}
                    </span>
                    <span style={{ marginLeft: "auto", display: "flex", gap: 4, flex: "none", pointerEvents: "auto" }}>
                      <button
                        onClick={() => setHeatOn((v) => !v)}
                        aria-pressed={heatOn}
                        data-tutorial-step="develop-heatmap-toggle"
                        title={t("develop.preview.heatmap")}
                        style={{ padding: "0 10px", height: 24, fontSize: TYPE.base, borderRadius: 5, border: "none", cursor: "pointer", background: heatOn ? ACCENT : "rgba(0,0,0,.55)", color: "#fff" }}
                      >{t("develop.preview.heatmap")}</button>
                      {rows.length > 1 && (
                        <>
                          <button onClick={() => moveActive(-1)} disabled={activeIdx <= 0} aria-label={t("develop.prev")} title={t("develop.prev")}
                            style={{ width: 24, height: 24, padding: 0, borderRadius: 5, background: "rgba(0,0,0,.55)", color: "#fff", border: "none", cursor: activeIdx <= 0 ? "not-allowed" : "pointer", fontSize: TYPE.lg, opacity: activeIdx <= 0 ? .4 : 1 }}>↑</button>
                          <button onClick={() => moveActive(1)} disabled={activeIdx >= rows.length - 1} aria-label={t("develop.next")} title={t("develop.next")}
                            style={{ width: 24, height: 24, padding: 0, borderRadius: 5, background: "rgba(0,0,0,.55)", color: "#fff", border: "none", cursor: activeIdx >= rows.length - 1 ? "not-allowed" : "pointer", fontSize: TYPE.lg, opacity: activeIdx >= rows.length - 1 ? .4 : 1 }}>↓</button>
                        </>
                      )}
                    </span>
                  </span>
                </>}
              >
                {/* Fills the pane rather than capping at its own pixel
                    size: a 512px source in a 926px pane used to sit in the
                    middle of a field of black. `contain` keeps the aspect,
                    so upscaling only ever costs sharpness, never shape. */}
                <img
                  draggable={false}
                  src={heatOn && heatResult ? `data:image/png;base64,${heatResult.heatmap_png_base64}` : `${imageUrl(preview.url)}?b=${activeBankId}`}
                  alt={preview.name}
                  style={{ width: "100%", height: "100%", objectFit: "contain", display: "block", opacity: heatOn && heatLoading ? .5 : 1 }}
                />
              </ImageViewer>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <span className="muted" style={{ fontSize: TYPE.base, fontVariantNumeric: "tabular-nums", minWidth: 0 }}>
                  {heatOn && heatResult && (
                    <>
                      max <b style={{ color: INK }}>{heatResult.max_score.toFixed(2)}</b>
                      {" · "}p99 <b style={{ color: INK }}>{heatResult.p99_score.toFixed(2)}</b>
                      {" · "}mean <b style={{ color: INK }}>{heatResult.mean_score.toFixed(2)}</b>
                      {" · "}
                      {heatAnchors
                        ? <span style={{ color: VERM }}>{t("develop.heatmap.absScale").replace("{thr}", heatAnchors.hi.toFixed(1))}</span>
                        : t("develop.heatmap.relScale")}
                    </>
                  )}
                </span>
              </div>
            </>
          ) : (
            <div className="muted" style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", fontSize: TYPE.md, textAlign: "center", padding: 24 }}>
              {t("develop.list.empty")}
            </div>
          )}
        </div>

      <Splitter label={t("develop.resizeRail")} onDrag={(dx) => resize("rail", dx)} />

      {/* Right rail, in the order it is read: the figure, the knobs that
          shape it, then the numbers for whichever image is in the viewer.
          Only the numbers scroll: the figure is the reason the rail is there,
          so it must not be the thing that goes off the bottom. */}
      <div style={{ width: cols.rail, flex: "none", minWidth: 0, minHeight: 0, display: "flex", flexDirection: "column", gap: 10 }}>
        <BankProjection
          activeBankId={activeBankId}
          bank={bank}
          images={images}
          onOpenImage={openPreview}
          onRawAnchors={onRawAnchors}
          onRunAll={() => void runAll()}
          runAllDisabled={orchestrating || images.length === 0}
          evalRunRef={evalRunRef}
          border={border}
          showToast={showToast}
        />
        {preview && (
          <ImageInfo preview={preview} previewEval={previewEval} heatAnchors={heatAnchors} border={border} />
        )}
        {/* Eats the slack so the panel above it keeps its own height. */}
        <span style={{ flex: 1, minHeight: 0 }} />
      </div>

      {/* Heatmap pre-render — blocking modal, cancellable. Deliberately NOT
          gated on `active`: teaching or scoring from another tab while the
          pre-render hammers the GPU would queue conflicting work, so the
          whole app waits (user request: no side work during heatmap build). */}
      {heatPrefetch && (
        <BusyDialog
          border={border}
          title={t("develop.heatmap.prefetch").replace("{n}", String(heatPrefetch.done)).replace("{total}", String(heatPrefetch.total))}
          hint={t("develop.busyHint")}
          onCancel={requestHeatPrefetchCancel}
          cancelLabel={heatPrefetchCancelling ? t("develop.eval.cancelling") : t("develop.eval.cancel")}
          cancelDisabled={heatPrefetchCancelling}
        />
      )}

    </div>
  );
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// The bank: import images, say what they are, mark the defects, assemble.
//
// Importing and labelling were briefly two tabs, mirroring the two halves of
// the architecture. They showed the same images twice, so the split bought
// nothing. What had to be separated was extraction from tier assignment in the
// DATA (see clscore.store) -- once that holds, the import button can sit next
// to the picker without dragging a re-extraction behind every relabel.
//
// A contact sheet of thumbnails came before this and was too heavy: the store
// keeps ORIGINAL images, so a 500-image project pulled hundreds of megabytes to
// fill 92px tiles. This is a list plus one large preview, which is also the
// only size at which a defect can actually be judged.
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  assembleBank,
  assignImages,
  bankExportUrl,
  createLabelSet,
  deleteFromStore,
  deleteLabelSet,
  fetchAssemblyStatus,
  fetchLabelSets,
  fetchStore,
  importBank,
  ingestImages,
  ingestZip,
  markStoreImage,
  migrateStore,
  selectBank,
  selectLabelSet,
  setStoreGroup,
  storeImageUrl,
  unassignImages,
  type AnnotationRect,
  type AssemblyStatus,
  type LabelSetInfo,
  type StoreImageInfo,
  type StoreListResponse,
  type Tier,
} from "../api/cls";
import BankImportDialog from "./BankImportDialog";
import Glyph from "./Glyph";
import BusyDialog from "./develop/BusyDialog";
import { ACCENT, ACCENT_SOFT, BORDER, INK, MUTED, PANEL, RULE, TYPE, VERM } from "../ui/tokens";
import { useI18n } from "../i18n";

type Props = {
  projectId: string | null;
  active: boolean;
  showToast: (msg: string) => void;
};

type Filter = "all" | "unlabeled" | Tier;

// Images per ingest request. The server batches the forward across the whole
// request, so larger is faster -- but progress only advances per request, and
// a drop of 500 with no visible movement reads as a hang.
const INGEST_CHUNK = 24;

const MIN_ZOOM = 1;
const MAX_ZOOM = 12;
// Below this the drag was a click, not a rectangle. Without it every stray
// click while in mark mode adds a degenerate mark.
const MIN_RECT = 0.004;

const border = RULE;
// Okabe-Ito green: "this one is behind you", and the same green the check
// tab's run button uses for "go". Never the only carrier -- the disc holds a
// tick instead of a digit, and the word 完了 sits beside it.
const DONE = "#009E73";
const DONE_EDGE = "rgba(0,158,115,.45)";

function fmt(n: number): string {
  return n.toLocaleString();
}

/** Largest w x h with ``ratio`` that fits in the box. Upscales: a 640px source
 *  in a 900px pane should fill it, because the point of the pane is to show the
 *  defect at a size a person can judge. */
function fitted(box: { w: number; h: number }, ratio: number): { w: number; h: number } {
  if (!box.w || !box.h || !ratio) return { w: 0, h: 0 };
  const byWidth = { w: box.w, h: box.w / ratio };
  return byWidth.h <= box.h ? byWidth : { w: box.h * ratio, h: box.h };
}

export default function BankTab({ projectId, active, showToast }: Props) {
  const { t } = useI18n();
  const [images, setImages] = useState<StoreImageInfo[]>([]);
  const [rows, setRows] = useState(0);
  const [dim, setDim] = useState(0);
  const [model, setModel] = useState("");
  const [status, setStatus] = useState<AssemblyStatus | null>(null);
  const [sets, setSets] = useState<LabelSetInfo[]>([]);
  const [activeSet, setActiveSet] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  // The row the preview is showing. Kept apart from the selection so a bulk
  // edit never costs you the image you were looking at.
  const [current, setCurrent] = useState<string>("");
  const [ngLabel, setNgLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  // Package import has no per-file progress to report — it is one request —
  // so it gets a line of its own rather than borrowing the ingest bar.
  const [pkgBusy, setPkgBusy] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  // The modal a long, opaque server call puts up. Assemble and migrate have no
  // progress channel; without this the tab simply froze for minutes.
  const [busyTitle, setBusyTitle] = useState<string | null>(null);
  const lastClicked = useRef<string | null>(null);
  const pkgRef = useRef<HTMLInputElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  // viewer
  const paneRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  // Zoom and pan are ONE state: pan is derived from the zoom step that
  // produced it, and holding them apart made a fast wheel spin apply only
  // its last event — every handler in the burst read the same stale zoom.
  const [view, setView] = useState({ z: 1, x: 0, y: 0 });
  const [markMode, setMarkMode] = useState(false);
  const [draft, setDraft] = useState<AnnotationRect | null>(null);
  const drag = useRef<{ kind: "pan" | "draw"; x: number; y: number; px: number; py: number } | null>(null);

  // Naming a group by hand. The rule-based modes moved to the teach tab,
  // where the sweep that uses them runs; this one stays because it writes to
  // the images selected in the list below.
  const [manualGroup, setManualGroup] = useState("");

  // Which project this tab's data belongs to. An import is many round trips,
  // and the user can leave for another project in the middle of one; every
  // long operation snapshots this and drops its results if it moved.
  const projectRef = useRef(projectId);
  useEffect(() => { projectRef.current = projectId; }, [projectId]);

  // The four setters of refreshStore, reusable against a listing the server
  // already handed us instead of a fresh GET. /store/delete and /store/group
  // both RETURN the complete new listing; refetching it was three round trips
  // where zero were needed.
  const applyListing = useCallback((s: StoreListResponse) => {
    setImages(s.images);
    setRows(s.total_rows);
    setDim(s.dim);
    setModel(s.model);
  }, []);

  // Just the image grid. An ingest changes only the store listing, so the
  // repaint after each chunk does not re-fetch assembly status and label sets
  // as well — those cannot have moved.
  //
  // The guard lives in the fetcher rather than at the call sites: the mount
  // effect's `cancelled` flag only decides whether a refresh is STARTED, so
  // two fast project switches can still let A's response resolve after B's and
  // win. Snapshotting here fixes every caller at once.
  const refreshStore = useCallback(async () => {
    const owner = projectRef.current;
    const s = await fetchStore();
    if (projectRef.current !== owner) return;
    applyListing(s);
  }, [applyListing]);

  // The non-store half of refresh(), for the sites whose image rows cannot move.
  const refreshMeta = useCallback(async () => {
    const owner = projectRef.current;
    const [a, l] = await Promise.all([fetchAssemblyStatus(), fetchLabelSets()]);
    if (projectRef.current !== owner) return;
    setStatus(a);
    setSets(l.labelsets);
    setActiveSet(l.active_id);
  }, []);

  const refresh = useCallback(async () => {
    await Promise.all([refreshStore(), refreshMeta()]);
  }, [refreshStore, refreshMeta]);

  // Functional per-row patch, the BankTab analogue of Operator's item patch.
  // Never reads current state, so overlapping in-flight edits cannot clobber.
  const patchImages = useCallback(
    (ids: Iterable<string>, patch: (im: StoreImageInfo) => StoreImageInfo) => {
      const set = ids instanceof Set ? ids : new Set(ids);
      setImages((prev) => prev.map((im) => (set.has(im.id) ? patch(im) : im)));
    }, []);

  // Optimistic edits paint immediately and settle on a trailing refetch, so a
  // run of keystrokes costs one listing parse instead of one per key.
  const reconcile = useRef<number | null>(null);
  const scheduleReconcile = useCallback(() => {
    if (reconcile.current !== null) window.clearTimeout(reconcile.current);
    reconcile.current = window.setTimeout(() => {
      reconcile.current = null;
      void refreshStore().catch(() => {});
    }, 400);
  }, [refreshStore]);
  useEffect(() => () => {
    if (reconcile.current !== null) window.clearTimeout(reconcile.current);
  }, []);

  useEffect(() => {
    if (!active || !projectId) return;
    let cancelled = false;
    selectBank(projectId)
      .then(() => (cancelled ? undefined : refresh()))
      .catch((e) => showToast(`bank select failed: ${(e as Error).message}`));
    return () => { cancelled = true; };
  }, [active, projectId, refresh, showToast]);

  const shown = useMemo(() => images.filter((im) => {
    if (filter === "all") return true;
    if (filter === "unlabeled") return im.tier === "";
    return im.tier === filter;
  }), [images, filter]);

  const counts = useMemo(() => {
    const c = { normal: 0, critical: 0, negative: 0, unlabeled: 0 };
    for (const im of images) {
      if (im.tier === "") c.unlabeled += 1;
      else c[im.tier] += 1;
    }
    return c;
  }, [images]);

  // Defect kinds that actually exist, with their counts. Typing a name into a
  // blank box is how a bank ends up with "scratch", "Scratch" and "scrach".
  const defectLabels = useMemo(() => {
    const m = new Map<string, number>();
    for (const im of images) {
      if (im.tier === "critical" && im.label) m.set(im.label, (m.get(im.label) ?? 0) + 1);
    }
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, [images]);

  // Which numbered steps are finished. Derived from the server's own view of
  // the store rather than remembered, which is what makes them clear
  // themselves: import 20 images and `unassigned` jumps off zero and the
  // assembly goes stale in the same response, so 2 and 3 untick without
  // anything having to notice that an import happened.
  const done = useMemo(() => {
    const has = images.length > 0;
    const labelled = has && !!status && status.unassigned === 0;
    const assembled = !!status && !status.stale && status.assigned > 0;
    return { 1: has, 2: labelled, 3: assembled };
  }, [images.length, status]);

  const currentImage = useMemo(
    () => images.find((im) => im.id === current) ?? null, [images, current],
  );
  const currentIndex = useMemo(
    () => shown.findIndex((im) => im.id === current), [shown, current],
  );

  // Keep the preview on something real: a filter change or a delete can take
  // the current row out from under it.
  useEffect(() => {
    if (shown.length === 0) { if (current) setCurrent(""); return; }
    if (!shown.some((im) => im.id === current)) setCurrent(shown[0].id);
  }, [shown, current]);

  // A new image starts fresh: carrying a 6x zoom onto the next photo hides
  // whatever you were about to judge.
  useEffect(() => { setView({ z: 1, x: 0, y: 0 }); setDraft(null); }, [current]);

  const measure = useCallback(() => {
    const el = paneRef.current;
    if (el) setBox({ w: el.clientWidth, h: el.clientHeight });
  }, []);

  useLayoutEffect(() => {
    const el = paneRef.current;
    if (!el) return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, [measure]);

  // The pane mounts inside a tab panel that may still be hidden, where it
  // measures zero — and the observer does not reliably deliver the moment the
  // panel is shown. Without this the preview stayed blank until the window
  // happened to be resized. Re-measure when the tab becomes active, and once
  // more after paint so a layout that settles late is picked up.
  useEffect(() => {
    if (!active) return;
    measure();
    const id = requestAnimationFrame(measure);
    return () => cancelAnimationFrame(id);
  }, [active, current, measure]);

  const stage = useMemo(() => {
    const ratio = currentImage && currentImage.height
      ? currentImage.width / currentImage.height : 0;
    return fitted(box, ratio);
  }, [box, currentImage]);

  const focusRow = useCallback((id: string) => {
    setCurrent(id);
    setSelected(new Set([id]));
    lastClicked.current = id;
    listRef.current?.querySelector(`[data-row="${id}"]`)?.scrollIntoView({ block: "nearest" });
  }, []);

  // ---- viewer -------------------------------------------------------------

  const onWheel = useCallback((e: React.WheelEvent) => {
    if (!stage.w) return;
    e.preventDefault();
    const pane = paneRef.current?.getBoundingClientRect();
    if (!pane) return;
    // Anchor the zoom on the cursor: zooming about the centre walks whatever
    // you were looking at off the screen and makes you chase it with pans.
    const cx = e.clientX - pane.left - pane.width / 2;
    const cy = e.clientY - pane.top - pane.height / 2;
    const up = e.deltaY < 0;
    setView((v) => {
      const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.z * (up ? 1.15 : 1 / 1.15)));
      if (next === MIN_ZOOM) return { z: 1, x: 0, y: 0 };
      const k = next / v.z;
      return { z: next, x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k };
    });
  }, [stage.w]);

  const normPoint = useCallback((clientX: number, clientY: number) => {
    const r = stageRef.current?.getBoundingClientRect();
    if (!r || !r.width || !r.height) return null;
    // getBoundingClientRect already reflects the transform, so this stays
    // correct at any zoom or pan without unwinding the matrix by hand.
    return {
      x: Math.min(1, Math.max(0, (clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (clientY - r.top) / r.height)),
    };
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (!stage.w) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    if (markMode) {
      const p = normPoint(e.clientX, e.clientY);
      if (!p) return;
      drag.current = { kind: "draw", x: p.x, y: p.y, px: 0, py: 0 };
      setDraft({ x: p.x, y: p.y, w: 0, h: 0 });
    } else {
      drag.current = { kind: "pan", x: e.clientX, y: e.clientY, px: view.x, py: view.y };
    }
  }, [markMode, normPoint, view.x, view.y, stage.w]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    if (d.kind === "pan") {
      setView((v) => ({ ...v, x: d.px + (e.clientX - d.x), y: d.py + (e.clientY - d.y) }));
      return;
    }
    const p = normPoint(e.clientX, e.clientY);
    if (!p) return;
    setDraft({
      x: Math.min(d.x, p.x), y: Math.min(d.y, p.y),
      w: Math.abs(p.x - d.x), h: Math.abs(p.y - d.y),
    });
  }, [normPoint]);

  const commitMark = useCallback(async (rects: AnnotationRect[]) => {
    if (!currentImage) return;
    const id = currentImage.id;
    const prev = currentImage.rects ?? [];
    // The shape is the user's — paint it now. Until this, the rectangle you
    // had just drawn visibly vanished and came back: onPointerUp clears the
    // draft at once and the committed rects only render once refresh() lands.
    //
    // The COUNT is not client-computable: /labelsets/mark resolves rectangles
    // against the store's sliding-window geometry and answers with the real
    // number of rows, so that arrives from the response.
    patchImages([id], (im) => ({ ...im, rects }));
    try {
      const r = await markStoreImage(id, rects);
      patchImages([id], (im) => ({ ...im, marks: r.marks }));
      setStatus(r.status);
    } catch (e) {
      patchImages([id], (im) => ({ ...im, rects: prev }));
      showToast(`mark failed: ${(e as Error).message}`);
      void refreshStore();
    }
  }, [currentImage, patchImages, refreshStore, showToast]);

  const onPointerUp = useCallback(() => {
    const d = drag.current;
    drag.current = null;
    if (!d || d.kind !== "draw") return;
    const r = draft;
    setDraft(null);
    if (!r || r.w < MIN_RECT || r.h < MIN_RECT || !currentImage) return;
    void commitMark([...(currentImage.rects ?? []), r]);
  }, [draft, currentImage, commitMark]);

  // ---- import (the expensive half) ----------------------------------------

  // Dropping files onto the tab imports them, the same gesture the Inspect tab
  // has always accepted. Named apart from the `drag` ref above, which is the
  // canvas pan/draw gesture and only shares the word.
  const [fileDragging, setFileDragging] = useState(false);
  const fileDragDepth = useRef(0);

  const onImport = useCallback(async (files: FileList | File[] | null) => {
    if (!files) return;
    const list = Array.from(files);
    if (list.length === 0) return;
    // Results belong to the project that was open when the import started.
    const owner = projectRef.current;
    const isStale = () => projectRef.current !== owner;

    // A zip goes to its own route: the server reads the members, so the
    // browser never unpacks anything and the names inside are left alone.
    const zips = list.filter((f) => /\.zip$/i.test(f.name));
    const images = list.filter((f) => !/\.zip$/i.test(f.name));

    setBusy(true);
    setProgress(images.length ? { done: 0, total: images.length } : null);
    const bad: string[] = [];
    try {
      for (let i = 0; i < images.length; i += INGEST_CHUNK) {
        if (isStale()) return;
        const chunk = images.slice(i, i + INGEST_CHUNK);
        const r = await ingestImages(chunk);
        if (isStale()) return;
        bad.push(...r.failed);
        // Every chunk answers with a fresh assembly status, so steps 2 and 3
        // untick as the import runs rather than all at once at the end.
        setStatus(r.status);
        setProgress({ done: Math.min(i + chunk.length, images.length), total: images.length });
        // Each chunk shows up as it lands. Waiting until the whole import
        // finished meant a 200-image drop sat on an unchanged grid for a
        // minute with only a counter moving, which reads as a hang.
        await refreshStore();
      }
      for (const z of zips) {
        if (isStale()) return;
        setBusyTitle(t("bank.importZipReading").replace("{name}", z.name));
        try {
          const r = await ingestZip(z);
          if (isStale()) return;
          bad.push(...r.failed);
          setStatus(r.status);
          await refreshStore();
          showToast(t("bank.importZipDone").replace("{n}", String(r.ingested)));
        } finally {
          setBusyTitle(null);
        }
      }
      if (bad.length) showToast(t("bank.someFailed").replace("{n}", String(bad.length)));
      // The last chunk already refetched the listing; only the label sets and
      // the assembly status still need a look.
      if (!isStale()) await refreshMeta();
    } catch (e) {
      if (!isStale()) showToast(`import failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  }, [refreshMeta, refreshStore, showToast, t]);

  // The bank as one file: features, marks, verdict settings and every taught
  // image. A package is minutes of upload plus a server-side extract, so the
  // buttons stay disabled and a line under them says it is still running.
  const onImportPackage = useCallback(async (file: File) => {
    setBusy(true);
    setPkgBusy(true);
    try {
      await importBank(file);
      await refresh();
      showToast(t("bank.pkgImported"));
    } catch (e) {
      showToast(`${t("bank.pkgImportFailed")}: ${(e as Error).message}`);
    } finally {
      setPkgBusy(false);
      setBusy(false);
    }
  }, [refresh, showToast, t]);

  const onMigrate = useCallback(async () => {
    const owner = projectRef.current;
    setBusy(true);
    // The migration re-carves every row and, by default, re-assembles the
    // whole bank to compare it — seconds to minutes, previously with a
    // disabled button as the only sign anything was happening.
    setBusyTitle(t("bank.migrating"));
    try {
      const r = await migrateStore();
      if (projectRef.current !== owner) return;
      showToast(t("bank.migrated").replace("{n}", fmt(r.images)).replace("{r}", fmt(r.rows)));
      setStatus(r.status);
      // The store is built from nothing here, so the full repaint is right.
      await refreshStore();
      await refreshMeta();
    } catch (e) {
      if (projectRef.current === owner) showToast(`migrate failed: ${(e as Error).message}`);
    } finally { setBusy(false); setBusyTitle(null); }
  }, [refreshStore, refreshMeta, showToast, t]);

  // ---- labelling (free, and as often as you like) -------------------------

  const onClickRow = useCallback((id: string, e: React.MouseEvent) => {
    if (e.shiftKey && lastClicked.current) {
      const ids = shown.map((im) => im.id);
      const a = ids.indexOf(lastClicked.current);
      const b = ids.indexOf(id);
      if (a >= 0 && b >= 0) {
        const next = new Set(selected);
        for (let i = Math.min(a, b); i <= Math.max(a, b); i++) next.add(ids[i]);
        setSelected(next);
        setCurrent(id);
        return;
      }
    }
    if (e.ctrlKey || e.metaKey) {
      setSelected((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id); else next.add(id);
        return next;
      });
      setCurrent(id);
      lastClicked.current = id;
      return;
    }
    focusRow(id);
  }, [shown, selected, focusRow]);

  const advance = useCallback(() => {
    // Step to the next row after an assign. With the "unlabelled" filter the
    // labelled row leaves the list, so the row that slides into this index is
    // already the next one to judge -- the rhythm is 1,1,2,1 straight down.
    const ids = shown.map((im) => im.id);
    const i = ids.indexOf(current);
    const next = ids[i + 1] ?? ids[i] ?? "";
    if (next) focusRow(next);
  }, [shown, current, focusRow]);

  // Labelling is the one thing done hundreds of times in a row, so it must not
  // wait on a round trip. `busy` is deliberately NOT set here.
  const assignSeq = useRef(0);

  const doAssign = useCallback(async (tier: Tier) => {
    if (selected.size === 0) return;
    const ids = [...selected];
    const owner = projectRef.current;
    const seq = ++assignSeq.current;
    const before = new Map(images.filter((im) => selected.has(im.id)).map((im) => [im.id, im]));

    // The server copies marks/rects from the previous assignment only when the
    // new tier is not "normal", so moving a row to normal has to strip them
    // here too, or the grid keeps a mark count the server just dropped.
    //
    // The label IS painted now. This used to be skipped because the listing
    // returned the label through safe_label(), which collapsed every
    // non-ASCII run — so a Japanese defect name did not round-trip to what
    // was typed. The note here said "it fills in on the reconcile"; it did
    // not, because the reconcile reads the same field. /store returns the
    // operator's own text today, so the optimistic value and the reconciled
    // one agree and the chip stops flickering back to a mangled name.
    patchImages(ids, (im) => (tier === "normal"
      ? { ...im, tier, label: "", severity: 0, marks: 0, rects: [] }
      : { ...im, tier, label: ngLabel }));
    advance();
    try {
      const r = await assignImages(ids, tier, tier === "normal" ? "" : ngLabel);
      if (projectRef.current !== owner || seq !== assignSeq.current) return;
      setStatus(r.status);
      scheduleReconcile();
    } catch (e) {
      if (projectRef.current !== owner) return;
      // One unknown id fails the WHOLE batch with nothing written, and a 409
      // means another client re-bound the bank — either way the optimistic
      // rows are fiction.
      setImages((prev) => prev.map((im) => before.get(im.id) ?? im));
      showToast(`assign failed: ${(e as Error).message}`);
      void refreshStore();
    }
  }, [selected, images, ngLabel, advance, patchImages, refreshStore, scheduleReconcile, showToast]);

  const doUnassign = useCallback(async () => {
    if (selected.size === 0) return;
    const ids = [...selected];
    const owner = projectRef.current;
    const seq = ++assignSeq.current;
    const before = new Map(images.filter((im) => selected.has(im.id)).map((im) => [im.id, im]));

    // Optimism is completely safe here: unassign is a pop and an unknown id is
    // a silent no-op, so there is no failure that leaves the row assigned.
    patchImages(ids, (im) => ({ ...im, tier: "", label: "", severity: 0, marks: 0, rects: [] }));
    try {
      const r = await unassignImages(ids);
      if (projectRef.current !== owner || seq !== assignSeq.current) return;
      setStatus(r.status);
      scheduleReconcile();
    } catch (e) {
      if (projectRef.current !== owner) return;
      setImages((prev) => prev.map((im) => before.get(im.id) ?? im));
      showToast(`unassign failed: ${(e as Error).message}`);
      void refreshStore();
    }
  }, [selected, images, patchImages, refreshStore, scheduleReconcile, showToast]);

  const onDelete = useCallback(async () => {
    if (selected.size === 0) return;
    if (!window.confirm(t("bank.removeConfirm").replace("{n}", String(selected.size)))) return;
    setBusy(true);
    try {
      // The response IS the new listing, so there is nothing to refetch.
      const r = await deleteFromStore([...selected]);
      setSelected(new Set());
      applyListing(r);
      // A delete drops the ids from every label set, so those counts do move.
      await refreshMeta();
    } catch (e) {
      showToast(`delete failed: ${(e as Error).message}`);
    } finally { setBusy(false); }
  }, [selected, applyListing, refreshMeta, showToast, t]);

  // Arrows move, 1 / 2 / 3 assign, 0 clears. The keyboard path is what makes a
  // 500-image pass bearable, so it is here from the start.
  useEffect(() => {
    if (!active) return;
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      // Every button carries disabled={busy}; the keys did not, so an ingest
      // or an assemble could be labelled straight through. Assigning no longer
      // sets `busy`, so this only blocks the genuinely exclusive jobs.
      if (busy) return;
      const ids = shown.map((im) => im.id);
      const i = ids.indexOf(current);
      if (e.key === "ArrowDown" || e.key === "ArrowRight") {
        e.preventDefault();
        if (ids.length) focusRow(ids[Math.min(ids.length - 1, i + 1)] ?? ids[0]);
      } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
        e.preventDefault();
        if (ids.length) focusRow(ids[Math.max(0, i - 1)] ?? ids[0]);
      } else if (e.key === "1") { e.preventDefault(); void doAssign("normal"); }
      else if (e.key === "2") { e.preventDefault(); void doAssign("critical"); }
      else if (e.key === "3") { e.preventDefault(); void doAssign("negative"); }
      else if (e.key === "0") { e.preventDefault(); void doUnassign(); }
      else if (e.key === "m" || e.key === "M") { e.preventDefault(); setMarkMode((v) => !v); }
      else if (e.key === "Escape") {
        if (markMode) setMarkMode(false);
        else setSelected(current ? new Set([current]) : new Set());
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, busy, shown, current, markMode, focusRow, doAssign, doUnassign]);

  // ---- label sets ---------------------------------------------------------

  const onDuplicate = useCallback(async () => {
    const name = window.prompt(t("label.newSetPrompt"), t("label.newSetDefault"));
    if (!name) return;
    setBusy(true);
    try {
      // Duplicating copies the assignments verbatim, so no image row can have
      // moved — only the set list and the name the panel shows.
      const r = await createLabelSet(name, true);
      setSets(r.labelsets);
      setActiveSet(r.active_id);
      setStatus(await fetchAssemblyStatus());
    } catch (e) {
      showToast(`create failed: ${(e as Error).message}`);
    } finally { setBusy(false); }
  }, [showToast, t]);

  const onSelectSet = useCallback(async (id: string) => {
    setBusy(true);
    try {
      // A different set gives every image a different tier/label/marks, so the
      // full refresh is legitimate here. Paint the chip first so the click
      // visibly lands while the grid catches up.
      const r = await selectLabelSet(id);
      setSets(r.labelsets);
      setActiveSet(r.active_id);
      await refresh();
    } catch (e) {
      showToast(`select failed: ${(e as Error).message}`);
    } finally { setBusy(false); }
  }, [refresh, showToast]);

  const onDeleteSet = useCallback(async (id: string) => {
    if (!window.confirm(t("label.deleteSetConfirm").replace("{id}", id))) return;
    setBusy(true);
    try {
      // Deleting the active set moves the pointer server-side, and every
      // image's tier changes with it, so the store refetch stays.
      const r = await deleteLabelSet(id);
      setSets(r.labelsets);
      setActiveSet(r.active_id);
      await refresh();
    } catch (e) {
      showToast(`delete failed: ${(e as Error).message}`);
    } finally { setBusy(false); }
  }, [refresh, showToast, t]);

  const onSetGroup = useCallback(async () => {
    if (selected.size === 0) return;
    setBusy(true);
    try {
      // The response is the whole listing. The assembly fingerprint hashes
      // only ids, row counts and assignments — `group` is in none of them —
      // and no assignment moved, so neither the status nor the label sets can
      // have changed. Three round trips become zero.
      applyListing(await setStoreGroup([...selected], manualGroup));
    } catch (e) {
      showToast(`group failed: ${(e as Error).message}`);
    } finally { setBusy(false); }
  }, [selected, manualGroup, applyListing, showToast]);

  const onAssemble = useCallback(async () => {
    const owner = projectRef.current;
    setBusy(true);
    setBusyTitle(t("bank.assembling"));
    try {
      // Assembly reads the label set without writing it, and the result
      // carries the new status, so no image row can have changed.
      const r = await assembleBank();
      if (projectRef.current !== owner) return;
      showToast(t("bank.assembled").replace("{n}", fmt(r.bank.normal + r.bank.critical + r.bank.negative)));
      setStatus(r.status);
    } catch (e) {
      if (projectRef.current === owner) showToast(`assemble failed: ${(e as Error).message}`);
    } finally { setBusy(false); setBusyTitle(null); }
  }, [showToast, t]);

  const bytes = useMemo(() => rows * dim * 2, [rows, dim]);
  const canMark = !!currentImage && currentImage.tier === "critical";
  const rects = currentImage?.rects ?? [];

  return (
    // Named like the other tabs' roots, so a layout probe can scope to it.
    <div
      className="bank-tab"
      style={{ position: "relative", display: "flex", gap: 12, height: "100%", minHeight: 0 }}
      onDragEnter={(e) => {
        if (!e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        fileDragDepth.current += 1;
        setFileDragging(true);
      }}
      onDragOver={(e) => { if (e.dataTransfer.types.includes("Files")) e.preventDefault(); }}
      onDragLeave={() => {
        fileDragDepth.current = Math.max(0, fileDragDepth.current - 1);
        if (fileDragDepth.current === 0) setFileDragging(false);
      }}
      onDrop={(e) => {
        if (!e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        fileDragDepth.current = 0;
        setFileDragging(false);
        if (e.dataTransfer.files.length) void onImport(e.dataTransfer.files);
      }}
    >
      {/* Whole-tab drop overlay, matching the Inspect tab: anywhere below the
          tab bar takes files. The guard above keeps the canvas's own drag
          gestures from raising it. */}
      {fileDragging && (
        <div style={{
          position: "absolute", inset: 0, zIndex: 60, pointerEvents: "none",
          border: `2px dashed ${ACCENT}`, borderRadius: 12,
          background: "rgba(34, 199, 219, 0.07)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <span style={{ fontSize: TYPE.hero, fontWeight: 700, color: ACCENT, textShadow: "0 1px 8px rgba(0,0,0,.25)" }}>
            {t("bank.dropOverlay")}
          </span>
        </div>
      )}
      {/* The dialog is the path you take when you need folders, renaming or a
          zip; the whole-tab drop above stays the fast path for a handful of
          already-well-named files, which is the gesture the Inspect tab and
          seg-studio both keep. */}
      <BankImportDialog
        open={importOpen}
        onClose={() => setImportOpen(false)}
        onImport={(files) => void onImport(files)}
      />
      {busyTitle && <BusyDialog title={busyTitle} hint={t("develop.busyHint")} border={border} />}
      {/* ---- left: the list ---- */}
      <div style={{ width: 288, flex: "none", display: "flex", flexDirection: "column", minHeight: 0, gap: 6 }}>
        {/* What changes the list, directly above the list it changes. Both of
            these lived in step 1 in the right column, which put "remove
            selected" 320px and a whole image viewer away from the rows you
            select it with -- you picked images here, then looked over there
            for the button that acts on them. Import belongs with them because
            the list is what gains the rows.
            Stacked rather than side by side: "選択した画像を取り除く" is ~174px
            at TYPE.md, so two buttons cannot share a 288px column without
            wrapping. The height comes out of the list, which has flex: 1. */}
        <div style={{ display: "flex", flexDirection: "column", gap: 5, flex: "none" }}>
          <button
            onClick={() => setImportOpen(true)}
            disabled={busy}
            style={{ padding: "6px 10px", fontSize: TYPE.md, borderRadius: 6, border: "none", background: ACCENT, color: "#fff", fontWeight: 600, cursor: busy ? "default" : "pointer" }}
          >
            {t("bank.import")}
          </button>
          <button onClick={() => void onDelete()} disabled={busy || !selected.size} style={{ ...chip(false), width: "100%", justifyContent: "center" }}>
            {t("bank.removeSelected")}
          </button>
        </div>
        {/* Wrapped, not scrolled. These were on one scrolling line while they
            were small enough to fit; at a size where the tier shapes actually
            read, the last one fell off the end -- and a filter you cannot see
            is a filter nobody uses. The two rows cost ~30px, which the list
            can spare now that labelling moved to the right column. */}
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", flex: "none" }}>
          <button onClick={() => setFilter("all")} style={chip(filter === "all", true)}>{t("label.all")} {images.length}</button>
          <button onClick={() => setFilter("unlabeled")} style={chip(filter === "unlabeled", true)}>{t("label.unlabeled")} {counts.unlabeled}</button>
          <button onClick={() => setFilter("normal")} style={chip(filter === "normal", true)}><Glyph tier="normal" /> {counts.normal}</button>
          <button onClick={() => setFilter("critical")} style={chip(filter === "critical", true)}><Glyph tier="critical" /> {counts.critical}</button>
          <button onClick={() => setFilter("negative")} style={chip(filter === "negative", true)}><Glyph tier="negative" /> {counts.negative}</button>
        </div>
        <div ref={listRef} style={{ flex: 1, overflow: "auto", minHeight: 0, border, borderRadius: 8 }}>
          {shown.length === 0 ? (
            <div className="muted" style={{ padding: 24, textAlign: "center", fontSize: TYPE.base }}>
              {images.length === 0 ? t("bank.empty") : t("label.none")}
            </div>
          ) : shown.map((im) => (
            <div
              key={im.id}
              data-row={im.id}
              onClick={(e) => onClickRow(im.id, e)}
              title={`${im.name}\n${fmt(im.rows)} rows`}
              style={{
                display: "flex", alignItems: "center", gap: 9, padding: "8px 10px",
                fontSize: TYPE.md, cursor: "pointer", userSelect: "none",
                borderLeft: im.id === current ? `4px solid ${ACCENT}` : "4px solid transparent",
                background: selected.has(im.id) ? ACCENT_SOFT : "transparent",
              }}
            >
              <Glyph tier={im.tier} size={15} />
              <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {im.name}
              </span>
              {im.label && <span style={{ fontSize: TYPE.base, color: VERM }}>{im.label}</span>}
              {im.group && <span className="muted" style={{ fontSize: TYPE.base, fontFamily: "ui-monospace, monospace" }}>{im.group}</span>}
              {im.marks > 0 && <span style={{ fontSize: TYPE.base, color: VERM, fontWeight: 700 }}>✎{im.marks}</span>}
            </div>
          ))}
        </div>
        <div className="muted" style={{ fontSize: TYPE.sm, flex: "none", fontVariantNumeric: "tabular-nums" }}>
          {shown.length > 0 && `${currentIndex + 1} / ${fmt(shown.length)}`}
        </div>
      </div>

      {/* ---- centre: the image, at a size you can judge ---- */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, minHeight: 0, gap: 8 }}>
        {/* Only what acts on the image in view. The label set, its progress
            and the import button used to share this line, which put a label
            fact and a bank action on the same strip as the zoom control --
            the same blur the right column had. */}
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap", minHeight: 27 }}>
          <button
            onClick={() => setMarkMode((v) => !v)}
            disabled={!canMark}
            title={canMark ? t("mark.hint") : t("mark.onlyDefect")}
            style={chip(markMode)}
          >
            {t("mark.mode")}
          </button>
          {view.z > 1 && (
            <button onClick={() => setView({ z: 1, x: 0, y: 0 })} style={chip(false)}>
              {view.z.toFixed(1)}× {t("mark.reset")}
            </button>
          )}
        </div>

        <div
          ref={paneRef}
          onWheel={onWheel}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onDoubleClick={() => setView({ z: 1, x: 0, y: 0 })}
          style={{
            flex: 1, minHeight: 0, border, borderRadius: 8,
            background: PANEL, position: "relative", overflow: "hidden",
            display: "flex", alignItems: "center", justifyContent: "center",
            cursor: markMode ? "crosshair" : view.z > 1 ? "grab" : "default",
            touchAction: "none",
          }}
        >
          {currentImage && currentImage.has_image && stage.w > 0 ? (
            <div
              ref={stageRef}
              style={{
                position: "absolute", width: stage.w, height: stage.h,
                transform: `translate(${view.x}px, ${view.y}px) scale(${view.z})`,
                transformOrigin: "center center",
              }}
            >
              <img
                key={currentImage.id}
                src={storeImageUrl(currentImage.id, "preview")}
                alt={currentImage.name}
                draggable={false}
                style={{ width: "100%", height: "100%", objectFit: "fill", display: "block", userSelect: "none" }}
              />
              <svg
                viewBox="0 0 1 1" preserveAspectRatio="none"
                style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
              >
                {/* non-scaling-stroke keeps the outline one screen pixel wide
                    at any zoom; without it a 0-1 viewBox scaled to the pane
                    turns a hairline into a slab that hides the defect. */}
                {rects.map((r, i) => (
                  <rect
                    key={i} x={r.x} y={r.y} width={r.w} height={r.h}
                    fill={`${VERM}22`} stroke={VERM} strokeWidth={1.5}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
                {draft && (
                  <rect
                    x={draft.x} y={draft.y} width={draft.w} height={draft.h}
                    fill={`${VERM}22`} stroke={VERM} strokeDasharray="4 3"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
              </svg>
            </div>
          ) : (
            <span className="muted" style={{ fontSize: TYPE.base }}>
              {currentImage ? t("bank.noPreview") : t("label.none")}
            </span>
          )}
        </div>

        {currentImage && (
          <div className="muted" style={{ fontSize: TYPE.sm, display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
            <span>{currentImage.name}</span>
            <span>{currentImage.width}×{currentImage.height}</span>
            <span>{fmt(currentImage.rows)} rows</span>
            {currentImage.grid_rows > 0 && currentImage.grid_rows !== currentImage.rows && (
              <span>/ {fmt(currentImage.grid_rows)} grid</span>
            )}
            {rects.length > 0 && (
              <>
                <span style={{ color: VERM, fontWeight: 600 }}>
                  {t("mark.count").replace("{r}", String(rects.length)).replace("{p}", fmt(currentImage.marks))}
                </span>
                <button onClick={() => void commitMark([])} disabled={busy} style={chip(false)}>{t("mark.clear")}</button>
              </>
            )}
            {markMode && <span style={{ color: VERM }}>{t("mark.drawing")}</span>}
          </div>
        )}

        {/* Only the import progress lives under the image. Every other control
            moved to the right column: the pane is the one thing here that
            wants height, and stacking rows beneath it was taking that height
            while a whole column sat empty. */}
        {progress && (
          <div style={{ borderTop: border, paddingTop: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: TYPE.base }}>
              <span>{t("bank.extracting").replace("{d}", String(progress.done)).replace("{t}", String(progress.total))}</span>
            </div>
            <div style={{ height: 5, borderRadius: 99, background: BORDER, overflow: "hidden", marginTop: 4 }}>
              <i style={{ display: "block", height: "100%", width: `${(progress.done / Math.max(1, progress.total)) * 100}%`, background: ACCENT }} />
            </div>
          </div>
        )}
      </div>

      {/* ---- right: the pass, in the order it is done ----
           These controls were interleaved -- assemble, tier buttons, defect
           kinds, bank figures, label sets, grouping -- so "how many rows the
           bank holds" sat between two labelling controls and every glance had
           to sort bank from label before it could read anything.
           Numbered, because the order is not guessable from the controls:
           labelling changes nothing that inspection sees until step 3, and a
           panel sitting above another reads as "do this first" whether or not
           that is true. Get images in, judge them, fold the judgement in,
           then measure. */}
      <div style={{ width: 320, flex: "none", display: "flex", flexDirection: "column", gap: 6, borderLeft: border, paddingLeft: 12, minHeight: 0, overflow: "auto" }}>

        <Panel step={1} done={done[1]} title={t("panel.bank")} note={t("panel.bankNote")} hint={t("panel.bankHint")}>
          {/* Two dense lines, not four aligned rows. These are reference
              figures, and as four rows they cost the height that pushed step
              4 off the bottom of the column -- the only step that is a
              control nobody can guess is there. */}
          <div className="muted" style={{ fontSize: TYPE.base, display: "flex", flexWrap: "wrap", gap: "1px 10px" }}>
            <span>{t("bank.imageCount")} <b style={{ color: INK, fontVariantNumeric: "tabular-nums" }}>{fmt(images.length)}</b></span>
            <span>{t("bank.rowCount")} <b style={{ color: INK, fontVariantNumeric: "tabular-nums" }}>{fmt(rows)}</b></span>
            <span>{t("bank.size")} <b style={{ color: INK, fontVariantNumeric: "tabular-nums" }}>{(bytes / 1024 ** 2).toFixed(0)} MB</b></span>
          </div>
          <div className="muted" style={{ fontSize: TYPE.base, display: "flex", gap: 6, minWidth: 0 }}>
            <span style={{ flex: "none" }}>{t("bank.encoder")}</span>
            <b style={{ color: INK, fontFamily: "ui-monospace, monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{model || "—"}</b>
          </div>

          {/* Import and remove-selected used to head this block; they now sit
              over the list, which is what they change and where the selection
              is made. What is left is bank-level and stays: a zip is the whole
              bank at once, not the rows you have highlighted. */}
          <div style={{ borderTop: border, paddingTop: 6, display: "flex", flexDirection: "column", gap: 5 }}>
            {/* The whole bank as one zip, in and out. This lived beside the
                plots on the check tab, where it was the only real control in
                a band of figures — but it is bank-level, so it belongs with
                the step that owns what the bank holds. */}
            <div style={{ display: "flex", gap: 5 }}>
              <button
                onClick={() => { window.location.href = bankExportUrl(); }}
                disabled={busy || rows === 0}
                title={t("bank.pkgExportHint")}
                style={{ ...chip(false), flex: 1, justifyContent: "center" }}
              >⬇ {t("bank.pkgExport")}</button>
              <button
                onClick={() => pkgRef.current?.click()}
                disabled={busy}
                title={t("bank.pkgImportHint")}
                style={{ ...chip(false), flex: 1, justifyContent: "center" }}
              >⬆ {t("bank.pkgImport")}</button>
            </div>
            <input
              ref={pkgRef} type="file" accept=".zip" hidden
              onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) void onImportPackage(f); }}
            />
            {pkgBusy && <span className="muted" style={{ fontSize: TYPE.sm }}>{t("bank.pkgImporting")}</span>}
          </div>

          {status && !status.migrated && (
            <div style={{ borderTop: border, paddingTop: 6, display: "flex", flexDirection: "column", gap: 6 }}>
              <span className="muted" style={{ fontSize: TYPE.sm }}>{t("bank.migrateHint")}</span>
              <button onClick={onMigrate} disabled={busy} style={{ padding: "6px 10px", borderRadius: 6, border, background: "transparent", color: INK, cursor: "pointer" }}>
                {t("bank.migrate")}
              </button>
            </div>
          )}

        </Panel>

        <Panel step={2} done={done[2]} title={t("panel.label")} note={t("panel.labelNote")} hint={t("panel.labelHint")}>
          {/* Which set the judgement lands in. It was in the strip over the
              image, two columns away from the tier buttons that write it. */}
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span className="muted" style={{ fontSize: TYPE.base, flex: "none" }}>{t("label.set")}</span>
            <b style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: TYPE.md }}>
              {status?.labelset_name || activeSet}
            </b>
            <button onClick={onDuplicate} disabled={busy} style={chip(false)}>{t("label.duplicate")}</button>
          </div>
          {/* The list only earns its space once there is a choice to make. */}
          {sets.length > 1 && sets.map((s) => (
            <div key={s.id} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              {/* Gated on busy: /labelsets/select is not bank-bound and the
                  active set is read when the request is handled, so switching
                  mid-assign lands that assign in the newly selected set. */}
              <button onClick={() => void onSelectSet(s.id)} disabled={busy} style={{ ...chip(s.id === activeSet), flex: 1, minWidth: 0, justifyContent: "space-between" }}>
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.name}</span>
                <span style={{ fontFamily: "ui-monospace, monospace", fontSize: TYPE.base }}>
                  {s.counts.normal ?? 0}/{s.counts.critical ?? 0}/{s.counts.negative ?? 0}
                </span>
              </button>
              <button onClick={() => void onDeleteSet(s.id)} disabled={busy} title={t("label.deleteSet")} style={{ ...chip(false), padding: "3px 6px" }}>×</button>
            </div>
          ))}
          {/* The "{a} of {t} labelled" line was here and is gone: the filter
              chips over the list already carry it, broken down by tier. */}

          {/* Full-width rows so the tier is aimed at rather than picked out
              of a strip of chips. */}
          <div style={{ borderTop: border, paddingTop: 6, display: "flex", flexDirection: "column", gap: 5 }}>
            <b style={{ fontSize: TYPE.base }}>
              {t("label.selected").replace("{n}", String(selected.size))}
            </b>
            <button onClick={() => void doAssign("normal")} disabled={busy || !selected.size} style={{ ...chip(false, true), justifyContent: "flex-start", width: "100%" }}>
              <Glyph tier="normal" /> <Key>1</Key> {t("label.tier.normal")}
            </button>
            <button onClick={() => void doAssign("critical")} disabled={busy || !selected.size} style={{ ...chip(false, true), justifyContent: "flex-start", width: "100%" }}>
              <Glyph tier="critical" /> <Key>2</Key> {t("label.tier.critical")}
            </button>
            <button onClick={() => void doAssign("negative")} disabled={busy || !selected.size} style={{ ...chip(false, true), justifyContent: "flex-start", width: "100%" }}>
              <Glyph tier="negative" /> <Key>3</Key> {t("label.tier.negative")}
            </button>
            {/* Only the reversible one stays here. Clearing a label and
                deleting the features were adjacent halves of one row, and
                they are not the same act -- the second cannot be undone
                without running the encoder again. */}
            <button onClick={() => void doUnassign()} disabled={busy || !selected.size} style={{ ...chip(false), width: "100%", justifyContent: "center" }}>
              {t("label.clear")}
            </button>
            <span className="muted" style={{ fontSize: TYPE.sm }}>{t("label.keysHint")}</span>
          </div>

          {/* Naming a lot by hand, so a grouped run on the teach tab can hold
              the whole lot out. It sits with the tier buttons because it is
              the same act on the same selection: a judgement written onto the
              images picked in the list. */}
          <div style={{ borderTop: border, paddingTop: 6, display: "flex", flexDirection: "column", gap: 5 }}>
            <b style={{ fontSize: TYPE.base }}>{t("label.group.assign")}</b>
            <div style={{ display: "flex", gap: 5 }}>
              <input
                value={manualGroup} onChange={(e) => setManualGroup(e.target.value)}
                placeholder={t("label.group.name")}
                style={{ flex: 1, minWidth: 0, padding: "3px 6px", borderRadius: 6, border, background: "transparent", color: INK, fontSize: TYPE.base }} />
              <button onClick={() => void onSetGroup()} disabled={busy || !selected.size} style={chip(false)}>
                {t("label.group.apply")}
              </button>
            </div>
          </div>

          {/* Defect kinds that exist, so a second one is picked rather than
              retyped -- retyping is how a bank grows "scratch" and "scrach". */}
          <div style={{ borderTop: border, paddingTop: 6, display: "flex", flexDirection: "column", gap: 5 }}>
            <b style={{ fontSize: TYPE.base }}>{t("label.defectKind")}</b>
            <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
              {defectLabels.map(([lab, n]) => (
                <button key={lab} onClick={() => setNgLabel(lab === ngLabel ? "" : lab)} style={chip(lab === ngLabel, true)}>
                  <Glyph tier="critical" /> {lab} {n}
                </button>
              ))}
            </div>
            <input
              value={ngLabel} onChange={(e) => setNgLabel(e.target.value)}
              placeholder={t("label.newKind")}
              style={{ padding: "5px 10px", borderRadius: 6, border, background: "transparent", color: INK, fontSize: TYPE.md }}
            />
          </div>
        </Panel>

        {/* Its own step, not a footer under the bank figures. Relabelling
            changes nothing the inspection tab can see until this runs, and a
            control that turns work into effect should not read as one more
            statistic about the store. */}
        {status && status.migrated && (
          <Panel step={3} done={done[3]} title={t("panel.assemble")} note={t("panel.assembleNote")} hint={t("panel.assembleHint")}>
            {status.stale ? (
              <span style={{ fontSize: TYPE.sm, color: VERM, fontWeight: 600 }}>⚠ {t("bank.stale")}</span>
            ) : (
              <span style={{ fontSize: TYPE.sm, color: DONE }}>✓ {t("label.upToDate")}</span>
            )}
            <button
              onClick={onAssemble}
              disabled={busy || status.assigned === 0}
              style={{
                padding: "6px 10px", fontSize: TYPE.md, borderRadius: 6, border: "none",
                background: status.stale ? ACCENT : "transparent",
                color: status.stale ? "#fff" : INK,
                fontWeight: status.stale ? 600 : 400,
                boxShadow: status.stale ? "none" : `inset 0 0 0 1px ${BORDER}`,
                cursor: busy ? "default" : "pointer",
              }}
            >
              {t("bank.assemble")}
            </button>
          </Panel>
        )}

      </div>
    </div>
  );
}

/** One concern per panel, named, on its own surface.
 *
 *  The sections it replaces were divided by a hairline rule and nothing else,
 *  which is enough to separate two paragraphs about the same subject and not
 *  enough to separate two subjects. The heading says which one this is, and
 *  the note says it in words, because "bank" and "label set" are both nouns
 *  this screen uses and neither name alone tells you which is the judgement. */
function Panel({ step, done, title, note, hint, children }: {
  step: number;
  /** Finished. Omitted by steps that are a setting rather than an act. */
  done?: boolean;
  title: string; note: string; hint: string; children: React.ReactNode;
}) {
  const { t } = useI18n();
  return (
    <section
      // The tutorial spotlights these by number: they are the three moves the
      // tab exists to walk you through, so they are what it has to point at.
      data-tutorial-step={`bank-step-${step}`}
      style={{
        border: done ? `1px solid ${DONE_EDGE}` : border,
        borderRadius: 8, background: PANEL,
        padding: "5px 9px 7px", display: "flex", flexDirection: "column", gap: 4, flex: "none",
      }}
    >
      {/* One line, and it has to stay one line: at three lines the notes cost
          more height than a whole panel, which then fell off the bottom of the
          column. The long version is the tooltip. */}
      <header title={hint} style={{ display: "flex", alignItems: "center", gap: 6, borderBottom: border, paddingBottom: 4 }}>
        {/* A disc, so the step number cannot be read as one of the 1/2/3 keys
            that assign a tier -- those are drawn as keycaps two panels down.
            Finished, it holds a tick instead: the discs are the loudest thing
            in the column, so they are where "this one is behind you" has to
            be said, and a tick against a digit is a difference in SHAPE, not
            just in colour. Costs no height, which the column has none of. */}
        <span style={{
          flex: "none", width: 17, height: 17, borderRadius: "50%",
          display: "inline-flex", alignItems: "center", justifyContent: "center",
          background: done ? DONE : ACCENT, color: "#fff",
          fontSize: TYPE.xs, fontWeight: 700, fontVariantNumeric: "tabular-nums",
        }}>{done ? "✓" : step}</span>
        <b style={{ fontSize: TYPE.md, flex: "none" }}>{title}</b>
        <span className="muted" style={{ fontSize: TYPE.xs, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{note}</span>
        {done && (
          <span style={{ flex: "none", marginLeft: "auto", fontSize: TYPE.xs, fontWeight: 600, color: DONE }}>{t("panel.done")}</span>
        )}
      </header>
      {children}
    </section>
  );
}

/** The keyboard shortcut on a tier button, drawn as a key rather than a bare
 *  digit: the panels are numbered now, and "2" beside a tier read as "step 2"
 *  the moment they were. */
function Key({ children }: { children: React.ReactNode }) {
  return (
    <b style={{
      flex: "none", minWidth: 16, padding: "0 3px", border, borderRadius: 4,
      textAlign: "center", fontSize: TYPE.xs, lineHeight: "15px",
      fontVariantNumeric: "tabular-nums", color: MUTED,
    }}>{children}</b>
  );
}

/** ``big`` for anything carrying a tier glyph: those are the controls the
 *  operator aims at hundreds of times in a pass, and they were sized like
 *  incidental metadata. */
function chip(on: boolean, big = false): React.CSSProperties {
  return {
    display: "inline-flex", alignItems: "center", gap: big ? 7 : 5,
    border: on ? `1px solid ${ACCENT}` : border,
    background: on ? ACCENT_SOFT : "transparent",
    color: on ? ACCENT : INK,
    fontWeight: on ? 700 : 400,
    borderRadius: 99, padding: big ? "6px 13px" : "4px 10px",
    fontSize: big ? TYPE.md : TYPE.base, cursor: "pointer",
    whiteSpace: "nowrap", flex: "none",
  };
}

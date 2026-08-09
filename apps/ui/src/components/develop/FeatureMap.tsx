// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../../i18n";
import {
  ALPHA_MAX,
  fetchEvalCmin,
  fetchProjection,
  patchFloorComposite,
  type BankImage,
  type BankState,
  type ProjectionPoint,
  type ProjectionResponse,
  type StoredImageEval,
  type Tier,
} from "../../api/cls";
import { ACCENT, INK, TYPE} from "../../ui/tokens";
import SeparationEval from "./SeparationCheck";
import {
  EVAL_FP_COLOR,
  aurocOf,
  statWithAlpha,
  youdenThreshold,
  type EvalVerdictKind,
  type EvalVerdicts,
} from "./separation";
import { VerdictAnchors } from "./types";

// Height of the feature map. The separation check sizes itself (histogram,
// then the controls under it), but the map is a plot with no intrinsic height
// at all -- without a number here it collapses to its own controls.
const MAP_FIG_H = 300;

// ---------- Bank projection panel (2D scatter of taught patches) ----------

type ProjectionMode = "auto" | "normal" | "anomaly";

const TIER_COLOR: Record<Tier, string> = {
  normal: "rgba(140, 150, 170, .55)",
  critical: "rgba(217, 91, 91, .85)",
  negative: "rgba(78, 158, 255, .85)",
};

export default function BankProjection({
  activeBankId,
  bank,
  images,
  onOpenImage,
  onRawAnchors,
  onEvalVerdicts,
  onRunAll,
  runAllDisabled,
  evalRunRef,
  border,
  showToast,
}: {
  activeBankId: string;
  bank: BankState | null;
  images: BankImage[];
  onOpenImage: (im: BankImage) => void;
  onRawAnchors: (a: VerdictAnchors | null) => void;
  /** Published upward for the image list, the same way onRawAnchors is. */
  onEvalVerdicts: (v: EvalVerdicts) => void;
  onRunAll: () => void;
  runAllDisabled: boolean;
  evalRunRef: React.MutableRefObject<(() => Promise<boolean>) | null>;
  border: string;
  showToast: (msg: string) => void;
}) {
  const { t } = useI18n();
  // The histogram is the one people read; the map is the second opinion.
  const [chartView, setChartView] = useState<"eval" | "map">("eval");
  // The plot fills the card width (no side gutters), so its shape is
  // rectangular. Track its real aspect ratio: the data range is extended to
  // match it, keeping both axes on the SAME scale — filling the card must not
  // stretch distances (a tight normal cluster must not look smeared flat).
  const plotRef = useRef<HTMLDivElement | null>(null);
  const [plotAspect, setPlotAspect] = useState(1);
  useLayoutEffect(() => {
    const el = plotRef.current;
    if (!el) return;
    const sync = () => {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) setPlotAspect(r.width / r.height);
    };
    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const [proj, setProj] = useState<ProjectionResponse | null>(null);
  const [mode, setMode] = useState<ProjectionMode>("auto");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  // Dot coloring: by tier, or by anomaly score (top-k distance to the
  // normal bank, computed server-side when requested).
  const [colorBy, setColorBy] = useState<"tier" | "score">("tier");
  // Point granularity: raw patches (with each image's cached top patches
  // guaranteed into the sample) or one aggregated point per taught image
  // (mean of its top-10 patch features, fixed — the map must not reshape
  // when the separation check's k slider moves). Image dots still follow
  // the check live for score color and FP / miss marks.
  const [granularity, setGranularity] = useState<"patch" | "image">("image");

  // Separation-check state lives up here so the map and the check stay in
  // sync: the metric picked in the check (max / p99 / top-k) also scores the
  // image-level map, and the check's threshold marks FP / miss dots on it.
  const [evalResults, setEvalResults] = useState<Map<string, StoredImageEval>>(new Map());
  // A bank (or project) switch invalidates every per-image evaluation on
  // screen — clear instead of waiting for the refetch to overwrite, so the
  // panel never shows the previous bank's numbers while the new one loads.
  useEffect(() => { setEvalResults(new Map()); }, [activeBankId]);
  // top-k mean is the default metric — it beat max/p99 on every bank tried
  // so far (small defects live in the top ~10 patches), and k=10 matches the
  // server-side aggregation default. There is no separate "max" option:
  // max ≡ top-k mean at k=1, so the k sweep already covers it.
  const [evalMetric, setEvalMetric] = useState<"p99" | "topk">("topk");
  const [evalTopK, setEvalTopK] = useState(10);

  // α exemplar boost: composite patch score = raw + α/(ε + cmin), where cmin
  // is the min distance to the NG exemplar rows (marked rows, else each NG's
  // auto top-10; own image excluded server-side). The server returns cmin
  // aligned with the cached top_scores, so sweeping α is a pure re-render.
  // Shared with the Operator tab via localStorage: the α proven here is the
  // one inspection runs with.
  const [alpha, setAlpha] = useState<number>(() => {
    const v = Number(localStorage.getItem("anom.alpha"));
    // Clamped, not discarded: values saved before the term was bounded run to
    // 840, and snapping those to 0 would silently turn the boost off for
    // anyone who had tuned it.
    return Number.isFinite(v) && v > 0 ? Math.min(v, ALPHA_MAX) : 0;
  });
  useEffect(() => { localStorage.setItem("anom.alpha", String(alpha)); }, [alpha]);
  const [cminMap, setCminMap] = useState<Map<string, number[]>>(new Map());
  const alphaOn = alpha > 0;
  // Mark edits must refetch cmin even though eval results are unchanged.
  const annotFingerprint = useMemo(
    () => images
      .filter((im) => im.tier === "critical" && im.annotations?.length)
      .map((im) => `${im.label}/${im.name}:${(im.annotations ?? []).map((r) => `${r.x},${r.y},${r.w},${r.h}`).join(";")}`)
      .join("|"),
    [images],
  );
  useEffect(() => {
    if (!alphaOn || evalResults.size === 0) { setCminMap(new Map()); return; }
    let cancelled = false;
    // Small debounce: evalResults updates once per image during a sweep.
    const tid = setTimeout(() => {
      fetchEvalCmin()
        .then((rows) => {
          if (cancelled) return;
          const m = new Map<string, number[]>();
          rows.forEach((r) => m.set(`${r.tier}/${r.label}/${r.name}`, r.top_cmin));
          setCminMap(m);
        })
        .catch(() => { if (!cancelled) setCminMap(new Map()); });
    }, 300);
    return () => { cancelled = true; clearTimeout(tid); };
  }, [alphaOn, evalResults, annotFingerprint]);

  // Image-level statistic under the selected metric. "topk" is the mean of
  // the k hottest patches, computed client-side from the returned top_scores
  // so moving the slider needs no re-scoring. With α > 0 every metric reads
  // from the re-ranked composite list instead (p99 approximated at the 1%
  // rank of the full patch count, which lies inside the cached top-256).
  const valOf = useCallback((r: StoredImageEval, kind: "p99" | "topk", k: number): number =>
    statWithAlpha(r, kind, k, alpha, cminMap.get(`${r.tier}/${r.label}/${r.name}`)),
  [alpha, cminMap]);

  // Sweep α over the slider range for the best separation and jump there.
  // AUROC decides; the normalized gap between the worst NG and the best OK
  // breaks ties (many α values reach AUROC 1.0 — pick the safest one).
  // cmin is fetched fresh because the state map is only populated while
  // α > 0; the sweep must work when starting from α = 0 too.
  // ``metricArg``/``kArg`` override the lifted state — the auto-tune chain
  // computes best-k and immediately sweeps α for THAT k, before React has
  // committed the setMetric/setTopK updates.
  const findBestAlpha = useCallback(async (metricArg?: "p99" | "topk", kArg?: number) => {
    const m = metricArg ?? evalMetric;
    const k = kArg ?? evalTopK;
    const entries = [...evalResults.values()];
    const ok = entries.filter((r) => r.tier === "normal");
    const ng = entries.filter((r) => r.tier === "critical");
    if (!ok.length || !ng.length) return;
    const cm = new Map<string, number[]>();
    try {
      (await fetchEvalCmin()).forEach((r) => cm.set(`${r.tier}/${r.label}/${r.name}`, r.top_cmin));
    } catch { /* without cmin the sweep degrades to α=0 raw scores */ }
    const stat = (r: StoredImageEval, a: number) =>
      statWithAlpha(r, m, k, a, cm.get(`${r.tier}/${r.label}/${r.name}`));
    let best = { a: 0, auroc: -1, margin: -Infinity };
    // Bounded by the slider's own ceiling. The sweep used to run to 1000 while
    // ALPHA_MAX was lowered to 200 around it, so 80 of the 101 candidates were
    // values neither slider nor the localStorage read could hold: the tuner
    // announced a best α the UI then quietly clamped away.
    for (let a = 0; a <= ALPHA_MAX; a += 5) {
      const okV = ok.map((r) => stat(r, a));
      const ngV = ng.map((r) => stat(r, a));
      const auroc = aurocOf(ngV, okV);
      const lo = Math.min(...okV, ...ngV), hi = Math.max(...okV, ...ngV);
      const margin = hi > lo ? (Math.min(...ngV) - Math.max(...okV)) / (hi - lo) : -Infinity;
      if (auroc > best.auroc + 1e-9 || (Math.abs(auroc - best.auroc) <= 1e-9 && margin > best.margin)) {
        best = { a, auroc, margin };
      }
    }
    setAlpha(best.a);
    // α can only lift an NG that has exemplar rows to be near. A single
    // taught NG excludes its own rows, so any α > 0 boosts only the OK side
    // and the sweep collapses to 0 — explain that instead of a bare result.
    const ngHasCmin = ng.some((r) => {
      const c = cm.get(`${r.tier}/${r.label}/${r.name}`);
      return !!c && c.length === (r.top_scores ?? []).length;
    });
    if (best.a === 0 && !ngHasCmin) {
      showToast(t("develop.eval.alphaNeedsTwo"));
    } else {
      showToast(t("develop.eval.bestAlphaFound")
        .replace("{a}", String(best.a))
        .replace("{auroc}", best.auroc.toFixed(4))
        .replace("{margin}", `${Math.round(best.margin * 100)}%`));
    }
  }, [evalResults, evalMetric, evalTopK, setAlpha, showToast, t]);

  // Manual threshold override; null = follow the Youden suggestion. Reset
  // whenever the statistic's scale changes (bank / metric / k / α), because
  // a value picked on one scale is meaningless on another.
  const [thrOverride, setThrOverride] = useState<number | null>(null);
  useEffect(() => { setThrOverride(null); }, [activeBankId, evalMetric, evalTopK, alpha]);

  const evalStats = useMemo(() => {
    const entries = [...evalResults.entries()];
    if (!entries.length) return null;
    const val = (r: StoredImageEval) => valOf(r, evalMetric, evalTopK);
    const ok = entries.filter(([, r]) => r.tier === "normal");
    const ng = entries.filter(([, r]) => r.tier === "critical");
    const fp = entries.filter(([, r]) => r.tier === "negative");
    const okV = ok.map(([, r]) => val(r));
    const ngV = ng.map(([, r]) => val(r));
    if (!okV.length || !ngV.length) return null;
    const auroc = aurocOf(ngV, okV);
    // Suggested threshold: maximize Youden's J over candidate midpoints. A
    // manual override takes precedence for everything downstream — the
    // miss / FP chips, the map marks and the histogram line all follow it.
    const autoThr = youdenThreshold(okV, ngV);
    const thr = thrOverride ?? autoThr;
    const misses = ng.filter(([, r]) => val(r) <= thr);
    const falsePos = ok.filter(([, r]) => val(r) > thr);
    return { okV, ngV, fpV: fp.map(([, r]) => val(r)), auroc, thr, autoThr, misses, falsePos, val };
  }, [evalResults, evalMetric, evalTopK, valOf, thrOverride]);

  // One verdict per swept image, for the image list's result column. Derived
  // here, where the metric, k, α and the threshold override all live, so the
  // list can never end up reading a threshold this card has moved off.
  const evalVerdicts = useMemo<EvalVerdicts>(() => {
    if (!evalStats) return null;
    const { val, thr } = evalStats;
    const m = new Map<string, { v: number; kind: EvalVerdictKind }>();
    for (const [k, r] of evalResults) {
      const v = val(r);
      const kind: EvalVerdictKind =
        r.tier === "normal" ? (v > thr ? "fp" : "ok")
        : r.tier === "critical" ? (v > thr ? "ng" : "miss")
        : "neg";
      m.set(k, { v, kind });
    }
    return m;
  }, [evalResults, evalStats]);
  useEffect(() => { onEvalVerdicts(evalVerdicts); }, [evalVerdicts, onEvalVerdicts]);

  // Raw-statistic threshold for the map's PATCH dots only: projection patch
  // dots carry raw scores (cmin exists just for each image's cached top
  // rows), so once α > 0 the composite threshold lives on a different scale
  // — Youden's J over the raw metric keeps "which patches drive the
  // verdicts" meaningful at any α. (Preview heatmaps are α-aware server-side
  // and use the composite anchors lifted below instead.)
  const rawStats = useMemo(() => {
    const entries = [...evalResults.values()];
    if (!entries.length) return null;
    const raw = (r: StoredImageEval): number => {
      if (evalMetric === "p99") return r.score_p99;
      const ts = r.top_scores ?? [];
      if (!ts.length) return r.score_max;
      const kk = Math.min(evalTopK, ts.length);
      let s = 0;
      for (let i = 0; i < kk; i++) s += ts[i];
      return s / kk;
    };
    const okV = entries.filter((r) => r.tier === "normal").map(raw).sort((a, b) => a - b);
    const ngV = entries.filter((r) => r.tier === "critical").map(raw);
    if (!okV.length || !ngV.length) return null;
    // OK-distribution mean/std anchor the T-score (偏差値) display: raw
    // scores are nearest-normal distances whose absolute scale differs per
    // bank, but "50 ± 10 = an ordinary OK" reads the same everywhere.
    const okMean = okV.reduce((a, b) => a + b, 0) / okV.length;
    const okStd = Math.sqrt(okV.reduce((a, b) => a + (b - okMean) ** 2, 0) / okV.length);
    return { thr: youdenThreshold(okV, ngV), okMedian: okV[Math.floor(okV.length / 2)], okMean, okStd };
  }, [evalResults, evalMetric, evalTopK]);
  const rawThr = rawStats?.thr ?? null;

  // Report the verdict-scale anchors up: the preview heatmap and the NG
  // degree render on the SAME α-composite scale the separation check judges
  // on, with the operative threshold (auto or hand-tuned) at the midpoint —
  // so heatmap colour, NG度 and the actual verdict always agree. The server
  // heatmap is α-aware too (/score composes raw + α/(ε+cmin) per patch when
  // it gets alpha). Debounced: dragging the α / threshold sliders re-anchors
  // once the user settles instead of nuking the heatmap cache every step.
  useEffect(() => {
    if (!evalStats || !evalStats.okV.length) { onRawAnchors(null); return; }
    const okV = [...evalStats.okV].sort((a, b) => a - b);
    const lo = okV[Math.floor(okV.length / 2)];
    if (!(evalStats.thr > lo)) { onRawAnchors(null); return; }
    const okMean = okV.reduce((a, b) => a + b, 0) / okV.length;
    const okStd = Math.sqrt(okV.reduce((a, b) => a + (b - okMean) ** 2, 0) / okV.length);
    // The heatmap's blue end is a PATCH floor, not the image-level `lo`. `lo`
    // is a near-maximum over an image's patches, so using it per pixel clipped
    // the picture flat -- see patchFloorWithAlpha. Falls back to `lo` only if
    // no OK image has patch rows, which keeps the old behaviour rather than
    // producing an inverted scale.
    const floors: number[] = [];
    evalResults.forEach((r) => {
      if (r.tier !== "normal") return;
      // cminMap is keyed on tier/label/name, which is NOT evalResults' key --
      // same lookup valOf builds a few lines above.
      floors.push(patchFloorComposite(r, alpha, cminMap.get(`${r.tier}/${r.label}/${r.name}`)));
    });
    floors.sort((a, b) => a - b);
    const heatLo = floors.length ? floors[Math.floor(floors.length / 2)] : lo;
    const tid = setTimeout(() => {
      const values = new Map<string, number>();
      evalResults.forEach((r, key) => values.set(key, evalStats.val(r)));
      onRawAnchors({
        lo, heatLo: Math.min(heatLo, lo), hi: evalStats.thr,
        okMean, okStd, alpha, metric: evalMetric, k: evalTopK, values,
      });
    }, 400);
    return () => clearTimeout(tid);
  }, [evalStats, evalResults, cminMap, alpha, evalMetric, evalTopK, onRawAnchors]);

  // Threshold verdicts by image key, for marking dots on the image-level map.
  const missKeys = useMemo(() => new Set((evalStats?.misses ?? []).map(([k]) => k)), [evalStats]);
  const fpKeys = useMemo(() => new Set((evalStats?.falsePos ?? []).map(([k]) => k)), [evalStats]);

  // Image-dot value under the synced metric (α composite included); falls
  // back to the server-side top-k mean when that image was never swept by
  // the separation check.
  const imagePointVal = useCallback((p: ProjectionPoint): number | null => {
    const r = evalResults.get(`${p.tier}/${p.label}/${p.image}`);
    return r ? valOf(r, evalMetric, evalTopK) : (p.score ?? null);
  }, [evalResults, valOf, evalMetric, evalTopK]);

  // Map dots and eval chips identify images by key — resolve to the taught
  // image and open the preview dialog (heatmap toggles there with "H").
  const openByKey = useCallback((key: string) => {
    const im = images.find((x) => `${x.tier}/${x.label}/${x.name}` === key);
    if (im) onOpenImage(im);
  }, [images, onOpenImage]);

  // Re-fetch whenever the bank membership changes or the user picks a mode.
  // We watch bank patch counts (not the images array) since projection lives
  // in patch space; new appends bump these numbers server-side.
  const bankFingerprint = bank ? `${bank.normal}/${bank.critical}/${bank.negative}` : "";

  const refresh = useCallback(async () => {
    if (!activeBankId) { setProj(null); return; }
    setLoading(true); setErr(null);
    // Patch scores are requested for the patch view: both the score coloring
    // and the below-threshold dimming need them, and having them up front
    // makes the tier/score color toggle a pure re-render (no refetch).
    // guaranteeTop is a fixed 10: it force-includes each evaluated image's
    // hottest rows into the patch sample (a uniform draw would almost never
    // contain the defective patches) and sets the image view's aggregation
    // k — deliberately NOT linked to the check's k slider, so the map never
    // reshapes while tuning the metric.
    try { setProj(await fetchProjection({ mode, withScores: granularity === "patch", granularity, guaranteeTop: 10 })); }
    catch (e) { setErr((e as Error).message); }
    finally { setLoading(false); }
  }, [activeBankId, mode, granularity]);

  useEffect(() => {
    if (!activeBankId) return;
    // Debounced: bank mutations arrive in bursts (multi-file teach).
    const tid = setTimeout(() => { void refresh(); }, 250);
    return () => clearTimeout(tid);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeBankId, mode, granularity, bankFingerprint]);

  // Score → color scale for colorBy="score": robust 5–95 percentile range,
  // grey (in-distribution) → vermilion (anomalous). Vermilion per the
  // colorblind-safe palette used across the app.
  const scoreScale = useMemo(() => {
    if (!proj) return null;
    const vals = proj.points
      .map((p) => (proj.granularity === "image" ? imagePointVal(p) : p.score))
      .filter((s): s is number => s != null)
      .sort((a, b) => a - b);
    if (vals.length < 2) return null;
    const lo = vals[Math.floor(vals.length * 0.05)];
    const hi = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.95))];
    return { lo, hi: hi > lo ? hi : lo + 1e-6 };
  }, [proj, imagePointVal]);

  const scoreColor = (s: number): string => {
    if (!scoreScale) return TIER_COLOR.normal;
    const u = Math.max(0, Math.min(1, (s - scoreScale.lo) / (scoreScale.hi - scoreScale.lo)));
    // grey #8C96AA → vermilion #D55E00
    const r = Math.round(140 + (213 - 140) * u);
    const g = Math.round(150 + (94 - 150) * u);
    const b = Math.round(170 + (0 - 170) * u);
    return `rgba(${r}, ${g}, ${b}, ${0.45 + 0.45 * u})`;
  };

  const box = useMemo(() => {
    if (!proj || !proj.points.length) return null;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const p of proj.points) {
      if (p.x < xmin) xmin = p.x; if (p.x > xmax) xmax = p.x;
      if (p.y < ymin) ymin = p.y; if (p.y > ymax) ymax = p.y;
    }
    const pad = 0.06;
    const w = (xmax - xmin) || 1;
    const h = (ymax - ymin) || 1;
    // Equal units per pixel on both axes (distances stay comparable), with
    // the data range extended to the plot's real aspect ratio — the plot
    // fills the card width, so the extra room goes into the range instead of
    // stretching the geometry.
    const side = Math.max(w, h);
    const ar = plotAspect > 0 ? plotAspect : 1;
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
    const halfY = (side / 2) * (1 + pad) * Math.max(1 / ar, 1);
    const halfX = (side / 2) * (1 + pad) * Math.max(ar, 1);
    return { xmin: cx - halfX, xmax: cx + halfX, ymin: cy - halfY, ymax: cy + halfY };
  }, [proj, plotAspect]);

  // viewBox width tracks the aspect so 1 viewBox unit is the same number of
  // pixels horizontally and vertically (circles stay circles).
  const vw = 100 * (plotAspect > 0 ? plotAspect : 1);

  const project = (p: ProjectionPoint) => {
    if (!box) return { cx: 0, cy: 0 };
    const w = box.xmax - box.xmin, h = box.ymax - box.ymin;
    return {
      cx: ((p.x - box.xmin) / w) * vw,
      cy: (1 - (p.y - box.ymin) / h) * 100,
    };
  };

  return (
    // One card, not a band of reference figures beside two. What the band
    // really held was the run button and the view toggle: the picker was a
    // dropdown over a list of length one, and every figure in it -- taught
    // counts, feature dim, bank name -- is printed by the bank tab, which is
    // where those counts are actually established. Dissolving it hands the
    // figure the 358px the band was holding.
    <div className="anom-projection" style={{ flex: "none", minWidth: 0, boxSizing: "border-box", border, borderRadius: 10, padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8 }}>
      {/* Header, stacked: no title (the toggle that picks the view names it),
          and the run button on its own line because the toggle plus the
          button does not fit on one at the rail's width. */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8, flex: "none", minWidth: 0 }}>
        <span style={{ display: "flex", minWidth: 0 }}>
          <button
            onClick={() => setChartView("eval")}
            aria-pressed={chartView === "eval"}
            style={{
              flex: 1, minWidth: 0, padding: "3px 8px", fontSize: TYPE.base, border, borderRadius: "6px 0 0 6px", cursor: "pointer", whiteSpace: "nowrap",
              background: chartView === "eval" ? ACCENT : "transparent",
              color: chartView === "eval" ? "#fff" : INK,
            }}
          >{t("develop.eval.title")}</button>
          <button
            onClick={() => setChartView("map")}
            aria-pressed={chartView === "map"}
            data-tutorial-step="develop-map-toggle"
            style={{
              flex: 1, minWidth: 0, padding: "3px 8px", fontSize: TYPE.base, border, borderLeft: "none", borderRadius: "0 6px 6px 0", cursor: "pointer", whiteSpace: "nowrap",
              background: chartView === "map" ? ACCENT : "transparent",
              color: chartView === "map" ? "#fff" : INK,
            }}
          >{t("develop.projection.scatterTitle")}</button>
        </span>
        {/* The one heavy action: sweeps the check, re-projects the map and
            pre-renders heatmaps. Green (Okabe–Ito) = "go", distinct from the
            teal toggles above it. The hint the band used to print in full is
            its tooltip: at 316px it was three lines of text for a sentence
            nobody reads twice. */}
        <button
          onClick={onRunAll}
          disabled={runAllDisabled}
          data-tutorial-step="develop-run-eval"
          title={t("develop.eval.hint")}
          style={{
            width: "100%", padding: "5px 16px", fontSize: TYPE.md, fontWeight: 600,
            borderRadius: 8, border: "none", whiteSpace: "nowrap",
            background: runAllDisabled ? "rgba(0,158,115,.35)" : "#009E73",
            color: "#fff", cursor: runAllDisabled ? "default" : "pointer",
          }}
        >▶ {t("develop.eval.run")}</button>
        {chartView === "map" && proj && proj.mode !== "empty" && (
          <span className="muted" style={{ minWidth: 0, fontSize: TYPE.base, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontVariantNumeric: "tabular-nums" }}>
            {t("develop.projection.sampled")}: <b style={{ color: INK }}>{proj.points.length.toLocaleString()}</b>
            {" / "}
            {(Object.values(proj.total).reduce((a, b) => a + b, 0)).toLocaleString()}
            {proj.granularity === "image" ? ` ${t("develop.projection.imgUnit")}` : " p"}
          </span>
        )}
      </div>

      {/* One figure at a time. Side by side they had half the width each and
          neither was wide enough -- and they answer the same question at
          different resolutions, so the second one was never worth the width
          it cost the first. */}
      <div style={{ minWidth: 0 }}>
        {chartView === "eval" && (
        <>
        <SeparationEval
          activeBankId={activeBankId}
          images={images} onPick={openByKey} border={border} showToast={showToast}
          results={evalResults} setResults={setEvalResults}
          metric={evalMetric} setMetric={setEvalMetric}
          topK={evalTopK} setTopK={setEvalTopK}
          alpha={alpha} setAlpha={setAlpha} cminCount={cminMap.size}
          onFindBestAlpha={(m, k) => void findBestAlpha(m, k)}
          thrOverride={thrOverride} setThrOverride={setThrOverride}
          stats={evalStats} valOf={valOf}
          onEvalDone={() => void refresh()}
          runRef={evalRunRef}
        />

        </>
        )}
        {/* Feature-separation map — the second opinion, behind the toggle. */}
        {chartView === "map" && (
          <div data-tutorial-step="develop-map" style={{ width: "100%", height: MAP_FIG_H, boxSizing: "border-box", overflow: "hidden", display: "flex", flexDirection: "column", gap: 8 }}>
            {/* The plot takes the whole width; its controls sit under it. As a
                96px rail down the right they were spending a third of a 314px
                column on buttons, and the plot is the thing being read. The
                data range, not the geometry, absorbs the extra width, so both
                axes stay on the same scale. */}
            <div ref={plotRef} style={{
              flex: 1, minWidth: 0, minHeight: 0,
              background: "rgba(255,255,255,.02)", borderRadius: 8,
              position: "relative", overflow: "hidden", border,
            }}>
              {loading && (
                <div className="muted" style={{ position: "absolute", top: 6, right: 8, fontSize: TYPE.sm, pointerEvents: "none" }}>
                  {t("develop.projection.loading")}
                </div>
              )}
              {err && (
                <div className="muted" style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: TYPE.base, padding: 12, textAlign: "center" }}>{err}</div>
              )}
              {!err && proj?.mode === "empty" && (
                <div className="muted" style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: TYPE.base, padding: 12, textAlign: "center" }}>
                  {t("develop.projection.empty")}
                </div>
              )}
              {!err && proj && proj.mode !== "empty" && box && (
                <svg viewBox={`0 0 ${vw} 100`} preserveAspectRatio="xMidYMid meet" style={{ width: "100%", height: "100%", display: "block" }}>
                  {proj.points.map((p, i) => {
                    const c = project(p);
                    const isImg = proj.granularity === "image";
                    const key = `${p.tier}/${p.label}/${p.image}`;
                    const v = isImg ? imagePointVal(p) : p.score;
                    // Image dots follow the separation check: same metric for
                    // the score color, blue fill for its false positives, and
                    // a ring (shape, not just color) on both FP and misses.
                    const isFP = isImg && fpKeys.has(key);
                    const isMiss = isImg && missKeys.has(key);
                    const fill = colorBy === "score" && v != null
                      ? scoreColor(v)
                      : isFP ? EVAL_FP_COLOR : TIER_COLOR[p.tier];
                    // Patches scoring below the raw-metric threshold (an NG
                    // image is >99% normal paper) fade out so only the
                    // patches that drive the image verdicts stay prominent.
                    // rawThr, not evalStats.thr: patch dots have no cmin, so
                    // the α-composite threshold would dim everything once
                    // α > 0. No threshold yet → nothing is dimmed.
                    const dim = !isImg && rawThr != null && (v == null || v < rawThr);
                    // Transparent stroke widens the hit target well beyond the
                    // dot so individual points are clickable. Image-level dots
                    // are few (one per taught image), so draw them larger.
                    const r = isImg ? 1.4 : dim ? 0.45 : 0.75;
                    const mark = isFP || isMiss;
                    return (
                      <circle
                        key={i} cx={c.cx} cy={c.cy} r={r} fill={fill}
                        opacity={dim ? 0.15 : 1}
                        stroke={mark ? INK : "transparent"}
                        strokeWidth={mark ? 0.35 : 1.4}
                        strokeDasharray={isMiss ? "0.7 0.5" : undefined}
                        style={p.image ? { cursor: "pointer" } : undefined}
                        onClick={p.image ? () => openByKey(key) : undefined}
                      >
                        {p.image && (
                          <title>
                            {p.image}{v != null ? ` · ${v.toFixed(2)}` : ""}
                            {isFP ? ` · ${t("develop.projection.fpMark")}` : ""}
                            {isMiss ? ` · ${t("develop.projection.missMark")}` : ""}
                          </title>
                        )}
                      </circle>
                    );
                  })}
                </svg>
              )}
              {/* Threshold-verdict legend — the ring marks only exist on the
                  image view and only once a threshold is known. */}
              {!err && proj?.granularity === "image" && evalStats && (
                <div
                  className="muted"
                  style={{
                    position: "absolute", left: 8, bottom: 6, display: "flex", gap: 12, alignItems: "center",
                    fontSize: TYPE.sm, pointerEvents: "none", padding: "2px 8px", borderRadius: 6,
                    background: "rgba(127,127,127,.10)",
                  }}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                    <svg width={14} height={14} viewBox="0 0 14 14">
                      <circle cx={7} cy={7} r={4.5} fill={EVAL_FP_COLOR} stroke={INK} strokeWidth={1.2} />
                    </svg>
                    {t("develop.projection.fpMark")} (OK)
                  </span>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                    <svg width={14} height={14} viewBox="0 0 14 14">
                      <circle cx={7} cy={7} r={4.5} fill={TIER_COLOR.critical} stroke={INK} strokeWidth={1.2} strokeDasharray="2.4 1.8" />
                    </svg>
                    {t("develop.projection.missMark")} (NG)
                  </span>
                </div>
              )}
            </div>
            {/* Grouped, so a wrap breaks BETWEEN granularity, axis mode and
                dot colour rather than through one of them. The dividers that
                used to separate them are the wider gap now: a vertical rule
                does not survive a line break. */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 12px", flex: "none", minWidth: 0 }}>
              <span style={{ display: "inline-flex", gap: 4 }}>
                {(["image", "patch"] as const).map((g) => (
                  <button
                    key={g}
                    onClick={() => setGranularity(g)}
                    aria-pressed={granularity === g}
                    style={{
                      flex: "none",
                      padding: "2px 8px", fontSize: TYPE.base, borderRadius: 6, border, cursor: "pointer",
                      background: granularity === g ? ACCENT : "transparent",
                      color: granularity === g ? "#fff" : INK,
                    }}
                  >{t(`develop.projection.gran.${g}` as "develop.projection.gran.patch")}</button>
                ))}
              </span>
              <span style={{ display: "inline-flex", gap: 4 }}>
                {(["auto", "normal", "anomaly"] as ProjectionMode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    aria-pressed={mode === m}
                    style={{
                      flex: "none",
                      padding: "2px 8px", fontSize: TYPE.base, borderRadius: 6, border, cursor: "pointer",
                      background: mode === m ? ACCENT : "transparent",
                      color: mode === m ? "#fff" : INK,
                    }}
                  >{t(`develop.projection.mode.${m}` as "develop.projection.mode.auto")}</button>
                ))}
              </span>
              <span style={{ display: "inline-flex", gap: 4 }}>
                {(["tier", "score"] as const).map((c) => (
                  <button
                    key={c}
                    onClick={() => setColorBy(c)}
                    aria-pressed={colorBy === c}
                    style={{
                      flex: "none",
                      padding: "2px 8px", fontSize: TYPE.base, borderRadius: 6, border, cursor: "pointer",
                      background: colorBy === c ? ACCENT : "transparent",
                      color: colorBy === c ? "#fff" : INK,
                    }}
                  >{t(`develop.projection.color.${c}` as "develop.projection.color.tier")}</button>
                ))}
              </span>
            </div>
            {/* Footer: only the score colour scale, and only in score mode —
                the old cPC/threshold caption read as noise even to its users. */}
            {colorBy === "score" && scoreScale && (
              <div className="muted" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6, fontSize: TYPE.base, fontVariantNumeric: "tabular-nums", minWidth: 0 }}>
                <span>{scoreScale.lo.toFixed(1)}</span>
                <span style={{ width: 72, height: 8, borderRadius: 4, background: "linear-gradient(90deg, rgb(140,150,170), rgb(213,94,0))" }} />
                <span>{scoreScale.hi.toFixed(1)}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

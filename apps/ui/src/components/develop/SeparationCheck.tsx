// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../../i18n";
import {
  ALPHA_MAX,
  evaluateBankImage,
  fetchBankImages,
  fetchCachedEvaluations,
  fetchGroupPreview,
  type BankImage,
  type GroupMode,
  type GroupPreview,
  type Grouping,
  type StoredImageEval,
  type Tier,
} from "../../api/cls";
import { isAbortError } from "../../api/shared";
import { ACCENT, BORDER, INK, TYPE, VERM} from "../../ui/tokens";
import BusyDialog from "./BusyDialog";
import { EVAL_FP_COLOR, EVAL_NG_COLOR, EVAL_OK_COLOR, aurocOf } from "./separation";
import { EvalStats } from "./separation";

export default function SeparationEval({
  activeBankId,
  images,
  onPick,
  border,
  showToast,
  results,
  setResults,
  metric,
  setMetric,
  topK,
  setTopK,
  alpha,
  setAlpha,
  cminCount,
  onFindBestAlpha,
  thrOverride,
  setThrOverride,
  stats,
  valOf,
  onEvalDone,
  runRef,
}: {
  activeBankId: string;
  images: BankImage[];
  onPick: (key: string) => void;
  border: string;
  showToast: (msg: string) => void;
  results: Map<string, StoredImageEval>;
  setResults: React.Dispatch<React.SetStateAction<Map<string, StoredImageEval>>>;
  metric: "p99" | "topk";
  setMetric: (m: "p99" | "topk") => void;
  topK: number;
  setTopK: (k: number) => void;
  alpha: number;
  setAlpha: (a: number) => void;
  cminCount: number;
  onFindBestAlpha: (metric?: "p99" | "topk", k?: number) => void;
  thrOverride: number | null;
  setThrOverride: (v: number | null) => void;
  stats: EvalStats;
  onEvalDone: () => void;
  runRef: React.MutableRefObject<(() => Promise<boolean>) | null>;
  valOf: (r: StoredImageEval, kind: "p99" | "topk", k: number) => number;
}) {
  const { t } = useI18n();
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const cancelRef = useRef(false);
  const sweepAbort = useRef<AbortController | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const requestSweepCancel = useCallback(() => {
    cancelRef.current = true;
    setCancelling(true);
    sweepAbort.current?.abort();
  }, []);

  // Sweep k=1..256 for the top-k mean with the best AUROC. Pure compute —
  // the manual button and the post-sweep auto-tune share it.
  const computeBestK = useCallback((): { k: number; auroc: number } | null => {
    const entries = [...results.values()];
    const ok = entries.filter((r) => r.tier === "normal");
    const ng = entries.filter((r) => r.tier === "critical");
    if (!ok.length || !ng.length) return null;
    let bestK = 1, bestA = -1;
    for (let k = 1; k <= 256; k++) {
      const a = aurocOf(ng.map((r) => valOf(r, "topk", k)), ok.map((r) => valOf(r, "topk", k)));
      if (a > bestA) { bestA = a; bestK = k; }
    }
    return { k: bestK, auroc: bestA };
  }, [results, valOf]);

  const findBestK = useCallback(() => {
    const best = computeBestK();
    if (!best) return;
    setMetric("topk"); setTopK(best.k);
    showToast(t("develop.eval.bestKFound").replace("{k}", String(best.k)).replace("{auroc}", best.auroc.toFixed(4)));
  }, [computeBestK, setMetric, setTopK, showToast, t]);

  const keyOf = (im: { tier: Tier; label: string; name: string }) => `${im.tier}/${im.label}/${im.name}`;

  // How the sweep holds images out. This was a numbered step on the bank tab,
  // where it previewed a rule and then never reached the evaluation: the
  // client did not send the parameters, so every run was leave-one-image-out
  // no matter what was selected here -- exactly the optimistic reading the
  // control exists to avoid. It belongs beside the sweep it governs.
  const [grouping, setGrouping] = useState<Grouping>({ mode: "none", sep: "_", fields: 1 });
  const [groupPreview, setGroupPreview] = useState<GroupPreview | null>(null);
  useEffect(() => {
    if (grouping.mode === "none") { setGroupPreview(null); return; }
    let cancelled = false;
    fetchGroupPreview(grouping.mode, grouping.sep, grouping.fields)
      .then((p) => { if (!cancelled) setGroupPreview(p); })
      .catch(() => { if (!cancelled) setGroupPreview(null); });
    return () => { cancelled = true; };
  }, [grouping, images]);

  // The server caches finished evals per bank state (memory + eval_cache.json),
  // so a page reload restores the last run instantly; a mutated bank returns []
  // and the stale display clears itself. Gated on activeBankId: until the
  // bank select round-trip finishes, the server would answer for whatever
  // bank was active before (e.g. the previous project's).
  useEffect(() => {
    if (running || !activeBankId) return;
    // A grouped run is deliberately never written to that cache, so restoring
    // from it would show leave-one-image-out numbers under a grouped heading.
    if (grouping.mode !== "none") { setResults(new Map()); return; }
    let cancelled = false;
    fetchCachedEvaluations()
      .then((rows) => {
        if (cancelled) return;
        const m = new Map<string, StoredImageEval>();
        rows.forEach((r) => m.set(`${r.tier}/${r.label}/${r.name}`, r));
        setResults(m);
      })
      .catch(() => { /* no active bank yet — nothing to restore */ });
    return () => { cancelled = true; };
  }, [images, running, activeBankId, grouping.mode]);

  // Sequential sweep: one request per image so progress is visible, the GPU
  // does one cdist batch at a time, and cancel takes effect between images.
  const [runTotal, setRunTotal] = useState(0);
  const autoTuneArm = useRef(false);
  const run = useCallback(async (): Promise<boolean> => {
    cancelRef.current = false;
    setCancelling(false);
    setRunning(true); setResults(new Map()); setProgress(0);
    // Fetch the list fresh: the bank tab can assemble a different set of
    // images while this tab sits open, and the prop may not have caught up.
    let list = images;
    try { list = (await fetchBankImages()).images; } catch { /* prop fallback */ }
    setRunTotal(list.length);
    for (let i = 0; i < list.length; i++) {
      if (cancelRef.current) break;
      const im = list[i];
      const ac = new AbortController();
      sweepAbort.current = ac;
      try {
        const r = await evaluateBankImage(im.tier, im.name, im.label, grouping, ac.signal);
        setResults((prev) => new Map(prev).set(keyOf(im), r));
      } catch (e) {
        // An abort is the cancel arriving, not an evaluate failure: no toast.
        if (isAbortError(e) || cancelRef.current) break;
        showToast(`evaluate failed (${im.name}): ${(e as Error).message}`);
        // 409 = active bank/project changed under the sweep — every later
        // image fails identically; abort instead of toasting per image.
        if ((e as { status?: number }).status === 409) {
          cancelRef.current = true;
        }
      } finally {
        sweepAbort.current = null;
      }
      setProgress(i + 1);
    }
    sweepAbort.current = null;
    const cancelled = cancelRef.current;
    // Arm the post-sweep auto-tune (k then α). Done via an effect, not a
    // direct call: this closure's computeBestK still sees the pre-sweep
    // (empty) results map. A cancelled sweep must not arm it: auto-tune is
    // seconds of synchronous main-thread work, and running it right after the
    // click is what made the *next* dialog's 中止 unresponsive too.
    autoTuneArm.current = !cancelled;
    setRunning(false);
    setCancelling(false);
    // Re-project the map: the patch sample's guarantee-top rows come from
    // the eval cache this sweep just (re)built.
    onEvalDone();
    return cancelled;
  }, [images, showToast, onEvalDone, grouping]);

  // Auto-tune after a completed sweep: jump to the best k, then sweep α for
  // that exact k (passed as override — the setMetric/setTopK updates land in
  // the same commit, after the α sweep already started). Manual hover-slider
  // edits are never overridden outside this armed, post-sweep moment.
  useEffect(() => {
    if (running || !autoTuneArm.current) return;
    autoTuneArm.current = false;
    const best = computeBestK();
    if (!best) return;
    setMetric("topk"); setTopK(best.k);
    onFindBestAlpha("topk", best.k);
  }, [running, computeBestK, setMetric, setTopK, onFindBestAlpha]);

  // Exposed so the run-all button on the bank card can trigger the sweep.
  useEffect(() => {
    runRef.current = run;
    return () => { runRef.current = null; };
  }, [run, runRef]);

  // Butterfly histogram: OK bars up (grey, FP blue overlaid), NG bars down.
  const histo = useMemo(() => {
    if (!stats) return null;
    const allV = [...stats.okV, ...stats.ngV, ...stats.fpV];
    const lo = Math.min(...allV), hi = Math.max(...allV);
    const span = hi > lo ? hi - lo : 1;
    const NBINS = 48, W = 640, H = 160, MID = 80, MAXH = 70;
    const binOf = (v: number) => Math.min(NBINS - 1, Math.max(0, Math.floor(((v - lo) / span) * NBINS)));
    const okB = new Array<number>(NBINS).fill(0);
    const ngB = new Array<number>(NBINS).fill(0);
    const fpB = new Array<number>(NBINS).fill(0);
    stats.okV.forEach((v) => okB[binOf(v)]++);
    stats.ngV.forEach((v) => ngB[binOf(v)]++);
    stats.fpV.forEach((v) => fpB[binOf(v)]++);
    const maxC = Math.max(...okB, ...ngB, ...fpB, 1);
    const x = (v: number) => ((v - lo) / span) * W;
    return { lo, hi, NBINS, W, H, MID, MAXH, okB, ngB, fpB, maxC, x, binW: W / NBINS };
  }, [stats]);

  const chip = (key: string, name: string, v: number) => (
    <button
      key={key}
      onClick={() => onPick(key)}
      title={`${name} · ${v.toFixed(2)}`}
      style={{ padding: "2px 10px", borderRadius: 999, border, fontSize: TYPE.base, background: "transparent", color: INK, cursor: "pointer", maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
    >{name} <b>{v.toFixed(2)}</b></button>
  );

  return (
    // A column in the rail, not a row across a card: the histogram first,
    // because it is the answer, and the knobs that shape it underneath. The
    // card sizes to this; the rail scrolls if the window is short.
    <div data-tutorial-step="develop-separation" style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Blocking modal while the sweep scores the bank image by image —
          same rationale as the heatmap pre-render: any other action taken
          meanwhile (teach, delete, bank switch) would race the sweep and
          the GPU, so the whole app waits. Cancel stays available. */}
      {/* What you read. */}
      <div style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
      {stats && histo && (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 16, fontSize: TYPE.md, fontVariantNumeric: "tabular-nums", flexWrap: "wrap" }}>
            <span>AUROC <b style={{ fontSize: TYPE.xl }}>{Number.isNaN(stats.auroc) ? "–" : stats.auroc.toFixed(4)}</b></span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6, marginLeft: "auto" }} className="muted">
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: EVAL_OK_COLOR }} /> OK
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: EVAL_FP_COLOR, marginLeft: 6 }} /> FP
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: EVAL_NG_COLOR, marginLeft: 6 }} /> NG
            </span>
          </div>
          {/* flex: none — the card is height-pinned with internal scroll, and
              a shrinkable svg is the one child flexbox can squash to zero. */}
          <svg viewBox={`0 0 ${histo.W} ${histo.H}`} preserveAspectRatio="none" style={{ width: "100%", height: 128, flex: "none", display: "block" }}>
            <line x1={0} y1={histo.MID} x2={histo.W} y2={histo.MID} stroke={BORDER} strokeWidth={1} />
            {histo.okB.map((c, i) => c > 0 && (
              <rect key={`ok${i}`} x={i * histo.binW + 0.5} width={histo.binW - 1}
                y={histo.MID - (c / histo.maxC) * histo.MAXH} height={(c / histo.maxC) * histo.MAXH}
                fill={EVAL_OK_COLOR} />
            ))}
            {histo.fpB.map((c, i) => c > 0 && (
              <rect key={`fp${i}`} x={i * histo.binW + 0.5} width={histo.binW - 1}
                y={histo.MID - (c / histo.maxC) * histo.MAXH} height={(c / histo.maxC) * histo.MAXH}
                fill={EVAL_FP_COLOR} opacity={0.7} />
            ))}
            {histo.ngB.map((c, i) => c > 0 && (
              <rect key={`ng${i}`} x={i * histo.binW + 0.5} width={histo.binW - 1}
                y={histo.MID} height={(c / histo.maxC) * histo.MAXH}
                fill={EVAL_NG_COLOR} />
            ))}
            <line x1={histo.x(stats.thr)} y1={4} x2={histo.x(stats.thr)} y2={histo.H - 4}
              stroke={INK} strokeWidth={1.5} strokeDasharray="5 4" />
            <text x={4} y={14} fontSize={11} fill="currentColor" opacity={0.6}>OK ↑</text>
            <text x={4} y={histo.H - 6} fontSize={11} fill="currentColor" opacity={0.6}>NG ↓</text>
          </svg>
          {stats.misses.length === 0 && stats.falsePos.length === 0 ? (
            <div style={{ fontSize: TYPE.md }}>{t("develop.eval.separated")}</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {stats.misses.length > 0 && (
                <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  <span className="muted" style={{ fontSize: TYPE.base, flex: "none" }}>{t("develop.eval.misses")}:</span>
                  {stats.misses.map(([k, r]) => chip(k, r.name, stats.val(r)))}
                </div>
              )}
              {stats.falsePos.length > 0 && (
                <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  <span className="muted" style={{ fontSize: TYPE.base, flex: "none" }}>{t("develop.eval.falsePos")}:</span>
                  {stats.falsePos.map(([k, r]) => chip(k, r.name, stats.val(r)))}
                </div>
              )}
            </div>
          )}
        </>
      )}
      {!stats && results.size > 0 && (
        <div className="muted" style={{ fontSize: TYPE.base }}>{t("develop.eval.empty")}</div>
      )}
      </div>
      {/* Everything you set, under the thing it changes. */}
      <div style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 6 }}>
      {running && (
        <BusyDialog
          border={border}
          title={t("develop.eval.progress").replace("{n}", String(progress)).replace("{total}", String(runTotal))}
          hint={t("develop.busyHint")}
          onCancel={requestSweepCancel}
          cancelLabel={cancelling ? t("develop.eval.cancelling") : t("develop.eval.cancel")}
          cancelDisabled={cancelling}
        />
      )}
      {/* Progress only, and only while a sweep runs. The title and the hint
          moved to the card header, and the 中止 that used to sit here
          rendered under the same `running` condition as the BusyDialog above,
          whose full-screen scrim is painted on top of it — so the click never
          reached the handler and the button was dead 100% of the time it was
          visible. The dialog's own cancel is the reachable one (2026-07-31). */}
      {running && (
        <span className="muted" style={{ fontSize: TYPE.base, fontVariantNumeric: "tabular-nums", flex: "none" }}>
          {t("develop.eval.progress").replace("{n}", String(progress)).replace("{total}", String(runTotal))}
        </span>
      )}

      {/* Rows 2-4: aligned label/control grid — metric+k, α boost, threshold.
          One control per row: nothing wraps, sliders share the same column. */}
      {stats && (
        <div style={{ display: "flex", flexDirection: "column", gap: 3, fontSize: TYPE.base, minWidth: 0 }}>
          {/* Labels over controls, not beside them. The two-column grid this
              replaces was laid out when the card was three times as wide; in
              a rail the label column squeezed the controls into a width they
              wrapped out of, which is what left the rows ragged.
              The comment lives INSIDE the div on purpose: one line up it is a
              second child of `{stats && (...)}`, which takes exactly one. */}
          <span className="muted" style={{ whiteSpace: "nowrap" }} title={t("develop.eval.autoTunedNote")}>{t("develop.eval.rowMetric")}</span>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", minWidth: 0 }}>
            {/* Joined segmented control: p99 | top-k平均 (k=N). k lives inside
                its segment — hover (or focus) the segment to slide/re-search.
                Both k and α are auto-tuned after every sweep (note at right). */}
            <span style={{ display: "inline-flex", alignItems: "stretch", flex: "none" }}>
              <button
                onClick={() => setMetric("p99")}
                aria-pressed={metric === "p99"}
                style={{
                  padding: "3px 10px", fontSize: TYPE.base, border, borderRadius: "6px 0 0 6px", cursor: "pointer", whiteSpace: "nowrap",
                  background: metric === "p99" ? ACCENT : "transparent",
                  color: metric === "p99" ? "#fff" : INK,
                }}
              >p99</button>
              <span className="hover-reveal" tabIndex={0}>
                <button
                  onClick={() => setMetric("topk")}
                  aria-pressed={metric === "topk"}
                  style={{
                    padding: "3px 10px", fontSize: TYPE.base, border, borderLeft: "none", borderRadius: "0 6px 6px 0", cursor: "pointer", whiteSpace: "nowrap",
                    background: metric === "topk" ? ACCENT : "transparent",
                    color: metric === "topk" ? "#fff" : INK,
                    display: "inline-flex", alignItems: "baseline", gap: 6,
                  }}
                >
                  {t("develop.eval.metric.topk")}
                  {metric === "topk" && (
                    <span style={{ fontSize: TYPE.sm, opacity: 0.85, fontVariantNumeric: "tabular-nums", borderBottom: "1px dashed currentColor" }}>k={topK}</span>
                  )}
                </button>
                {metric === "topk" && (
                  <span className="hover-pop">
                    <input
                      type="range" min={1} max={256} value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                      aria-label="top-k"
                      style={{ width: 160 }}
                    />
                    <button
                      onClick={findBestK}
                      style={{ padding: "2px 10px", fontSize: TYPE.base, borderRadius: 6, border, background: "transparent", color: INK, cursor: "pointer", flex: "none", whiteSpace: "nowrap" }}
                    >{t("develop.eval.findBestK")}</button>
                  </span>
                )}
              </span>
            </span>
            {/* Thin divider so metric and α read as two groups, not one blob. */}
            <span aria-hidden style={{ width: 1, alignSelf: "stretch", background: BORDER, flex: "none" }} />
            <span data-tutorial-step="develop-alpha" style={{ display: "inline-flex", alignItems: "center", gap: 6, flex: "none" }}>
              <span className="muted" style={{ whiteSpace: "nowrap" }} title={t("develop.eval.alphaHint")}>{t("develop.eval.alpha")}</span>
              <span className="hover-reveal" tabIndex={0}>
                <span className="hover-reveal-value" style={{ fontVariantNumeric: "tabular-nums", fontWeight: 600 }} title={t("develop.eval.alphaHint")}>{alpha}</span>
                <span className="hover-pop">
                  <input
                    type="range" min={0} max={ALPHA_MAX} step={5} value={alpha}
                    onChange={(e) => setAlpha(Number(e.target.value))}
                    aria-label={t("develop.eval.alpha")}
                    style={{ width: 160 }}
                  />
                  <button
                    onClick={() => onFindBestAlpha()}
                    style={{ padding: "2px 10px", fontSize: TYPE.base, borderRadius: 6, border, background: "transparent", color: INK, cursor: "pointer", flex: "none", whiteSpace: "nowrap" }}
                  >{t("develop.eval.findBestAlpha")}</button>
                </span>
              </span>
            </span>
          </div>

          {histo && (
            <>
              <span className="muted" style={{ whiteSpace: "nowrap", color: thrOverride != null ? VERM : undefined }}>
                {thrOverride == null ? t("develop.eval.threshold") : t("develop.eval.thresholdManual")}
              </span>
              <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
                <input
                  type="range"
                  min={histo.lo} max={histo.hi}
                  step={(histo.hi - histo.lo) / 400 || 0.01}
                  value={stats.thr}
                  onChange={(e) => setThrOverride(Number(e.target.value))}
                  aria-label={t("develop.eval.thresholdManual")}
                  style={{ flex: "1 1 80px", minWidth: 80, maxWidth: 220 }}
                />
                <b style={{ color: thrOverride == null ? INK : VERM, minWidth: 52, fontSize: TYPE.md, fontVariantNumeric: "tabular-nums" }}>{stats.thr.toFixed(2)}</b>
                {thrOverride != null && (
                  <button
                    className="btn-ghost"
                    onClick={() => setThrOverride(null)}
                    title={t("develop.eval.thresholdReset").replace("{thr}", stats.autoThr.toFixed(2))}
                  >↺</button>
                )}
              </div>
            </>
          )}
        </div>
      )}


      {/* How the sweep holds images out. Last of the settings because it is
          the one that changes what the numbers mean rather than how they are
          read: a grouped run is a different measurement, not a different view
          of the same one. */}
      <div style={{ display: "flex", flexDirection: "column", gap: 5, borderTop: border, paddingTop: 6 }}>
        <b style={{ fontSize: TYPE.base }}>{t("panel.check")}</b>
        <span className="muted" style={{ fontSize: TYPE.sm }}>{t("panel.checkHint")}</span>
        <select
          value={grouping.mode}
          onChange={(e) => setGrouping((g) => ({ ...g, mode: e.target.value as GroupMode }))}
          title={t("label.groupHint")}
          style={{ padding: "3px 6px", borderRadius: 6, border, background: "transparent", color: INK, fontSize: TYPE.base }}
        >
          <option value="none">{t("label.group.none")}</option>
          <option value="datetime">{t("label.group.datetime")}</option>
          <option value="prefix">{t("label.group.prefix")}</option>
          <option value="manual">{t("label.group.manual")}</option>
        </select>
        {grouping.mode === "prefix" && (
          <div style={{ display: "flex", gap: 5 }}>
            <input
              value={grouping.sep}
              onChange={(e) => setGrouping((g) => ({ ...g, sep: e.target.value }))}
              title={t("label.group.sep")}
              style={{ width: 40, padding: "3px 6px", borderRadius: 6, border, background: "transparent", color: INK, fontSize: TYPE.base }} />
            <input
              type="number" min={1} max={9} value={grouping.fields}
              onChange={(e) => setGrouping((g) => ({ ...g, fields: Math.max(1, Number(e.target.value) || 1) }))}
              title={t("label.group.fields")}
              style={{ width: 56, padding: "3px 6px", borderRadius: 6, border, background: "transparent", color: INK, fontSize: TYPE.base }} />
          </div>
        )}
        {groupPreview && (
          <div style={{ fontSize: TYPE.sm }}>
            <div className="muted">
              {t("label.group.result").replace("{g}", String(Object.keys(groupPreview.groups).filter((k) => k).length))}
            </div>
            {groupPreview.ungrouped > 0 && (
              // The number that says whether the naming convention was guessed
              // right: a rule that places nothing degrades silently into
              // leave-one-out under a different name.
              <div style={{ color: VERM, fontWeight: 600 }}>
                {t("label.group.ungrouped").replace("{n}", groupPreview.ungrouped.toLocaleString())}
              </div>
            )}
          </div>
        )}
      </div>

      {stats && alpha > 0 && cminCount === 0 && (
        <div className="muted" style={{ fontSize: TYPE.base }}>{t("develop.eval.alphaNoCmin")}</div>
      )}
      </div>
    </div>
  );
}

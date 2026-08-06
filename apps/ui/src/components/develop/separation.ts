// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// The separation check's arithmetic, with no view attached: the alpha-composite
// statistic, Youden's threshold and AUROC. The feature map needs the same
// numbers to colour its points, so they cannot live inside the panel that
// displays them.
import { BOOST_FLOOR, type StoredImageEval } from "../../api/cls";

// ---------- Separation check: OK vs NG image-score histogram + AUROC ----------

// Image-level statistic on the α-composite scale: raw + α/(ε+cmin) over the
// cached top rows, falling back to the raw statistic when α is 0 or the
// image has no cmin rows (e.g. it was never swept, or it is the only taught
// NG — its own rows are excluded from the exemplars). Single source for the
// live valOf and the α sweep, which must evaluate α values other than the
// current slider position.
export function statWithAlpha(
  r: StoredImageEval, kind: "p99" | "topk", k: number,
  alphaVal: number, cm: number[] | undefined,
): number {
  const ts = r.top_scores ?? [];
  if (alphaVal > 0 && ts.length && cm && cm.length === ts.length) {
    const comp = ts.map((v, i) => v + alphaVal / (BOOST_FLOOR + cm[i])).sort((a, b) => b - a);
    if (kind === "p99") return comp[Math.min(comp.length - 1, Math.max(0, Math.round(r.patches * 0.01)))];
    const kk = Math.min(k, comp.length);
    let s = 0;
    for (let i = 0; i < kk; i++) s += comp[i];
    return s / kk;
  }
  if (kind === "p99") return r.score_p99;
  if (!ts.length) return r.score_max;
  const kk = Math.min(k, ts.length);
  let s = 0;
  for (let i = 0; i < kk; i++) s += ts[i];
  return s / kk;
}

// Youden's J-maximising midpoint between two score sets (NG treated as
// positive, i.e. "above threshold" flags NG).
export function youdenThreshold(okV: number[], ngV: number[]): number {
  const cand = [...okV, ...ngV].sort((a, b) => a - b);
  let thr = cand[0], bestJ = -Infinity;
  for (let c = 0; c < cand.length - 1; c++) {
    const m = (cand[c] + cand[c + 1]) / 2;
    const tpr = ngV.filter((v) => v > m).length / ngV.length;
    const fpr = okV.filter((v) => v > m).length / okV.length;
    if (tpr - fpr > bestJ) { bestJ = tpr - fpr; thr = m; }
  }
  return thr;
}

// Rank-based AUROC (Mann-Whitney U) with average ranks for ties.
export function aurocOf(pos: number[], neg: number[]): number {
  if (!pos.length || !neg.length) return NaN;
  const all = [...pos.map((v) => ({ v, p: 1 })), ...neg.map((v) => ({ v, p: 0 }))].sort((a, b) => a.v - b.v);
  const ranks = new Array<number>(all.length).fill(0);
  let i = 0;
  while (i < all.length) {
    let j = i;
    while (j + 1 < all.length && all[j + 1].v === all[i].v) j++;
    const avg = (i + j + 2) / 2; // 1-based average rank of the tie group
    for (let k = i; k <= j; k++) ranks[k] = avg;
    i = j + 1;
  }
  let rankSumPos = 0;
  all.forEach((e, idx) => { if (e.p) rankSumPos += ranks[idx]; });
  return (rankSumPos - (pos.length * (pos.length + 1)) / 2) / (pos.length * neg.length);
}

export const EVAL_OK_COLOR = "rgba(140, 150, 170, .8)";
export const EVAL_NG_COLOR = "rgba(213, 94, 0, .85)";   // vermilion — colorblind-safe
export const EVAL_FP_COLOR = "rgba(78, 158, 255, .85)";

// Shape of the lifted separation-check summary (BankProjection computes it so
// the map and the check card share one metric / threshold / verdict).
export type EvalStats = {
  okV: number[]; ngV: number[]; fpV: number[];
  auroc: number; thr: number; autoThr: number;
  misses: [string, StoredImageEval][]; falsePos: [string, StoredImageEval][];
  val: (r: StoredImageEval) => number;
} | null;

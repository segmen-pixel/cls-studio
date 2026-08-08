// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// Types the teach tab's panels hand to each other. They live outside
// Develop.tsx because the panels are imported BY it -- putting the shared
// shape in the importer is how a module cycle starts.

// Verdict-scale anchors lifted from the separation check to the preview
// heatmap and the NG-degree display. Everything is on the α-composite scale
// the verdicts are judged on (raw + α/(ε+cmin); equals the raw scale at
// α=0), so "looks red", "NG度 > 50" and "judged NG" always agree.
export type VerdictAnchors = {
  /** OK images' median statistic — the heatmap's transparent end. */
  lo: number;
  /** Operative threshold (auto Youden or the manual slider override). */
  hi: number;
  okMean: number;
  okStd: number;
  /** α the stats were computed with — forwarded to /score so the server
   *  composes the same per-patch scale into the heatmap. */
  alpha: number;
  metric: "p99" | "topk";
  k: number;
  /** Per-image operative statistic keyed by `tier/label/name`. */
  values: Map<string, number>;
};

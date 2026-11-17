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
//
// One scale, TWO anchors, and that part is not a redundancy. NG度 scores a
// whole image, so it is anchored on an image-level statistic (`lo`); the
// heatmap colours one patch, so it is anchored on a patch-level one
// (`heatLo`). They were the same field once, which meant the heatmap's blue
// end sat near an image's own maximum -- on a live 667-image bank, 99.7% of a
// good image was clipped to one flat blue and the hot side saturated too, so
// there was no gradient left to read.
export type VerdictAnchors = {
  /** OK images' median IMAGE-level statistic. Anchors NG度, which judges a
   *  whole image against `hi` — not the heatmap, which colours one patch at a
   *  time and needs `heatLo`. Sharing one anchor between the two is what made
   *  the map a flat colour: an image-level statistic is a near-maximum over
   *  patches, so as a per-pixel floor it clipped almost everything. */
  lo: number;
  /** OK images' median PATCH floor, same α-composite scale — the heatmap's
   *  blue end. */
  heatLo: number;
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

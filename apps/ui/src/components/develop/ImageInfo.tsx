// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import React from "react";
import { useI18n } from "../../i18n";
import { type BankImage, type StoredImageEval } from "../../api/cls";
import { AZURE, BORDER, INK, TYPE, VERM } from "../../ui/tokens";
import Glyph from "../Glyph";
import { VerdictAnchors } from "./types";

/** Per-image facts and raw scores for whatever is in the viewer. It used to
 *  fill the letterbox dead space beside the image; in the rail it is the last
 *  of the three things read top to bottom -- the figure, its knobs, then the
 *  numbers for this one image. */
export default function ImageInfo({ preview, previewEval, heatAnchors, border }: {
  preview: BankImage;
  previewEval: StoredImageEval | null;
  heatAnchors: VerdictAnchors | null;
  border: string;
}) {
  const { t } = useI18n();
  const ts = previewEval?.top_scores ?? [];
  const kk = Math.min(10, ts.length);
  const rawTopk = kk > 0 ? ts.slice(0, kk).reduce((a, b) => a + b, 0) / kk : null;
  // Operative statistic on the verdict scale: the α-composite value the
  // separation check judged this image with (same metric / k / α). Falls back
  // to the raw top-10 mean when the image was never swept — anchors are absent
  // then anyway.
  const statKey = `${preview.tier}/${preview.label}/${preview.name}`;
  const stat = heatAnchors?.values.get(statKey) ?? rawTopk;
  const statLabel = heatAnchors
    ? (heatAnchors.metric === "p99" ? "p99" : t("develop.info.topkK").replace("{k}", String(heatAnchors.k)))
      + (heatAnchors.alpha > 0 ? t("develop.info.withAlpha") : "")
    : t("develop.info.topk");
  const thr = heatAnchors?.hi ?? null;
  // NG degree on a fixed 0–100 scale, sharing the heatmap's anchors: 0 = OK
  // level (OK median), 50 = exactly at the operative threshold, 100 = where
  // the heatmap saturates full vermilion. Clamped — comparable across banks by
  // design, and consistent with the verdict at any α (>50 ⇔ judged NG).
  const ngdo = stat != null && heatAnchors && heatAnchors.hi > heatAnchors.lo
    ? Math.max(0, Math.min(100, ((stat - heatAnchors.lo) / (2 * (heatAnchors.hi - heatAnchors.lo))) * 100))
    : null;
  const row = (label: string, value: React.ReactNode) => (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 8, fontSize: TYPE.base }}>
      <span className="muted" style={{ whiteSpace: "nowrap" }}>{label}</span>
      <span style={{ fontVariantNumeric: "tabular-nums", textAlign: "right", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>{value}</span>
    </div>
  );
  return (
    <div data-tutorial-step="develop-info" style={{ flex: "0 1 auto", minWidth: 0, minHeight: 0, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6, border, borderRadius: 10, padding: "10px 12px" }}>
      <b style={{ fontSize: TYPE.md }}>{t("develop.info.title")}</b>
      {row(t("develop.info.tier"), (
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
          <Glyph tier={preview.tier} size={12} />
          {t(`develop.tier.${preview.tier}` as "develop.tier.normal")}
        </span>
      ))}
      {(preview.tier === "critical" || preview.tier === "negative") &&
        row(t("develop.info.label"), preview.label && preview.label !== "_default" ? preview.label : t("develop.unlabeled"))}
      {row(t("develop.info.patches"), (preview.patches || previewEval?.patches || 0).toLocaleString())}
      {preview.tier === "critical" && row(t("develop.info.marks"), String(preview.annotations?.length ?? 0))}
      <span style={{ height: 1, background: BORDER, margin: "2px 0" }} />
      <b style={{ fontSize: TYPE.md }}>{t("develop.info.scores")}</b>
      {previewEval ? (
        <>
          {ngdo != null && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4, padding: "2px 0 4px" }}>
              <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
                <span className="muted" style={{ fontSize: TYPE.base }}>{t("develop.info.ngdo")}</span>
                <span style={{ fontVariantNumeric: "tabular-nums" }}>
                  <b style={{ fontSize: TYPE.hero, color: ngdo >= 50 ? VERM : AZURE }}>{Math.round(ngdo)}</b>
                  <span className="muted" style={{ fontSize: TYPE.sm }}> / 100</span>
                </span>
              </div>
              {/* 0–100 bar with a tick at 50 (= the threshold) */}
              <div style={{ position: "relative", height: 8, borderRadius: 999, background: "rgba(127,127,127,.18)", overflow: "hidden" }}>
                <div style={{ width: `${ngdo}%`, height: "100%", background: ngdo >= 50 ? VERM : AZURE }} />
                <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 2, background: INK, opacity: .7 }} />
              </div>
              <span className="muted" style={{ fontSize: TYPE.xs }}>{t("develop.info.ngdoScale")}</span>
            </div>
          )}
          {row(statLabel, <b style={{ fontSize: TYPE.md }}>{stat != null ? stat.toFixed(2) : "–"}</b>)}
          {row("max", previewEval.score_max.toFixed(2))}
          {row("p99", previewEval.score_p99.toFixed(2))}
          {row("mean", previewEval.score_mean.toFixed(2))}
          {thr != null && row(t("develop.info.threshold"), thr.toFixed(2))}
        </>
      ) : (
        <span className="muted" style={{ fontSize: TYPE.sm }}>{t("develop.info.scoring")}</span>
      )}
    </div>
  );
}

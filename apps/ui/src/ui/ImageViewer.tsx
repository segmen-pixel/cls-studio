// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// The black image pane the teach and inspect tabs both look through.
//
// It owns the three things that have to agree between them: the pane itself
// and its grab cursor, the layer that carries the zoom/pan transform, and the
// zoom readout. What is being shown goes in as children and rides the
// transform; chrome that must stay put -- a file name, a heatmap toggle --
// goes in as `overlay` and does not.
import React from "react";
import { useI18n } from "../i18n";
import { TYPE } from "./tokens";
import type { ZoomPan } from "./useZoomPan";

type Props = {
  zoom: ZoomPan;
  /** Rides the zoom/pan transform. */
  children: React.ReactNode;
  /** Pinned to the pane, above the image. */
  overlay?: React.ReactNode;
  title?: string;
  /** Merged over the pane's own layout, for the odd container difference. */
  style?: React.CSSProperties;
  "data-tutorial-step"?: string;
};

export default function ImageViewer({ zoom, children, overlay, title, style, ...rest }: Props) {
  const { t } = useI18n();
  const { view, panning, spacePressed, attach, handlers } = zoom;

  return (
    <div
      ref={attach}
      {...handlers}
      {...rest}
      title={title}
      style={{
        flex: 1, minHeight: 0, background: "#000", borderRadius: 8,
        overflow: "hidden", position: "relative", touchAction: "none",
        cursor: panning ? "grabbing" : (spacePressed || view.scale > 1) ? "grab" : "default",
        ...style,
      }}
    >
      <div style={{
        position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
        transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`, transformOrigin: "0 0",
      }}>
        {children}
      </div>
      {overlay}
      {view.scale > 1 && (
        <span style={{
          position: "absolute", top: 6, left: 8, fontSize: TYPE.sm, padding: "2px 6px", borderRadius: 4,
          background: "rgba(0,0,0,.55)", color: "#fff", pointerEvents: "none", fontVariantNumeric: "tabular-nums",
        }}>
          {t("develop.viewer.zoom").replace("{pct}", String(Math.round(view.scale * 100)))}
        </span>
      )}
    </div>
  );
}

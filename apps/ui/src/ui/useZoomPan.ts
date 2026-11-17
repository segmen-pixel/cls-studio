// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// Wheel-zoom and drag-pan for an image pane.
//
// The teach tab and the inspect tab had this written out twice, line for line
// -- the same 1x-8x clamp, the same cursor-anchored zoom, the same
// best-effort pointer capture -- and the copies had already begun to drift:
// one gated its Space-key listener on the tab being visible and the other did
// not, so a component that is always mounted was calling preventDefault on
// Space no matter which tab you were looking at.
//
// Zoom is anchored on the cursor rather than the pane's centre: zooming about
// the centre walks whatever you were looking at off the edge.
import React, { useCallback, useEffect, useRef, useState } from "react";

export type View = { scale: number; x: number; y: number };

/** Fully zoomed out. Also what a double-click and a 1x wheel-out return to. */
export const FIT: View = { scale: 1, x: 0, y: 0 };

const MAX_SCALE = 8;

export type ZoomPan = {
  view: View;
  panning: boolean;
  spacePressed: boolean;
  reset: () => void;
  /** Ref callback for the pane. Attaches the wheel listener as it mounts. */
  attach: (el: HTMLDivElement | null) => void;
  handlers: {
    onPointerDown: (e: React.PointerEvent) => void;
    onPointerMove: (e: React.PointerEvent) => void;
    onPointerUp: () => void;
    onPointerCancel: () => void;
    onDoubleClick: () => void;
  };
};

/**
 * @param resetKey  Zoom returns to fit whenever this changes -- carrying a 6x
 *                  zoom onto the next image hides what you switched to see.
 * @param active    Whether the owning tab is showing. The Space key is only
 *                  watched while it is, because the handler preventDefaults.
 */
export function useZoomPan(resetKey: unknown, active: boolean): ZoomPan {
  const [view, setView] = useState<View>(FIT);
  const [spacePressed, setSpacePressed] = useState(false);
  const [panning, setPanning] = useState(false);
  const panStart = useRef<{ px: number; py: number; x: number; y: number } | null>(null);

  const reset = useCallback(() => setView(FIT), []);
  useEffect(() => { setView(FIT); }, [resetKey]);

  useEffect(() => {
    if (!active) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "BUTTON") return;
      e.preventDefault();
      setSpacePressed(true);
    };
    const onKeyUp = (e: KeyboardEvent) => { if (e.code === "Space") setSpacePressed(false); };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [active]);

  // Non-passive, because React's onWheel is passive and cannot preventDefault
  // the page scroll. A ref callback rather than an effect over a ref: the pane
  // is conditionally rendered, and this attaches exactly when it appears.
  const paneRef = useRef<HTMLDivElement | null>(null);
  const onWheel = useRef((e: WheelEvent) => {
    e.preventDefault();
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setView((v) => {
      const factor = Math.sign(e.deltaY) > 0 ? 0.9 : 1.1;
      const next = Math.max(1, Math.min(MAX_SCALE, v.scale * factor));
      if (next === v.scale) return v;
      if (next === 1) return FIT;
      // Keep the point under the cursor stationary while zooming.
      const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
      return {
        scale: next,
        x: cx - ((cx - v.x) / v.scale) * next,
        y: cy - ((cy - v.y) / v.scale) * next,
      };
    });
  }).current;

  const attach = useCallback((el: HTMLDivElement | null) => {
    if (paneRef.current) paneRef.current.removeEventListener("wheel", onWheel);
    paneRef.current = el;
    if (el) el.addEventListener("wheel", onWheel, { passive: false });
  }, [onWheel]);

  useEffect(() => () => {
    if (paneRef.current) paneRef.current.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    const pan = view.scale > 1 && (e.button === 1 || spacePressed || e.button === 0);
    if (!pan) return;
    e.preventDefault();
    panStart.current = { px: e.clientX, py: e.clientY, x: view.x, y: view.y };
    setPanning(true);
    // Best-effort capture: keeps the pan alive when the cursor leaves the
    // pane mid-drag. Throws for synthetic pointers — never abort the pan.
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* ignore */ }
  }, [view, spacePressed]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const st = panStart.current;
    if (!st) return;
    setView((v) => ({ ...v, x: st.x + (e.clientX - st.px), y: st.y + (e.clientY - st.py) }));
  }, []);

  const onPointerUp = useCallback(() => { panStart.current = null; setPanning(false); }, []);

  return {
    view, panning, spacePressed, reset, attach,
    handlers: {
      onPointerDown, onPointerMove, onPointerUp,
      onPointerCancel: onPointerUp,
      onDoubleClick: reset,
    },
  };
}

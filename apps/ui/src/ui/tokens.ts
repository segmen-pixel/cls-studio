// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// The design tokens, for the parts of the UI that style themselves inline.
//
// An inline style cannot reach a stylesheet, so every component that wanted a
// palette colour wrote `var(--accent, "#0f766e")` by hand -- and the second
// argument drifted. Nine sites carried a teal from the palette this app grew
// out of, seventeen carried the current cyan, and two named a variable that
// was never declared at all, so their fallback was what actually painted, in
// both colour schemes. A fallback is a promise about what the page looks like
// when the stylesheet fails to load; it should be made once, here, and match
// :root in styles/base.css exactly.
//
// Colours stay var() strings rather than hexes so the theme still drives them.

/**
 * The type scale. Everything the panels render is one of these; 12px is the
 * floor, because these sit on dense lists that are read at a glance.
 */
export const TYPE = {
  xs: 12,
  sm: 12.5,
  base: 13,
  md: 14,
  lg: 15,
  xl: 16,
  xxl: 18,
  hero: 20,
  verdict: 40,
} as const;

/* Palette. Each fallback is the dark-scheme :root value verbatim. */
export const ACCENT = "var(--accent, #22C7DB)";
export const ACCENT_SOFT = "var(--accent-soft, rgba(34, 199, 219, 0.15))";
export const INK = "var(--ink, #F5F5F7)";
export const MUTED = "var(--muted, #98989D)";
export const PANEL = "var(--panel, #2C2C2E)";
export const PANEL_2 = "var(--panel-2, #3A3A3C)";
/** The floating strips that sit ON something -- a toolbox over a picture.
 *  Not PANEL: an opaque slab over an image reads as a hole punched in it. */
export const GLASS = "var(--glass, rgba(44,44,46,0.72))";
export const BORDER = "var(--border, rgba(255,255,255,0.12))";
export const DANGER = "var(--danger, #FF453A)";
export const OK = "var(--ok, #30D158)";

/** The hairline these panels draw between everything. */
export const RULE = `1px solid ${BORDER}`;

/*
 * Data marks. Okabe-Ito, and deliberately not themed: a tier has to read the
 * same in both schemes, and shape carries the meaning here because this
 * operator cannot separate blue from purple. Vermilion, never purple.
 */
export const VERM = "#D55E00";
export const AZURE = "#0072B2";
export const SLATE = "#4A5F6D";

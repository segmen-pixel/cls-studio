// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// The tier mark, in one place.
//
// It was a shaped <i> on the bank tab and a text bullet on the check tab, at
// different sizes and in different colours, so the same three tiers had two
// alphabets depending on which list you happened to be reading. One list is
// the other list.
//
// Shape carries the tier and colour only reinforces it: this operator cannot
// separate blue from purple, and a list scanned quickly must not depend on
// hue. Vermilion, never purple. At the 8px these started out at, a circle, a
// triangle and a square are the same grey smudge -- hence sized here, and
// defaulted large.
import React from "react";
import type { Tier } from "../api/cls";
import { MUTED, SLATE, VERM } from "../ui/tokens";

/** The glyph's own ink, for text that sits beside one and has to agree with it. */
export const TIER_INK: Record<Tier, string> = {
  normal: MUTED,
  critical: VERM,
  negative: SLATE,
};

export default function Glyph({ tier, size = 14 }: { tier: "" | Tier; size?: number }) {
  const base: React.CSSProperties = { display: "inline-block", flex: "none" };
  if (tier === "normal") {
    return <i style={{ ...base, width: size, height: size, borderRadius: "50%", background: TIER_INK.normal }} />;
  }
  if (tier === "critical") {
    return (
      <i style={{
        ...base, width: 0, height: 0,
        borderLeft: `${size * 0.58}px solid transparent`,
        borderRight: `${size * 0.58}px solid transparent`,
        borderBottom: `${size}px solid ${VERM}`,
      }} />
    );
  }
  if (tier === "negative") {
    return <i style={{ ...base, width: size, height: size, borderRadius: 2, background: SLATE }} />;
  }
  // Unlabelled: the same circle, hollow. A tier that has not been decided yet
  // should read as an empty slot, not as a fourth kind of thing.
  return (
    <i style={{
      ...base, width: size, height: size, borderRadius: "50%",
      border: `2px dashed ${MUTED}`, boxSizing: "border-box", opacity: 0.7,
    }} />
  );
}

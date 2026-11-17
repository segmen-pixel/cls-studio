// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
/** localStorage for values that belong to one bank, not to the browser.
 *
 * The alpha and beta sliders were filed under plain "anom.alpha" / "anom.beta".
 * Both tabs read them, which is the point -- the alpha proven on the separation
 * check is what inspection should run with -- but the key said nothing about
 * WHICH bank it was proven on. Open another project and its exemplars are
 * scored with the previous project's weight, silently, because a number that
 * loads is indistinguishable from a number that belongs.
 *
 * Keys are scoped by the binding the server itself guards writes with, so the
 * two identities cannot drift apart. The authoritative copy of these values is
 * runtime_config.json in the bank directory; this is only what carries them
 * between tabs before they are saved there.
 */

const SEP = "@";

function scopedName(name: string, scope: string | null): string | null {
  return scope ? `${name}${SEP}${scope}` : null;
}

/** Read a number filed under this bank. Returns null when unscoped or unset,
 *  so a caller can tell "no value yet" from "a value of zero". */
export function readScopedNumber(name: string, scope: string | null): number | null {
  const key = scopedName(name, scope);
  if (!key) return null;
  const raw = localStorage.getItem(key);
  if (raw === null) return null;
  const v = Number(raw);
  return Number.isFinite(v) ? v : null;
}

/** Write a number under this bank. A write with no scope is dropped rather
 *  than falling back to a global key -- that fallback is the bug. */
export function writeScopedNumber(name: string, scope: string | null, value: number): void {
  const key = scopedName(name, scope);
  if (!key) return;
  try { localStorage.setItem(key, String(value)); } catch { /* quota, private mode */ }
}

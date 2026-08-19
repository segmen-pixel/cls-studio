// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// The modal a long job puts up. Portalled to <body>: it is deliberately not
// gated on the tab being active -- these jobs block the whole app on purpose,
// and rendered inside a tab panel `display: none` took the only cancel button
// away with the tab.
import { createPortal } from "react-dom";
import { INK, PANEL, TYPE } from "../../ui/tokens";

/** Full-screen modal that blocks every interaction behind a long-running
 *  job (bank activation, separation sweep, heatmap pre-render). These jobs
 *  hold the GPU and mutate bank/eval state server-side — side actions taken
 *  meanwhile would race them — so the whole app waits, with an optional
 *  cancel for the jobs that support it. */
export default function BusyDialog({ title, hint, onCancel, cancelLabel, cancelDisabled, border }: {
  title: string;
  hint?: string;
  onCancel?: () => void;
  cancelLabel?: string;
  /** Set once the cancel has been requested: the click must visibly land even
   *  while the loop is still winding down the in-flight request. */
  cancelDisabled?: boolean;
  border: string;
}) {
  // Portalled to <body>, not rendered in place: these dialogs live inside a
  // tab panel that goes display:none when the user switches tabs, so the whole
  // overlay — including the only cancel button for a job that keeps running —
  // simply vanished, with no way left to stop it. The z-index also clears the
  // tutorial overlay (9999, pointer-events:auto on its plain backdrop), which
  // otherwise sat above and swallowed the clicks (2026-07-31).
  return createPortal(
    <div
      role="alertdialog" aria-modal="true" aria-busy="true" aria-label={title}
      style={{ position: "fixed", inset: 0, zIndex: 10000, background: "rgba(0,0,0,.35)", display: "flex", alignItems: "center", justifyContent: "center" }}
    >
      <div style={{ background: PANEL, border, borderRadius: 12, padding: "18px 24px", display: "flex", alignItems: "center", gap: 14, boxShadow: "0 8px 32px rgba(0,0,0,.45)" }}>
        <span className="train-spinner" />
        <div>
          <div style={{ fontWeight: 600, fontSize: TYPE.lg, fontVariantNumeric: "tabular-nums" }}>{title}</div>
          {hint && <div className="muted" style={{ fontSize: TYPE.base, marginTop: 2 }}>{hint}</div>}
        </div>
        {onCancel && (
          <button
            onClick={onCancel}
            disabled={cancelDisabled}
            style={{ padding: "4px 12px", fontSize: TYPE.base, borderRadius: 6, border, background: "transparent", color: INK, cursor: cancelDisabled ? "default" : "pointer", opacity: cancelDisabled ? 0.6 : 1, whiteSpace: "nowrap" }}
          >{cancelLabel}</button>
        )}
      </div>
    </div>,
    document.body,
  );
}

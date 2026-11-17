// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import React from "react";
import type { StartupWarning } from "../types";
import { TYPE } from "../../ui/tokens";

type StartupWarningsProps = {
  warnings: StartupWarning[];
  onDismiss: () => void;
};

export default React.memo(function StartupWarnings({ warnings, onDismiss }: StartupWarningsProps) {
  if (warnings.length === 0) return null;

  return (
    <div className="settings-overlay" onClick={onDismiss}>
      <div className="settings-panel" style={{ maxWidth: 480 }} onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2 style={{ fontSize: TYPE.xl }}>{warnings[0].level === "error" ? "\u26A0 " : "\u2139 "}{warnings[0].title}</h2>
          <button className="ghost" onClick={onDismiss} aria-label="Close">&times;</button>
        </div>
        {warnings.map((w, i) => (
          <section key={i} style={{ marginBottom: 12 }}>
            {i > 0 && <h3 style={{ fontSize: TYPE.base, marginBottom: 4 }}>{w.level === "error" ? "\u26A0 " : "\u2139 "}{w.title}</h3>}
            <div style={{ fontSize: TYPE.xs, color: "#bbb", whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{w.message}</div>
          </section>
        ))}
        <button
          className="primary"
          style={{ width: "100%", marginTop: 8 }}
          onClick={() => {
            sessionStorage.setItem("seg_startup_warnings_dismissed", "1");
            onDismiss();
          }}
        >OK</button>
      </div>
    </div>
  );
});

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// Raised when the server answers 401 — i.e. this browser reached a LAN-bound
// server without a session. Before this existed the app showed a spinner that
// never resolved, which reads as "the server is down" when the server is fine
// and simply does not know who you are.
import React, { useCallback, useRef, useState } from "react";
import { signIn } from "../api/auth";
import { useI18n } from "../i18n";
import { ACCENT, BORDER, INK, PANEL, RULE, TYPE, VERM } from "../ui/tokens";

type Props = { onSignedIn: () => void };

export default function SignInGate({ onSignedIn }: Props) {
  const { t } = useI18n();
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const submit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token.trim() || busy) return;
    setBusy(true);
    setFailed(false);
    try {
      await signIn(token.trim());
      onSignedIn();
    } catch {
      setFailed(true);
      setBusy(false);
      inputRef.current?.select();
    }
  }, [token, busy, onSignedIn]);

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 9000,
      background: "rgba(8, 12, 16, .72)", backdropFilter: "blur(3px)",
      display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
    }}>
      <form
        onSubmit={submit}
        style={{
          width: "min(460px, 100%)", background: PANEL,
          border: RULE, borderRadius: 12,
          padding: "22px 24px", display: "flex", flexDirection: "column", gap: 12,
          boxShadow: "0 18px 48px -20px rgba(0,0,0,.7)",
        }}
      >
        <b style={{ fontSize: TYPE.xl }}>{t("auth.title")}</b>
        <span className="muted" style={{ fontSize: TYPE.sm, lineHeight: 1.7 }}>
          {t("auth.body")}
        </span>
        <input
          ref={inputRef}
          type="password"
          autoFocus
          autoComplete="current-password"
          value={token}
          onChange={(e) => { setToken(e.target.value); setFailed(false); }}
          placeholder={t("auth.token")}
          style={{
            padding: "9px 12px", borderRadius: 8, fontSize: TYPE.md,
            border: `1px solid ${failed ? VERM : BORDER}`,
            background: "transparent", color: INK,
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          }}
        />
        {failed && (
          <span style={{ fontSize: TYPE.xs, color: VERM, fontWeight: 600 }}>{t("auth.failed")}</span>
        )}
        <button
          type="submit"
          disabled={busy || !token.trim()}
          style={{
            padding: "9px 14px", borderRadius: 8, border: "none",
            background: ACCENT, color: "#fff", fontWeight: 700,
            fontSize: TYPE.md, cursor: busy ? "default" : "pointer", opacity: token.trim() ? 1 : 0.5,
          }}
        >
          {busy ? t("auth.signingIn") : t("auth.signIn")}
        </button>
        <span className="muted" style={{ fontSize: TYPE.xs, lineHeight: 1.7 }}>
          {t("auth.where")}
        </span>
      </form>
    </div>
  );
}

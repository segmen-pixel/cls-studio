// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import React, { useState, useEffect } from "react";
import { useI18n } from "../i18n";
import { MUTED, RULE, TYPE, VERM } from "../ui/tokens";
import { fetchLanToken } from "../api/auth";
import { fetchNetworkSettings, updateNetworkSettings } from "../api";
import type { NetworkSettings } from "../api";
import { fetchBankCapacity, setBankCapacity, fetchCompressionSettings, updateCompressionSettings } from "../api/cls";
import type { BankCapacity, BankCapacityInfo, CompressionSettings } from "../api/cls";

export interface SettingsValues {
  valRatio: number;
  testRatio: number;
  previewStyle: number;
}

interface SettingsDialogProps {
  open: boolean;
  onClose: () => void;
  values: SettingsValues;
  onChange: <K extends keyof SettingsValues>(key: K, value: SettingsValues[K]) => void;
  showToast?: (msg: string) => void;
  onLibraryChanged?: () => void;
}

function NetworkSettingsSection() {
  const { t } = useI18n();
  const [state, setState] = useState<NetworkSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const [revealed, setRevealed] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetchNetworkSettings().then(setState).catch(() => {});
    // 403 unless this page is served from the machine running the server; that
    // is the expected answer everywhere else, so it stays silent and the block
    // below simply does not render.
    fetchLanToken().then(setToken).catch(() => setToken(""));
  }, []);

  const copyToken = async () => {
    try {
      await navigator.clipboard.writeText(token);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch {
      // No clipboard permission (or an insecure context): show it instead so
      // it can at least be selected by hand.
      setRevealed(true);
    }
  };

  const handleToggle = async (lan_access: boolean) => {
    setSaving(true);
    setSavedMsg(null);
    try {
      const next = await updateNetworkSettings(lan_access);
      setState(next);
      setSavedMsg(t("settings.network.restartRequired"));
      setTimeout(() => setSavedMsg(null), 6000);
    } catch (err) {
      setSavedMsg(`Failed: ${(err as Error).message}`);
    } finally {
      setSaving(false);
    }
  };

  if (!state) return null;
  // Prefer the port the UI is actually served from (single-port layout serves
  // UI and API on the same origin); 8791 is only the fallback for dev setups
  // where the UI runs on a separate dev server.
  const port = Number(window.location.port) || 8791;
  const showTokenWarn = state.lan_access && !state.api_token_configured;
  const showProxyWarn = state.lan_access && (state.cvat_proxy_configured || state.annotation_proxy_configured);
  const currentLabel = state.current_bind_host === "0.0.0.0"
    ? t("settings.network.currentBind.lan")
    : t("settings.network.currentBind.loopback");

  return (
    <section className="settings-section-divider">
      <h3 style={{ color: "#4fc3f7" }}>🌐 {t("settings.network")}</h3>
      <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
        <input
          type="checkbox"
          checked={state.lan_access}
          disabled={saving}
          onChange={(e) => handleToggle(e.target.checked)}
        />
        <span>{t("settings.network.lanAccess")}</span>
      </label>
      <p className="muted" style={{ fontSize: TYPE.xs, marginBottom: 6 }}>
        {t("settings.network.lanAccess.desc")}
      </p>
      <div style={{ fontSize: TYPE.xs, color: MUTED, marginBottom: 6 }}>
        {currentLabel}
        {state.restart_required && (
          <span style={{ color: "#ffb74d", marginLeft: 8 }}>
            ⟳ {t("settings.network.restartRequired")}
          </span>
        )}
      </div>
      {state.lan_access && (
        <div style={{ fontSize: TYPE.xs, marginBottom: 6 }}>
          <span style={{ color: MUTED }}>{t("settings.network.lanUrls")}: </span>
          {state.lan_addresses.length > 0 ? (
            state.lan_addresses.map((ip) => (
              <code key={ip} style={{ marginRight: 8, color: "#a5d6a7" }}>
                http://{ip}:{port}/ui/
              </code>
            ))
          ) : (
            <span style={{ color: "#ef9a9a" }}>{t("settings.network.lanUrls.none")}</span>
          )}
        </div>
      )}
      {/* The token, but only to somebody sitting at this machine. The launcher
          prints it once into runtime_settings.json, which is no help when you
          are holding a phone and the browser is asking you to sign in. */}
      {state.lan_access && token && (
        <div style={{ fontSize: TYPE.xs, marginBottom: 6, display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ color: MUTED }}>{t("settings.network.token")}</span>
          <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
            <code style={{
              color: "#a5d6a7", padding: "3px 8px", borderRadius: 4,
              background: "rgba(255,255,255,.04)", userSelect: "all",
              letterSpacing: revealed ? 0 : 2,
            }}>
              {revealed ? token : "•".repeat(Math.min(token.length, 24))}
            </code>
            <button type="button" onClick={() => setRevealed((v) => !v)}>
              {revealed ? t("settings.network.token.hide") : t("settings.network.token.show")}
            </button>
            <button type="button" onClick={copyToken}>
              {copied ? t("settings.network.token.copied") : t("settings.network.token.copy")}
            </button>
          </div>
          <span className="muted" style={{ fontSize: TYPE.xs, lineHeight: 1.6 }}>
            {t("settings.network.token.desc")}
          </span>
        </div>
      )}
      {savedMsg && (
        <div style={{ fontSize: TYPE.xs, color: savedMsg.startsWith("Failed") ? "#ef9a9a" : "#a5d6a7", marginBottom: 6 }}>
          {savedMsg}
        </div>
      )}
      {(showTokenWarn || showProxyWarn) && (
        <div style={{ background: "rgba(255,152,0,0.08)", border: "1px solid rgba(255,152,0,0.3)", borderRadius: 6, padding: "8px 10px", fontSize: TYPE.xs, lineHeight: 1.5 }}>
          <div style={{ fontWeight: 600, color: "#ff9800", marginBottom: 4 }}>
            ⚠ {t("settings.network.warnings")}
          </div>
          {showTokenWarn && <div>• {t("settings.network.warnTokenMissing")}</div>}
          {showProxyWarn && <div>• {t("settings.network.warnProxyEnabled")}</div>}
          <div>• {t("settings.network.firewallHint")}</div>
        </div>
      )}
    </section>
  );
}

function BankCapacitySection() {
  const { t } = useI18n();
  const [info, setInfo] = useState<BankCapacityInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  useEffect(() => {
    // 404s when no bank is active → section stays hidden.
    fetchBankCapacity().then(setInfo).catch(() => {});
  }, []);

  const choose = async (cap: BankCapacity) => {
    if (busy || info?.capacity === cap) return;
    setBusy(true);
    setSavedMsg(null);
    try {
      setInfo(await setBankCapacity(cap));
      setSavedMsg(t("settings.bankCapacity.saved"));
      setTimeout(() => setSavedMsg(null), 4000);
    } catch (err) {
      setSavedMsg(`Failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  if (!info) return null;

  const label: Record<BankCapacity, string> = {
    small: t("settings.bankCapacity.small"),
    medium: t("settings.bankCapacity.medium"),
    large: t("settings.bankCapacity.large"),
  };
  const full = info.ceiling > 0 && info.normal >= info.ceiling;
  // Colour-blind safe: bar colour is backed by a text label (⚠ At capacity),
  // and the active tier is marked with ● + a border, not colour alone.
  const barColor = full || info.pct >= 90 ? VERM : "#4fc3f7";

  return (
    <section className="settings-section-divider">
      <h3 style={{ color: "#4fc3f7" }}>🗄 {t("settings.bankCapacity")}</h3>
      <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
        {(["small", "medium", "large"] as BankCapacity[]).map((tier) => {
          const active = info.capacity === tier;
          return (
            <button
              key={tier}
              disabled={busy}
              aria-pressed={active}
              onClick={() => choose(tier)}
              style={{
                flex: 1,
                padding: "6px 4px",
                fontSize: TYPE.base,
                borderRadius: 6,
                border: active ? "2px solid #4fc3f7" : RULE,
                background: active ? "rgba(79,195,247,0.15)" : "transparent",
                fontWeight: active ? 700 : 400,
                cursor: busy ? "default" : "pointer",
              }}
            >
              {active ? "● " : ""}{label[tier]}
            </button>
          );
        })}
      </div>
      <div style={{ height: 8, borderRadius: 4, background: "rgba(255,255,255,0.08)", overflow: "hidden", marginBottom: 4 }}>
        <div style={{ width: `${Math.min(100, info.pct)}%`, height: "100%", background: barColor, transition: "width .2s" }} />
      </div>
      <div style={{ fontSize: TYPE.xs, color: MUTED, marginBottom: 6 }}>
        {info.normal.toLocaleString()} / {info.ceiling.toLocaleString()} {t("settings.bankCapacity.patches")} ({info.pct}%) · ~{info.est_vram_mb} MB
        {full && (
          <span style={{ color: VERM, marginLeft: 8, fontWeight: 600 }}>⚠ {t("settings.bankCapacity.full")}</span>
        )}
      </div>
      {/* The bar above only tracks the normal tier, because that is all the
          ceiling governs. The labelled tiers are resident too and are not
          capped at all, so leaving them out made the gauge read near zero on a
          bank holding gigabytes of them. */}
      {info.labeled > 0 && (
        <div style={{ fontSize: TYPE.xs, color: MUTED, marginBottom: 6 }}>
          {t("settings.bankCapacity.labeled")
            .replace("{n}", info.labeled.toLocaleString())
            .replace("{total}", String(info.est_vram_total_mb))}
        </div>
      )}
      <p className="muted" style={{ fontSize: TYPE.xs, marginBottom: 6 }}>{t("settings.bankCapacity.desc")}</p>
      {savedMsg && (
        <div style={{ fontSize: TYPE.xs, color: savedMsg.startsWith("Failed") ? "#ef9a9a" : "#a5d6a7" }}>{savedMsg}</div>
      )}
    </section>
  );
}

function CompressionSection() {
  const { t } = useI18n();
  const [state, setState] = useState<CompressionSettings | null>(null);
  const [busy, setBusy] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  useEffect(() => {
    fetchCompressionSettings().then(setState).catch(() => {});
  }, []);

  const apply = async (next: CompressionSettings) => {
    if (busy) return;
    setBusy(true);
    setSavedMsg(null);
    try {
      setState(await updateCompressionSettings(next));
      setSavedMsg(t("settings.compression.saved"));
      setTimeout(() => setSavedMsg(null), 6000);
    } catch (err) {
      setSavedMsg(`Failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  if (!state) return null;

  return (
    <section className="settings-section-divider">
      <h3 style={{ color: "#4fc3f7" }}>🗜 {t("settings.compression")}</h3>
      <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
        <input
          type="checkbox"
          checked={state.int8}
          disabled={busy}
          onChange={(e) => apply({ ...state, int8: e.target.checked })}
        />
        <span>{t("settings.compression.int8")}</span>
      </label>
      <p className="muted" style={{ fontSize: TYPE.xs, marginBottom: 8 }}>
        {t("settings.compression.int8.desc")}
      </p>
      <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
        <input
          type="checkbox"
          checked={state.ivf}
          disabled={busy}
          onChange={(e) => apply({ ...state, ivf: e.target.checked })}
        />
        <span>{t("settings.compression.ivf")}</span>
      </label>
      <p className="muted" style={{ fontSize: TYPE.xs, marginBottom: 8 }}>
        {t("settings.compression.ivf.desc")}
      </p>
      {state.ivf && (
        <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8, fontSize: TYPE.xs }}>
          {t("settings.compression.nprobe")}
          <select
            value={state.ivf_nprobe}
            disabled={busy}
            onChange={(e) => apply({ ...state, ivf_nprobe: parseInt(e.target.value, 10) })}
            style={{ fontSize: TYPE.xs, padding: "2px 6px" }}
          >
            {[2, 4, 8, 16].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
      )}
      {savedMsg && (
        <div style={{ fontSize: TYPE.xs, color: savedMsg.startsWith("Failed") ? "#ef9a9a" : "#a5d6a7" }}>{savedMsg}</div>
      )}
    </section>
  );
}

const SettingsDialog: React.FC<SettingsDialogProps> = ({ open, onClose }) => {
  const { t } = useI18n();

  if (!open) return null;

  return (
    <div className="settings-overlay" onClick={onClose} onTouchEnd={(e) => { if (e.target === e.currentTarget) onClose(); }} onTouchMove={(e) => e.stopPropagation()}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()} onTouchEnd={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>{t("settings.title")}</h2>
          <button className="ghost" onClick={onClose} data-desc={t("common.close")} data-desc-pos="bottom">×</button>
        </div>
        {/* Runtime memory-bank size budget */}
        <BankCapacitySection />
        {/* Bank compression (int8 / IVF) */}
        <CompressionSection />
        {/* Network access */}
        <NetworkSettingsSection />
      </div>
    </div>
  );
};

export default React.memo(SettingsDialog);

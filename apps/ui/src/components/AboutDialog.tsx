// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import React from "react";
import type { HealthInfo } from "../api";
import { useI18n } from "../i18n";
import { TYPE } from "../ui/tokens";

const CHANGELOG: { version: string; date: string; changes: { ja: string[]; en: string[] } }[] = [
  {
    version: "0.2.0",
    date: "2026-07-10",
    changes: {
      ja: [
        "🚀 v2: FastAPI + React シェルに全面リビルド (Operator / Develop タブ)",
        "🔍 コア (DINOv2 特徴 / メモリバンク / HITL append / BG-aware backbone) を clscore パッケージに集約",
        "⚡ 推論デバイスを利用可能な最良 GPU に自動選択",
        "⚡ append/score の DINOv2 forward をイベントループ外で実行 (UI 応答性改善)",
        "🌐 単一ポート (8791) 構成、ファイアウォール開放も 1 ポートのみ",
      ],
      en: [
        "🚀 v2: full rebuild on a FastAPI + React shell (Teach / Inspect tabs)",
        "🔍 Core pieces (DINOv2 features / memory bank / HITL append / BG-aware backbone) consolidated into the clscore package",
        "⚡ Inference device auto-selects the best available GPU",
        "⚡ append/score DINOv2 forward now runs off the event loop (snappier UI)",
        "🌐 Single-port (8791) layout — only one firewall port to open",
      ],
    },
  },
];

const LICENSE_ENTRIES = [
  "FastAPI — MIT",
  "Starlette — BSD-3-Clause",
  "Uvicorn — BSD-3-Clause",
  "SQLModel — MIT",
  "SQLAlchemy — MIT",
  "Pydantic — MIT",
  "python-multipart — Apache-2.0",
  "httpx — BSD-3-Clause",
  "PyTorch — BSD-3-Clause",
  "NumPy — BSD-3-Clause",
  "SciPy — BSD-3-Clause",
  "scikit-learn — BSD-3-Clause",
  "Pillow — MIT-CMU",
  "opencv-python-headless — Apache-2.0; bundled FFmpeg LGPL-2.1+",
  "Zarr / numcodecs — MIT",
  "DINOv2 weights — Apache-2.0 (Meta)",
  "React / React DOM — MIT",
];

interface AboutDialogProps {
  open: boolean;
  onClose: () => void;
  healthInfo: HealthInfo | null;
}

const AboutDialog: React.FC<AboutDialogProps> = ({ open, onClose, healthInfo }) => {
  const { t, lang } = useI18n();
  if (!open) return null;
  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel about-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>{t("about.title")}</h2>
          <button className="ghost" onClick={onClose} data-desc={t("common.close")} data-desc-pos="bottom">×</button>
        </div>
        <section className="about-info">
          <div className="about-title">Cls-Studio</div>
          <div className="about-meta">
            <span>v{__APP_VERSION__}</span>
            <span className="muted">Build: {__BUILD_DATE__}</span>
          </div>
          <div className="muted" style={{ fontSize: TYPE.xs }}>Copyright 2026 The Cls-Studio Contributors. Apache License 2.0.</div>
        </section>
        {healthInfo && (
          <section>
            <h3>{t("about.system")}</h3>
            <div className="about-system-grid">
              {healthInfo.disk && (
                <div className="about-system-item">
                  <span className="about-system-label">Disk</span>
                  <span>{healthInfo.disk.free_gb} GB free / {healthInfo.disk.total_gb} GB ({healthInfo.disk.used_pct}% used)</span>
                </div>
              )}
              {healthInfo.ram && (
                <div className="about-system-item">
                  <span className="about-system-label">RAM</span>
                  <span>{healthInfo.ram.available_gb} GB free / {healthInfo.ram.total_gb} GB ({healthInfo.ram.used_pct}% used)</span>
                </div>
              )}
              {healthInfo.gpu && (
                <div className="about-system-item">
                  <span className="about-system-label">GPU</span>
                  <span>{healthInfo.gpu.name} — {healthInfo.gpu.vram_total_mb} MB VRAM</span>
                </div>
              )}
            </div>
          </section>
        )}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <section>
            <h3>{t("about.licenses")}</h3>
            <div className="about-license-scroll">
              <ul className="license-list">
                {LICENSE_ENTRIES.map((e) => <li key={e}>{e}</li>)}
              </ul>
            </div>
          </section>
          <section>
            <h3>{t("about.changelog")}</h3>
            <div className="about-license-scroll">
              {CHANGELOG.map((entry) => (
                <div key={entry.version} style={{ marginBottom: 12 }}>
                  <div style={{ fontWeight: 600, fontSize: TYPE.base }}>
                    v{entry.version} <span className="muted" style={{ fontWeight: 400 }}>({entry.date})</span>
                  </div>
                  <ul style={{ margin: "4px 0 0", paddingLeft: 20, fontSize: TYPE.xs, lineHeight: 1.6 }}>
                    {entry.changes[lang].map((c, i) => <li key={i}>{c}</li>)}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default React.memo(AboutDialog);

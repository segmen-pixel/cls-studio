// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// Inline icons in this file use SVG path data from Feather Icons
// (https://github.com/feathericons/feather), MIT licensed,
// Copyright (c) 2013-2023 Cole Bemis. Some paths are adapted.
import React, { useEffect, useState, useRef, useCallback } from "react";
import { createPortal } from "react-dom";
import { API_BASE } from "../api";
import { releaseVramNow } from "../api/cls";
import { AZURE, VERM } from "../ui/tokens";
import { visibleInterval } from "../app/hooks/useVisibleInterval";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface GpuSnapshot {
  name?: string;
  gpu_util?: number;
  mem_util?: number;
  temp_c?: number;
  vram_used_mb?: number;
  vram_total_mb?: number;
  fan_pct?: number | null;
  power_w?: number | null;
  power_limit_w?: number | null;
  clock_graphics_mhz?: number | null;
  clock_memory_mhz?: number | null;
}

/** This server process's own VRAM. The only number that can distinguish
 *  "cls-studio is holding it" from "something else on this box is". */
interface ProcessVram {
  available: boolean;
  device?: string | null;
  allocated_mb?: number;
  reserved_mb?: number;
}

interface GpuStatsResponse {
  available: boolean;
  gpus: GpuSnapshot[];
  process?: ProcessVram;
  error?: string;
}

const HISTORY_LEN = 60; // 60 samples × 3s = 3 minutes

// ---------------------------------------------------------------------------
// Sparkline SVG (pure, no deps)
// ---------------------------------------------------------------------------
function Sparkline({ data, max, color, width = 80, height = 20 }: {
  data: number[];
  max: number;
  color: string;
  width?: number;
  height?: number;
}) {
  if (data.length < 2) return null;
  const effectiveMax = max || 1;
  const step = width / (HISTORY_LEN - 1);
  const offset = HISTORY_LEN - data.length;
  const pts = data.map((v, i) => {
    const x = (offset + i) * step;
    const y = height - (v / effectiveMax) * (height - 2) - 1;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const areaPath = `M${pts[0]} ${pts.map((p) => `L${p}`).join(" ")} L${((offset + data.length - 1) * step).toFixed(1)},${height} L${(offset * step).toFixed(1)},${height} Z`;
  return (
    <svg width={width} height={height} className="gpu-sparkline">
      <path d={areaPath} fill={color} opacity="0.15" />
      <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth="1.2" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// GPU-Z Style Row
// ---------------------------------------------------------------------------
function StatRow({ label, value, unit, sparkData, sparkMax, color }: {
  label: string;
  value: string | number;
  unit: string;
  sparkData: number[];
  sparkMax: number;
  color: string;
}) {
  return (
    <div className="gpuz-row">
      <span className="gpuz-label">{label}</span>
      <Sparkline data={sparkData} max={sparkMax} color={color} />
      <span className="gpuz-value">{value}</span>
      <span className="gpuz-unit">{unit}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export default React.memo(function GpuMonitor() {
  const [response, setResponse] = useState<GpuStatsResponse | null>(null);
  // "Release now" lives here because this panel is where an operator looks
  // when they think cls-studio is hogging the card.
  const [releasing, setReleasing] = useState(false);
  const [releaseMsg, setReleaseMsg] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [selectedGpu, setSelectedGpu] = useState(0);
  const monitorRef = useRef<HTMLDivElement>(null);
  const [panelPos, setPanelPos] = useState<{ top: number; right: number } | null>(null);
  const historyRef = useRef<Map<number, {
    gpu_util: number[]; temp_c: number[]; vram_used: number[];
    clock_gfx: number[]; clock_mem: number[]; power_w: number[];
  }>>(new Map());
  const prevJsonRef = useRef("");

  const pushHistory = useCallback((gpuIdx: number, snap: GpuSnapshot) => {
    let h = historyRef.current.get(gpuIdx);
    if (!h) {
      h = { gpu_util: [], temp_c: [], vram_used: [], clock_gfx: [], clock_mem: [], power_w: [] };
      historyRef.current.set(gpuIdx, h);
    }
    const push = (arr: number[], val: number) => {
      arr.push(val);
      if (arr.length > HISTORY_LEN) arr.shift();
    };
    push(h.gpu_util, snap.gpu_util ?? 0);
    push(h.temp_c, snap.temp_c ?? 0);
    push(h.vram_used, snap.vram_used_mb ?? 0);
    push(h.clock_gfx, snap.clock_graphics_mhz ?? 0);
    push(h.clock_mem, snap.clock_memory_mhz ?? 0);
    push(h.power_w, snap.power_w ?? 0);
  }, []);

  const handleToggle = useCallback(() => {
    setExpanded((v) => {
      if (!v && monitorRef.current) {
        const rect = monitorRef.current.getBoundingClientRect();
        setPanelPos({ top: rect.bottom + 6, right: window.innerWidth - rect.right });
      }
      return !v;
    });
  }, []);

  // Close on outside click
  useEffect(() => {
    if (!expanded) return;
    const handler = (e: MouseEvent) => {
      if (monitorRef.current && !monitorRef.current.contains(e.target as Node)) {
        const panel = document.querySelector(".gpuz-panel");
        if (panel && panel.contains(e.target as Node)) return;
        setExpanded(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [expanded]);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(`${API_BASE}/hardware/gpu/stats`);
        if (!res.ok) return;
        const text = await res.text();
        if (text !== prevJsonRef.current) {
          prevJsonRef.current = text;
          const parsed: GpuStatsResponse = JSON.parse(text);
          parsed.gpus.forEach((g, i) => pushHistory(i, g));
          setResponse(parsed);
        } else {
          const parsed: GpuStatsResponse = JSON.parse(text);
          parsed.gpus.forEach((g, i) => pushHistory(i, g));
        }
      } catch { /* ignore */ }
    };
    fetchStats();
    return visibleInterval(fetchStats, 3_000);
  }, [pushHistory]);

  if (!response || !response.available || response.gpus.length === 0) return null;

  const gpu = response.gpus[selectedGpu] ?? response.gpus[0];
  const proc = response.process;
  const hist = historyRef.current.get(selectedGpu);
  const gpuPct = gpu.gpu_util ?? 0;
  const temp = gpu.temp_c ?? 0;
  const barColor = (pct: number) => pct > 90 ? "#f44" : pct > 70 ? "#ffa726" : "#4caf50";
  const tempColor = temp > 85 ? "#f44" : temp > 70 ? "#ffa726" : "#8cf";

  const gpuzPanel = expanded && panelPos ? createPortal(
    <div className="gpuz-panel" style={{ top: panelPos.top, right: panelPos.right }}
      onClick={(e) => e.stopPropagation()}>
      {response!.gpus.length > 1 && (
        <div className="gpuz-tabs">
          {response!.gpus.map((g, i) => (
            <button key={i} className={`gpuz-tab ${i === selectedGpu ? "active" : ""}`}
              onClick={() => setSelectedGpu(i)}>
              GPU {i}
            </button>
          ))}
        </div>
      )}
      <div className="gpuz-header">{gpu.name ?? "GPU"}</div>
      <div className="gpuz-grid">
        <StatRow label="GPU Load" value={gpuPct} unit="%"
          sparkData={hist?.gpu_util ?? []} sparkMax={100} color={barColor(gpuPct)} />
        <StatRow label="Temperature" value={temp} unit="°C"
          sparkData={hist?.temp_c ?? []} sparkMax={100} color={tempColor} />
        {/* MACHINE-WIDE: nvidia-smi cannot attribute VRAM to a process under
            WDDM, so a training run on the same card lands in this number. */}
        <StatRow label="VRAM (all)" value={`${gpu.vram_used_mb ?? 0}/${gpu.vram_total_mb ?? 0}`} unit="MB"
          sparkData={hist?.vram_used ?? []} sparkMax={gpu.vram_total_mb ?? 4096} color={VERM} />
        {proc?.available && (
          <StatRow label="VRAM (cls-studio)" value={proc.reserved_mb ?? 0} unit="MB"
            sparkData={[]} sparkMax={gpu.vram_total_mb ?? 4096} color={AZURE} />
        )}
        {gpu.clock_graphics_mhz != null && (
          <StatRow label="GPU Clock" value={gpu.clock_graphics_mhz} unit="MHz"
            sparkData={hist?.clock_gfx ?? []} sparkMax={2500} color="#64b5f6" />
        )}
        {gpu.clock_memory_mhz != null && (
          <StatRow label="Mem Clock" value={gpu.clock_memory_mhz} unit="MHz"
            sparkData={hist?.clock_mem ?? []} sparkMax={10000} color="#81c784" />
        )}
        {gpu.power_w != null && (
          <StatRow label="Power" value={Math.round(gpu.power_w)} unit={`/ ${Math.round(gpu.power_limit_w ?? 0)}W`}
            sparkData={hist?.power_w ?? []} sparkMax={gpu.power_limit_w ?? 300} color="#ffb74d" />
        )}
        {gpu.fan_pct != null && (
          <StatRow label="Fan" value={gpu.fan_pct} unit="%"
            sparkData={[]} sparkMax={100} color="#80cbc4" />
        )}
      </div>
      {/* Sibling of the grid, and NOT a .gpuz-row — that is a 4-column grid
          sized for label/sparkline/value/unit, and a button dropped into its
          72px first column wraps and overflows the row. */}
      <div
        onClick={(e) => e.stopPropagation()}
        style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6 }}
      >
        <button
          className="ghost"
          style={{ whiteSpace: "nowrap", fontSize: 11, padding: "2px 8px" }}
          disabled={releasing}
          onClick={async () => {
            if (releasing) return;
            setReleasing(true);
            try {
              const r = await releaseVramNow();
              // A busy server is a status, not an error: empty_cache cannot
              // reclaim a referenced block, so forcing would free nothing.
              setReleaseMsg(r.released ? `-${r.freed_mb ?? 0} MB` : "busy");
            } catch {
              setReleaseMsg("failed");
            } finally {
              setReleasing(false);
              setTimeout(() => setReleaseMsg(null), 5000);
            }
          }}
        >
          Release VRAM
        </button>
        <span className="gpuz-unit">{releaseMsg ?? ""}</span>
      </div>
    </div>,
    document.body
  ) : null;

  return (
    <div className="gpu-monitor" ref={monitorRef} onClick={handleToggle}
      title={expanded ? "" : `${gpu.name}\nGPU ${gpuPct}% | ${temp}°C`}>
      <div className="gpu-mini">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="4" y="4" width="16" height="12" rx="2" />
          <line x1="8" y1="20" x2="16" y2="20" /><line x1="12" y1="16" x2="12" y2="20" />
        </svg>
        <span className="gpu-mini-bar" style={{ width: 40 }}>
          <span className="gpu-mini-fill" style={{ width: `${gpuPct}%`, background: barColor(gpuPct) }} />
        </span>
        <span className="gpu-mini-label">{gpuPct}%</span>
        <span className="gpu-temp" style={{ color: tempColor }}>{temp}°</span>
      </div>
      {gpuzPanel}
    </div>
  );
})

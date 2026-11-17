// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import { apiGet, apiPost } from "./shared";

export type TorchDevice = {
  id: string;
  label: string;
  kind: string;
  available: boolean;
  selected: boolean;
  busy?: boolean;
  busy_owner_kind?: string;
  busy_owner_id?: string;
  index?: number;
  memory_mb?: number | null;
  allocated_mb?: number | null;
  reserved_mb?: number | null;
};

export type TorchDeviceState = {
  configured_device: string;
  selected_device: string;
  devices: TorchDevice[];
};

// ---------------------------------------------------------------------------
// Health API
// ---------------------------------------------------------------------------

export type HealthInfo = {
  status: string;
  version: string;
  build_date: string;
  disk: { total_gb: number; free_gb: number; used_pct: number } | null;
  ram: { total_gb: number; available_gb: number; used_pct: number } | null;
  gpu: { name: string; vram_total_mb: number; vram_allocated_mb: number } | null;
};

export function fetchHealth(): Promise<HealthInfo> {
  return apiGet<HealthInfo>("/health");
}

export function fetchTorchDevices(): Promise<TorchDeviceState> {
  return apiGet<TorchDeviceState>("/hardware/torch/devices");
}

export function setTorchDevice(device: string): Promise<TorchDeviceState> {
  return apiPost<TorchDeviceState>("/hardware/torch/device", { device }, { method: "PUT" });
}

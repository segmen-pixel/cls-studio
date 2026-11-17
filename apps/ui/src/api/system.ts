// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import { apiGet, apiPost } from "./shared";

export interface NetworkSettings {
  lan_access: boolean;
  current_bind_host: string;
  expected_bind_host: string;
  restart_required: boolean;
  lan_addresses: string[];
  api_token_configured: boolean;
  cvat_proxy_configured: boolean;
  annotation_proxy_configured: boolean;
}

export function fetchNetworkSettings(): Promise<NetworkSettings> {
  return apiGet<NetworkSettings>("/system/network");
}

export function updateNetworkSettings(lan_access: boolean): Promise<NetworkSettings> {
  return apiPost<NetworkSettings>("/system/network", { lan_access }, { method: "PUT" });
}

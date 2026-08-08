// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// Barrel re-export — keeps `from "./api"` / `from "../api"` working.

export {
  API_BASE,
  API_ORIGIN,
  MAX_UPLOAD_BYTES,
  assertFileSize,
  ApiError,
  parseApiError,
} from "./shared";

export * from "./projects";
// datasets.ts exports removed — the backend datasets router is dropped
// distill.ts exports removed — unused from frontend
export * from "./hardware";
export * from "./system";

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors

/**
 * App-level shared types used across hooks and components.
 */

// "bank" is import + labelling + assemble in one place. They were briefly two
// tabs, but both showed the same contact sheet, so the split only cost a
// second look at the same thumbnails -- what had to be separated was
// extraction from tier assignment in the DATA, not in the UI. "develop" stays
// reachable until the new tab has been proven on real projects.
export const BASE_TABS = ["projects", "bank", "develop", "operator"] as const;
// "annotate" / "training" are retained in the union so residual cls-studio
// references still typecheck; they are no longer reachable tabs.
export type TabId =
  | (typeof BASE_TABS)[number]
  | "annotate"
  | "training"
  | "inspect";
export type ThemeMode = "light" | "dark" | "system";

export type StartupWarning = {
  level: string;
  title: string;
  message: string;
};

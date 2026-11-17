// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import { useState } from "react";

/**
 * Global inference status shared between the tab content and StatusToast.
 *
 * cls-studio has no training pipeline: the predecessor's /train/global-status
 * and /train/runs polls are removed (they 404'd here). Only the inferStatus
 * plumbing remains; setInferStatus is retained for future live-inference use.
 */
export function useGlobalPolling() {
  const [inferStatus, setInferStatus] = useState("");

  return { inferStatus, setInferStatus } as const;
}

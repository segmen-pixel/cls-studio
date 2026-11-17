// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import { useCallback, useEffect, useMemo, useState } from "react";
import type { SettingsValues } from "../../components/SettingsDialog";

/**
 * Manages all user settings (persisted to localStorage).
 */
export function useSettings() {
  const [valRatio, setValRatio] = useState<number>(() => {
    const saved = localStorage.getItem("seg-val-ratio");
    return saved ? Number(saved) : 0.15;
  });
  const [testRatio, setTestRatio] = useState<number>(() => {
    const saved = localStorage.getItem("seg-test-ratio");
    return saved ? Number(saved) : 0.10;
  });
  const [previewStyle, setPreviewStyle] = useState<number>(() => {
    const saved = localStorage.getItem("seg_preview_style");
    return saved != null ? Number(saved) : 0;
  });
  const [descMode, setDescMode] = useState(false);

  // Persist to localStorage
  useEffect(() => { localStorage.setItem("seg-val-ratio", String(valRatio)); }, [valRatio]);
  useEffect(() => { localStorage.setItem("seg-test-ratio", String(testRatio)); }, [testRatio]);
  useEffect(() => { localStorage.setItem("seg_preview_style", String(previewStyle)); }, [previewStyle]);

  const settingsValues: SettingsValues = useMemo(() => ({
    valRatio, testRatio,
    previewStyle,
  }), [valRatio, testRatio, previewStyle]);

  const handleSettingsChange = useCallback(<K extends keyof SettingsValues>(key: K, value: SettingsValues[K]) => {
    const setters: Partial<Record<keyof SettingsValues, (v: any) => void>> = {
      valRatio: setValRatio, testRatio: setTestRatio,
      previewStyle: setPreviewStyle,
    };
    setters[key]?.(value);
  }, []);

  return {
    valRatio, testRatio, previewStyle,
    descMode, setDescMode, setPreviewStyle,
    settingsValues, handleSettingsChange,
  } as const;
}

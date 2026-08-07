// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import { useCallback, useEffect } from "react";
import { BASE_TABS, type TabId } from "../types";

/**
 * Manages tab switching and keyboard navigation.
 * activeTab / setActiveTab are owned by the caller (App) to avoid circular deps.
 */
export function useTabNavigation(
  setActiveTab: React.Dispatch<React.SetStateAction<TabId>>,
) {
  const switchTab = useCallback((tab: TabId) => {
    setActiveTab((prev) => {
      if (tab === prev) return prev;
      sessionStorage.setItem("seg-tab", tab);
      return tab;
    });
  }, [setActiveTab]);

  const navigateTab = useCallback((direction: -1 | 1) => {
    setActiveTab((prev) => {
      const allTabs: TabId[] = [...BASE_TABS];
      const idx = allTabs.indexOf(prev);
      if (idx < 0) return prev;
      const next = (idx + direction + allTabs.length) % allTabs.length;
      const nextTab = allTabs[next];
      sessionStorage.setItem("seg-tab", nextTab);
      return nextTab;
    });
  }, [setActiveTab]);

  // Keyboard: Ctrl+Arrow to navigate tabs
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if ((e.target as HTMLElement)?.isContentEditable) return;
      if (e.ctrlKey && !e.shiftKey && !e.altKey && !e.metaKey) {
        if (e.key === "ArrowLeft") { e.preventDefault(); navigateTab(-1); }
        else if (e.key === "ArrowRight") { e.preventDefault(); navigateTab(1); }
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [navigateTab]);

  return { switchTab, navigateTab } as const;
}

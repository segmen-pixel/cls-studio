// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import React from "react";
import { useI18n } from "../../i18n";
import { BASE_TABS, type TabId } from "../types";

type TabBarProps = {
  activeTab: TabId;
  switchTab: (tab: TabId) => void;
};

export default React.memo(function TabBar({
  activeTab, switchTab,
}: TabBarProps) {
  const { t } = useI18n();

  return (
    <div className="tabs-row">
      {/* The count class sizes the grid; see .tabs-fixed-N in shell.css.
          Deriving it from BASE_TABS keeps a new tab from wrapping the strip
          onto a second row. */}
      <div className={`tabs tabs-fixed tabs-fixed-${BASE_TABS.length}`}>
        {BASE_TABS.map((tab) => (
          <button
            key={tab}
            data-tutorial-step={`${tab}-tab`}
            className={activeTab === tab ? "active" : ""}
            onClick={() => switchTab(tab)}
          >
            {t(`tab.${tab}` as "tab.projects" | "tab.bank" | "tab.develop" | "tab.operator")}
          </button>
        ))}
      </div>
    </div>
  );
});

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// Inline icons in this file use SVG path data from Feather Icons
// (https://github.com/feathericons/feather), MIT licensed,
// Copyright (c) 2013-2023 Cole Bemis. Some paths are adapted.
import React, { Suspense, useCallback, useEffect, useState } from "react";
import ProjectsPanel from "./components/ProjectsPanel";
import { setIntendedProject } from "./api/cls";
import type { SettingsValues } from "./components/SettingsDialog";

// Lazy-loaded tab components (code-split per tab)
// cls-studio work tabs (replacing the predecessor's annotate/training pair)
const BankTab = React.lazy(() => import("./components/BankTab"));
const Develop = React.lazy(() => import("./components/Develop"));
const Operator = React.lazy(() => import("./components/Operator"));
const AboutDialog = React.lazy(() => import("./components/AboutDialog"));
const SignInGate = React.lazy(() => import("./components/SignInGate"));
const SettingsDialog = React.lazy(() => import("./components/SettingsDialog"));

import { useI18n } from "./i18n";

import {
  useTheme,
  useToast,
  useGlobalPolling,
  useSettings,
  useTabNavigation,
  useApiConnection,
  useTutorial,
} from "./app/hooks";
import TutorialOverlay from "./app/components/TutorialOverlay";

import AppHeader from "./app/components/AppHeader";
import TabBar from "./app/components/TabBar";
import StatusToast from "./app/components/StatusToast";
import StartupWarnings from "./app/components/StartupWarnings";

import { UNAUTHORIZED_EVENT } from "./api/shared";
import { BASE_TABS, type TabId } from "./app/types";

export default function App() {
  const { lang, t } = useI18n();

  // --- Core hooks ---
  const { themeMode, cycleTheme } = useTheme();
  const toast = useToast();
  const api = useApiConnection(toast.showToast);
  const settings = useSettings();

  // activeTab is lifted here so useTabNavigation stays decoupled from App state
  const [activeTab, setActiveTab] = useState<TabId>(() => {
    const saved = sessionStorage.getItem("seg-tab");
    if (!saved) return "projects";
    if ((BASE_TABS as readonly string[]).includes(saved)) return saved as TabId;
    return "projects";
  });

  const tabNav = useTabNavigation(setActiveTab);
  const polling = useGlobalPolling();
  const tutorial = useTutorial(activeTab, tabNav.switchTab);

  // --- Local state ---
  // Raised only by an actual 401, never by a status check at mount: a browser
  // ON the server is admitted by the loopback rule and must not be asked for
  // a token it does not need.
  const [needsAuth, setNeedsAuth] = useState(false);
  useEffect(() => {
    const raise = () => setNeedsAuth(true);
    window.addEventListener(UNAUTHORIZED_EVENT, raise);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, raise);
  }, []);

  const [aboutOpen, setAboutOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Prevent browser from opening dropped files (global safety net)
  useEffect(() => {
    const prevent = (e: DragEvent) => { e.preventDefault(); e.stopPropagation(); };
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", prevent);
    return () => { window.removeEventListener("dragover", prevent); window.removeEventListener("drop", prevent); };
  }, []);

  // Refresh projects when revisiting projects tab
  useEffect(() => {
    if (activeTab === "projects" && api.apiStatus === "connected") {
      void api.refreshProjects(api.projectsSummaryReady);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, api.apiStatus]);

  // Opening a project lands on the bank: it is where a project is built now,
  // and the legacy teach tab is on its way out.
  const openProjectWorkspace = useCallback((projectId: string, tab: "bank" | "develop" | "operator" = "bank") => {
    // Synchronously, before any panel can render or fire a request: until the
    // server confirms the new binding, mutations are refused rather than sent
    // against the project we are leaving.
    setIntendedProject(projectId);
    api.setSelectedProjectId(projectId);
    sessionStorage.setItem("seg-project", projectId);
    tabNav.switchTab(tab);
  }, [api, tabNav]);

  // The intended project follows the SELECTED one however it changed -- a tile's
  // first click selects without opening (see ProjectTile), so hooking only
  // openProjectWorkspace leaves the intent stale, and a stale intent refuses
  // every mutation against the project actually on screen. The synchronous call
  // in openProjectWorkspace stays as well: child effects run before parent
  // effects, so this alone would be one commit late for the panels below.
  useEffect(() => { setIntendedProject(api.selectedProjectId); }, [api.selectedProjectId]);

  const settingsValues: SettingsValues = settings.settingsValues;
  const handleSettingsChange = settings.handleSettingsChange;

  // --- Instant tooltips (body-level): rich data-desc in desc-mode, else the
  //     element's native title — shown immediately, with the browser title
  //     suppressed during hover so there is no delayed / double tooltip. ---
  useEffect(() => {
    let tip: HTMLDivElement | null = null;
    let activeEl: HTMLElement | null = null;
    const restore = (el: HTMLElement | null) => {
      if (el && el.dataset.nativeTitle != null) {
        el.setAttribute("title", el.dataset.nativeTitle);
        el.removeAttribute("data-native-title");
      }
    };
    const show = (e: MouseEvent) => {
      const el = (e.target as HTMLElement).closest?.("[data-desc],[title]") as HTMLElement | null;
      if (!el || el === activeEl) return;
      restore(activeEl);
      activeEl = el;
      const desc = settings.descMode ? el.getAttribute("data-desc") : null;
      const compact = !desc;
      let text = desc;
      if (!text) {
        text = el.getAttribute("title");
        if (text) { el.dataset.nativeTitle = text; el.removeAttribute("title"); }
      }
      if (!text) { activeEl = null; return; }
      if (!tip) { tip = document.createElement("div"); document.body.appendChild(tip); }
      tip.className = compact ? "desc-tooltip desc-tooltip-compact visible" : "desc-tooltip visible";
      tip.textContent = text;
      const r = el.getBoundingClientRect();
      const pos = el.getAttribute("data-desc-pos") || "top";
      // measure tooltip
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      let top: number, left: number;
      if (pos === "bottom") { top = r.bottom + 8; left = r.left + r.width / 2 - tw / 2; }
      else if (pos === "right") { top = r.top + r.height / 2 - th / 2; left = r.right + 8; }
      else if (pos === "left") { top = r.top + r.height / 2 - th / 2; left = r.left - tw - 8; }
      else { top = r.top - th - 8; left = r.left + r.width / 2 - tw / 2; }
      // clamp to viewport
      if (left < 4) left = 4;
      if (left + tw > window.innerWidth - 4) left = window.innerWidth - tw - 4;
      if (top < 4) { top = r.bottom + 8; } // flip to bottom
      if (top + th > window.innerHeight - 4) { top = r.top - th - 8; } // flip to top
      tip.style.top = `${top}px`;
      tip.style.left = `${left}px`;
    };
    const hide = (e: MouseEvent) => {
      if (!activeEl) return;
      const to = e.relatedTarget as Node | null;
      if (to && activeEl.contains(to)) return;
      if (tip) tip.classList.remove("visible");
      restore(activeEl);
      activeEl = null;
    };
    document.addEventListener("mouseover", show);
    document.addEventListener("mouseout", hide);
    return () => {
      document.removeEventListener("mouseover", show);
      document.removeEventListener("mouseout", hide);
      restore(activeEl);
      if (tip) { tip.remove(); tip = null; }
    };
  }, [settings.descMode]);

  // --- Render ---
  return (
    <div className={`app-shell tab-${activeTab}${settings.descMode ? " desc-mode" : ""}`}>
      <div className="app-top">
        <AppHeader
          currentProject={api.currentProject}
          activeTabIsProjects={activeTab === "projects"}
          themeMode={themeMode}
          cycleTheme={cycleTheme}
          descMode={settings.descMode}
          setDescMode={settings.setDescMode}
          onOpenAbout={() => { setAboutOpen(true); api.refreshHealthInfo(); }}
          onOpenSettings={() => setSettingsOpen(true)}
          onRestartTutorial={tutorial.restart}
          showToast={toast.showToast}
        />
        <TabBar
          activeTab={activeTab}
          switchTab={tabNav.switchTab}
        />
      </div>
      <main className="app-content">
        <Suspense fallback={<div className="muted" style={{ padding: 32, textAlign: "center" }}>Loading...</div>}>
        <section className={`panel ${activeTab === "annotate" ? "panel-tight" : ""}`}>
          <div className={`tab-panel ${activeTab === "projects" ? "active" : ""}`}>
            <ProjectsPanel
              projects={api.projects}
              setProjects={api.setProjects}
              selectedProjectId={api.selectedProjectId}
              setSelectedProjectId={api.setSelectedProjectId}
              projectPreviews={api.projectPreviews}
              setProjectPreviews={api.setProjectPreviews}
              projectsLoading={api.projectsLoading}
              projectsSummaryReady={api.projectsSummaryReady}
              apiStatus={api.apiStatus}
              currentProject={api.currentProject}
              currentProjectPreview={api.currentProjectPreview}
              openProjectWorkspace={openProjectWorkspace}
              showToast={toast.showToast}
            />
          </div>
          {/* key={projectId}: every workspace panel is mounted for the whole
              session and hidden with CSS (see .tab-panel in styles/shell.css),
              so switching projects left each panel holding the previous one's
              state -- its image list, its selection, its scores -- until some
              effect happened to clear it. Remounting makes "did you remember to
              reset this useState?" a question that cannot be asked, which is
              the point: the fourteenth piece of state added later cannot forget.
              The load effects stay gated on `active`, so a remounted inactive
              panel still fetches nothing until it is visited. */}
          <div className={`tab-panel ${activeTab === "bank" ? "active" : ""}`}>
            <BankTab
              key={api.selectedProjectId ?? "none"}
              projectId={api.selectedProjectId}
              active={activeTab === "bank"}
              showToast={toast.showToast}
            />
          </div>
          <div className={`tab-panel ${activeTab === "develop" ? "active" : ""}`}>
            <Develop
              key={api.selectedProjectId ?? "none"}
              projectId={api.selectedProjectId}
              active={activeTab === "develop"}
              showToast={toast.showToast}
            />
          </div>
          <div className={`tab-panel ${activeTab === "operator" ? "active" : ""}`}>
            <Operator
              key={api.selectedProjectId ?? "none"}
              projectId={api.selectedProjectId}
              active={activeTab === "operator"}
              showToast={toast.showToast}
            />
          </div>
        </section>
        </Suspense>
        <button
          className="tab-nav-arrow tab-nav-arrow-left"
          onClick={() => tabNav.navigateTab(-1)}
          aria-label="Previous tab"
          tabIndex={-1}
          data-desc={t("projects.prevTab")}
          data-desc-pos="right"
        >
          <svg width="20" height="36" viewBox="0 0 20 36" fill="none">
            <path d="M16 4 L4 18 L16 32" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        <button
          className="tab-nav-arrow tab-nav-arrow-right"
          onClick={() => tabNav.navigateTab(1)}
          aria-label="Next tab"
          tabIndex={-1}
          data-desc={t("projects.nextTab")}
          data-desc-pos="left"
        >
          <svg width="20" height="36" viewBox="0 0 20 36" fill="none">
            <path d="M4 4 L16 18 L4 32" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </main>
      {needsAuth && (
        <Suspense fallback={null}>
          {/* A reload rather than a re-fetch: every hook in the shell has
              already failed and settled, and re-running them piecemeal is
              more moving parts than the one thing that certainly works. */}
          <SignInGate onSignedIn={() => window.location.reload()} />
        </Suspense>
      )}
      <StartupWarnings
        warnings={api.startupWarnings}
        onDismiss={() => api.setStartupWarnings([])}
      />
      <AboutDialog open={aboutOpen} onClose={() => setAboutOpen(false)} healthInfo={api.healthInfo} />
      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        values={settingsValues}
        onChange={handleSettingsChange}
        showToast={toast.showToast}
        onLibraryChanged={() => window.dispatchEvent(new Event("library-changed"))}
      />
      <div className="fab-stack">
        {/* cls-studio training widgets (CloudJobWidget / NewModelsWidget /
            FloatingTrainingWidget) removed for cls-studio — no training runs. */}
        <StatusToast
          toastMsg={toast.toastMsg}
          toastCopied={toast.toastCopied}
          inferStatus={polling.inferStatus}
          onToastClick={toast.handleToastClick}
          onMouseEnter={toast.handleToastHoverEnter}
          onMouseLeave={toast.handleToastHoverLeave}
        />
      </div>
      {tutorial.active && tutorial.currentStep && (
        <TutorialOverlay
          step={tutorial.currentStep}
          stepIndex={tutorial.stepIndex}
          totalSteps={tutorial.totalSteps}
          lang={lang}
          onNext={tutorial.next}
          onBack={tutorial.back}
          onSkip={tutorial.skip}
          onChooseMode={tutorial.chooseMode}
        />
      )}
    </div>
  );
}

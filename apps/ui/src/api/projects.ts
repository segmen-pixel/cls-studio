// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import { API_BASE, apiGet, apiPost, parseApiError } from "./shared";

export type Project = {
  id: string;
  name: string;
  description?: string | null;
  memo?: string | null;
  sort_order?: number;
  tags?: string[];
  created_at: string;
  updated_at: string;
};

export type ProjectSummary = Project & {
  image_count: number;
  mask_count: number;
  /** True when the annotate dataset is empty but a bank-taught image exists. */
  has_bank_thumbnail?: boolean;
};

/** Card-thumbnail fallback for projects taught straight into a memory bank. */
export function bankThumbnailUrl(projectId: string) {
  return `${API_BASE}/projects/${projectId}/bank-thumbnail`;
}

export type ProjectPreview = { thumbUrl: string | null; imageCount: number; maskCount: number };

/** Card preview from one summary row — the ONLY place that decides which
 * thumbnail a project card shows. Every summary consumer (useApiConnection's
 * initial load, ProjectsPanel's post-mutation refreshes) must map through
 * this: the bank-thumbnail fallback was silently missing on page load
 * because this mapping used to be duplicated per call site. */
export function projectPreviewOf(s: ProjectSummary): ProjectPreview {
  return {
    thumbUrl: s.has_bank_thumbnail ? bankThumbnailUrl(s.id) : null,
    imageCount: s.image_count,
    maskCount: s.mask_count,
  };
}

export function fetchProjects(): Promise<Project[]> {
  return apiGet<Project[]>("/projects");
}

export function fetchProjectsSummary(): Promise<ProjectSummary[]> {
  return apiGet<ProjectSummary[]>("/projects/summary");
}

export function createProject(payload: { name: string; description?: string; tags?: string[] }): Promise<Project> {
  return apiPost<Project>("/projects", payload);
}

export function updateProject(id: string, payload: { name?: string; description?: string; memo?: string; tags?: string[] }): Promise<Project> {
  return apiPost<Project>(`/projects/${id}`, payload, { method: "PUT" });
}

export async function deleteProject(id: string) {
  const res = await fetch(`${API_BASE}/projects/${id}`, { method: "DELETE" });
  if (!res.ok) throw await parseApiError(res);
}

export async function reorderProjects(order: string[]): Promise<void> {
  const res = await fetch(`${API_BASE}/projects/reorder`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ order }),
  });
  if (!res.ok) throw await parseApiError(res);
}

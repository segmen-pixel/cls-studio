// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// Client for the cls-studio anomaly-detection API (bank / score / captures).
// The active project's bank is bound server-side by selectBank(); the rest of
// the calls then operate on it.
import { API_BASE, apiGet, apiPost, parseApiError } from "./shared";

export type Tier = "normal" | "critical" | "negative";

/** Denominator floor of the α / β terms: `alpha / (BOOST_FLOOR + d)`.
 *
 * Mirrors clscore.scoring.BOOST_FLOOR and MUST stay equal to it. The server
 * bounded the term (it used to be `alpha / (1e-6 + d)`, which reached 8.4e8
 * on an image identical to a taught NG); the client kept composing with the
 * old epsilon, so the separation check and the verdict were computing two
 * different numbers and a threshold picked on one did not mean the same
 * thing on the other. */
export const BOOST_FLOOR = 1;

/** Slider ceiling for α and β. Bounding the term changed what α *is*: it is
 * now the MAXIMUM BOOST IN SCORE UNITS rather than an unbounded coefficient,
 * so against a threshold near 59 the working range is roughly 50-100 and the
 * old ceiling of 1000 is an order of magnitude too generous to aim with. */
export const ALPHA_MAX = 200;

export type BankState = {
  normal: number;
  critical: number;
  negative: number;
  critical_by_label: Record<string, number>;
  negative_by_label: Record<string, number>;
  dim: number;
};

export type BankInfo = {
  id: string;
  name: string;
  images: Record<string, number>;
};

export type BankList = {
  banks: BankInfo[];
  active_bank_id: string | null;
};

export type SelectResult = {
  project_id: string;
  bank_id: string;
  bank_dir: string;
  device: string;
  bank: BankState;
  banks: BankInfo[];
};

export type AppendResult = {
  tier: Tier;
  label: string;
  appended_patches: number;
  bank: BankState;
};

export type ScoreResult = {
  max_score: number;
  mean_score: number;
  p99_score: number;
  /** Mean of the k hottest patches — same statistic as the separation check. */
  topk_score: number;
  /** Critical exemplar rows the α term used (0 = α off or no exemplars). */
  n_exemplar_rows: number;
  heatmap_png_base64: string;
  /** Downscaled JPEG of the upload (server-transcoded — TIFF etc. render too). */
  original_jpeg_base64?: string;
  /** Id of the persisted inspection-log entry ("" if persistence was off/failed). */
  inspection_id?: string;
  critical_attribution: Record<string, number>;
  negative_attribution: Record<string, number>;
  timings: Record<string, number>;
};

/** One defect-mark rectangle, normalized (0..1) to the original image size. */
export type AnnotationRect = { x: number; y: number; w: number; h: number };

export type BankImage = {
  name: string;
  tier: Tier;
  label: string;
  patches: number;
  url: string;
  /** Defect marks previously saved for this image (critical tier only). */
  annotations?: AnnotationRect[];
};
export type ProjectionMode = "normal" | "anomaly" | "empty";
export type ProjectionPoint = { tier: Tier; label: string; x: number; y: number; image?: string; score?: number | null };

export type StoredImageEval = {
  name: string;
  tier: Tier;
  label: string;
  patches: number;
  score_max: number;
  score_p99: number;
  score_mean: number;
  /** Patch scores sorted descending, truncated to 256 — any top-k stat is client-computable. */
  top_scores: number[];
  /** Local row indices aligned with top_scores (absent in caches written before this field). */
  top_indices?: number[];
};
export type ProjectionAxisInfo = {
  mode: "normal" | "anomaly";
  pc1_pct: number | null;
  pc2_pct: number | null;
  pc1_contrast: number | null;
  pc2_contrast: number | null;
  n_normal: number;
  n_ng: number;
  alpha: number | null;
};
export type ProjectionResponse = {
  mode: ProjectionMode;
  granularity: "patch" | "image";
  /** Points force-included from cached per-image top patches (patch granularity only). */
  guaranteed: number;
  axis_info: ProjectionAxisInfo | null;
  points: ProjectionPoint[];
  total: Record<string, number>;
  sampled: Record<string, number>;
};

// The (project, bank) this client believes is active, refreshed from every
// SelectResult. Sent as X-Bank-Binding on mutating / scoring requests: the
// server operates on ONE process-global active bank shared by every LAN
// client, and 409s when another client re-bound it — without this a teach
// or delete would silently land in the other client's bank.
let activeBinding: string | null = null;

function rememberBinding(r: SelectResult): SelectResult {
  activeBinding = `${r.project_id}/${r.bank_id}`;
  return r;
}

function bindingHeaders(): Record<string, string> {
  return activeBinding ? { "X-Bank-Binding": activeBinding } : {};
}

// Thin wrappers over the shared JSON helpers that attach the X-Bank-Binding
// header (see activeBinding above) to every mutating request.
function jpost<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return apiPost<T>(path, body, { headers: bindingHeaders(), signal });
}

function jget<T>(path: string, signal?: AbortSignal): Promise<T> {
  return apiGet<T>(path, undefined, signal);
}

function jput<T>(path: string, body: unknown): Promise<T> {
  return apiPost<T>(path, body, { method: "PUT", headers: bindingHeaders() });
}

/** Bind a project (and optionally one of its banks) as the active one. */
export function selectBank(projectId: string, bankId?: string): Promise<SelectResult> {
  return jpost<SelectResult>("/bank/select", { project_id: projectId, bank_id: bankId ?? null })
    .then(rememberBinding);
}

/** List the memory banks in the active project. */
export function listBanks(): Promise<BankList> {
  return jget<BankList>("/banks");
}

/** Create a new empty bank in the active project and select it. */
export function createBank(name: string): Promise<SelectResult> {
  return jpost<SelectResult>("/banks/create", { name }).then(rememberBinding);
}

/** Switch to a different bank within the active project. */
export function selectBankId(bankId: string): Promise<SelectResult> {
  return jpost<SelectResult>("/banks/select", { bank_id: bankId }).then(rememberBinding);
}

/** Delete a bank from the active project; the response's active bank is the
 *  one selected afterwards (the next remaining bank, or a fresh empty one). */
export function deleteBank(bankId: string): Promise<SelectResult> {
  return jpost<SelectResult>("/banks/delete", { bank_id: bankId }).then(rememberBinding);
}

/** Absolute URL for a bank image's thumbnail (from BankImage.url).
 *
 *  `url` arrives percent-encoded and may already carry `?id=` (the store
 *  entry, so duplicate filenames resolve exactly), so it is concatenated
 *  verbatim — encoding it again here would double-encode every `%`. Pass the
 *  bank id as `bust` rather than appending `?b=` at the call site: a second
 *  `?` would make the whole query string one opaque parameter value. */
export function imageUrl(url: string, bust?: string): string {
  const base = `${API_BASE}${url}`;
  if (!bust) return base;
  return `${base}${url.includes("?") ? "&" : "?"}b=${encodeURIComponent(bust)}`;
}

export function fetchBank(): Promise<BankState> {
  return jget<BankState>("/bank");
}

export function fetchProjection(opts: { mode?: "auto" | "normal" | "anomaly"; maxPointsPerTier?: number; alpha?: number; withScores?: boolean; granularity?: "patch" | "image"; guaranteeTop?: number } = {}): Promise<ProjectionResponse> {
  const qs = new URLSearchParams({
    mode: opts.mode ?? "auto",
    max_points_per_tier: String(opts.maxPointsPerTier ?? 800),
    alpha: String(opts.alpha ?? 1.0),
    with_scores: String(opts.withScores ?? false),
    granularity: opts.granularity ?? "patch",
    guarantee_top: String(opts.guaranteeTop ?? 10),
  });
  return jget<ProjectionResponse>(`/bank/projection?${qs.toString()}`);
}

/** Image-level anomaly stats from the image's stored patch rows (no forward). */
export function evaluateBankImage(
  tier: Tier, name: string, label = "",
  group?: Grouping, signal?: AbortSignal,
): Promise<StoredImageEval> {
  const qs = new URLSearchParams({ tier, name, label });
  // Sent only when it changes the answer: a grouped run bypasses the server's
  // eval cache, so asking for the default would cost the cache for nothing.
  if (group && group.mode !== "none") {
    qs.set("group_mode", group.mode);
    qs.set("group_sep", group.sep);
    qs.set("group_fields", String(group.fields));
  }
  return jget<StoredImageEval>(`/bank/images/evaluate?${qs.toString()}`, signal);
}

/** Server-cached evals for the active bank's current contents ([] if stale). */
export function fetchCachedEvaluations(): Promise<StoredImageEval[]> {
  return jget<StoredImageEval[]>("/bank/evaluation/cached");
}

export type AnnotateResult = {
  tier: Tier;
  label: string;
  name: string;
  rows_marked: number;
  rects: AnnotationRect[];
};

/** Save an NG image's defect marks; the covered bank rows become α exemplars. */
export function annotateBankImage(
  tier: Tier, name: string, label: string, rects: AnnotationRect[],
): Promise<AnnotateResult> {
  return jpost<AnnotateResult>("/bank/images/annotate", { tier, name, label, rects });
}

export type ImageCmin = { name: string; tier: Tier; label: string; top_cmin: number[] };

/** Per-image min distances (aligned with top_scores) to the NG exemplar rows. */
export function fetchEvalCmin(): Promise<ImageCmin[]> {
  return jget<ImageCmin[]>("/bank/evaluation/cmin");
}

/** Verdict recipe persisted in the bank dir — travels with bank exports. */
export type RuntimeConfig = {
  version: number;
  metric: "topk_mean";
  topk: number;
  k: number;
  alpha: number;
  beta: number;
  exemplar_alpha: boolean;
  threshold: number | null;
  saved_at: string;
  /** Content fingerprint of the bank the recipe was tuned on (server-stamped). */
  bank_fingerprint: string;
  /** Set on read: the bank changed since this recipe was saved. */
  stale?: boolean;
  /** Runtime memory-bank size budget (operational knob, not a tuning field). */
  bank_capacity?: BankCapacity;
};

export type BankCapacity = "small" | "medium" | "large";

export type BankCapacityInfo = {
  capacity: BankCapacity;
  /** Max total normal patches allowed at this tier. */
  ceiling: number;
  /** Current normal-bank patches. */
  normal: number;
  /** normal / ceiling * 100, clamped to 100. */
  pct: number;
  /** Rough resident VRAM of the normal bank at this fill (fp16), in MB. */
  est_vram_mb: number;
  /** Current critical + negative patches. Resident, but uncapped and outside the ceiling. */
  labeled: number;
  /** Rough resident VRAM of ALL tiers at this fill (fp16), in MB. */
  est_vram_total_mb: number;
};

/** The active bank's size budget and how full the normal tier is against it. */
export function fetchBankCapacity(): Promise<BankCapacityInfo> {
  return jget<BankCapacityInfo>("/bank/capacity");
}

/** Set the active bank's size budget. Takes effect on the next teach;
 * existing patches are never evicted. */
export function setBankCapacity(capacity: BankCapacity): Promise<BankCapacityInfo> {
  return jput<BankCapacityInfo>("/bank/capacity", { capacity });
}

/** The active bank's saved verdict recipe, or null if never saved. */
export function fetchRuntimeConfig(): Promise<RuntimeConfig | null> {
  return jget<RuntimeConfig | null>("/bank/runtime-config");
}

/** Persist the verdict recipe into the bank directory. Server stamps
 * saved_at / bank_fingerprint, so callers only pass the tuning fields. */
export function saveRuntimeConfig(
  cfg: Omit<RuntimeConfig, "version" | "metric" | "saved_at" | "bank_fingerprint" | "stale"> & Partial<RuntimeConfig>,
): Promise<RuntimeConfig> {
  return jput<RuntimeConfig>("/bank/runtime-config", cfg);
}

/** Download URL for the active bank as a one-file package (settings included). */
export function bankExportUrl(includeImages = true): string {
  return `${API_BASE}/bank/export?include_images=${includeImages}`;
}

/** Load an exported bank package as a new bank of the active project. */
export async function importBank(file: File, name = ""): Promise<SelectResult> {
  const fd = new FormData();
  fd.append("archive", file, file.name);
  fd.append("name", name);
  const res = await fetch(`${API_BASE}/banks/import`, {
    method: "POST", body: fd, headers: bindingHeaders(),
  });
  if (!res.ok) throw await parseApiError(res);
  return (res.json() as Promise<SelectResult>).then(rememberBinding);
}

export function clearTier(tier: Tier): Promise<BankState> {
  return jpost<BankState>(`/bank/clear/${tier}`, {});
}

/** Teach one image into a tier (extracts patch tokens server-side). */
export async function appendImage(tier: Tier, file: File, label = ""): Promise<AppendResult> {
  const fd = new FormData();
  fd.append("image", file, file.name);
  fd.append("label", label);
  const res = await fetch(`${API_BASE}/bank/append/${tier}`, {
    method: "POST", body: fd, headers: bindingHeaders(),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json() as Promise<AppendResult>;
}

/** Teach many same-tier images in one request; the server batches all their
 *  windows through the backbone together (full max_batch forwards) and saves
 *  once — far higher GPU utilisation than one append per image. */
export async function teachBatch(tier: Tier, files: File[], label = ""): Promise<AppendResult> {
  const fd = new FormData();
  for (const f of files) fd.append("images", f, f.name);
  fd.append("label", label);
  const res = await fetch(`${API_BASE}/bank/append_batch/${tier}`, {
    method: "POST", body: fd, headers: bindingHeaders(),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json() as Promise<AppendResult>;
}

export async function scoreImage(
  file: File,
  opts: { alpha?: number; beta?: number; k?: number; topk?: number; hmLo?: number; hmHi?: number; recordHits?: boolean; signal?: AbortSignal } = {},
): Promise<ScoreResult> {
  const qs = new URLSearchParams({
    alpha: String(opts.alpha ?? 0),
    beta: String(opts.beta ?? 0),
    k: String(opts.k ?? 5),
    topk: String(opts.topk ?? 10),
  });
  // Absolute heatmap anchors (OK median → raw threshold): switches the
  // server to the anomaly-focus overlay instead of per-image percentiles.
  if (opts.hmLo != null && opts.hmHi != null) {
    qs.set("hm_lo", String(opts.hmLo));
    qs.set("hm_hi", String(opts.hmHi));
  }
  // Heatmap previews of already-taught images are not real inspections —
  // callers pass false so freshness / hit counters stay untouched.
  if (opts.recordHits === false) qs.set("record_hits", "false");
  const fd = new FormData();
  fd.append("image", file, file.name);
  const res = await fetch(`${API_BASE}/score?${qs}`, {
    method: "POST", body: fd, headers: bindingHeaders(), signal: opts.signal,
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json() as Promise<ScoreResult>;
}

/** Image-level top-k mean of the composite score `raw + α/(1+cmin)` from a
 * cached eval — the shared metric between the Teach tab's separation check
 * and the Operator verdict, so a threshold picked on one applies to the other.
 * That only holds while the denominator matches the server's; see BOOST_FLOOR. */
export function topkComposite(r: StoredImageEval, k: number, alpha: number, cmin?: number[]): number {
  const ts = r.top_scores ?? [];
  if (!ts.length) return r.score_max;
  let comp = ts;
  if (alpha > 0 && cmin && cmin.length === ts.length) {
    comp = ts.map((v, i) => v + alpha / (BOOST_FLOOR + cmin[i])).sort((a, b) => b - a);
  }
  const kk = Math.min(k, comp.length);
  let s = 0;
  for (let i = 0; i < kk; i++) s += comp[i];
  return s / kk;
}

export function fetchBankImages(): Promise<{ images: BankImage[] }> {
  return jget<{ images: BankImage[] }>("/bank/images");
}

// ---- server-side staging (upload-on-drop) ----------------------------------
// Dropped files land in <bank>/_staging/ immediately, so a reload / crash
// never discards them; labels live server-side too and the list is restored
// on mount. Teach consumes only the files that actually reached the bank.

export type StagedItem = { name: string; tier: Tier | null };

export function fetchStaging(): Promise<{ items: StagedItem[] }> {
  return jget<{ items: StagedItem[] }>("/bank/staging");
}

export async function uploadStaging(files: File[]): Promise<{ items: StagedItem[] }> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  const res = await fetch(`${API_BASE}/bank/staging/upload`, {
    method: "POST", body: fd, headers: bindingHeaders(),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json() as Promise<{ items: StagedItem[] }>;
}

/** Absolute URL of a staged file (browser-renderable formats only). */
export function stagingFileUrl(name: string): string {
  return `${API_BASE}/bank/staging/file/${encodeURIComponent(name)}`;
}

export function labelStaging(names: string[], tier: Tier | null): Promise<{ items: StagedItem[] }> {
  return jpost<{ items: StagedItem[] }>("/bank/staging/label", { names, tier });
}

export function deleteStaging(names: string[]): Promise<{ items: StagedItem[] }> {
  return jpost<{ items: StagedItem[] }>("/bank/staging/delete", { names });
}

export type StagedTeachResult = {
  tier: Tier;
  label: string;
  appended_patches: number;
  /** Staged names that reached the bank (consumed from staging). */
  taught: string[];
  /** Staged names that did NOT reach the bank (still staged — retry). */
  failed: string[];
  bank: BankState;
};

export function teachStaged(names: string[], tier: Tier, label = "", signal?: AbortSignal): Promise<StagedTeachResult> {
  return jpost<StagedTeachResult>("/bank/staging/teach", { names, tier, label }, signal);
}

/** Remove taught images (and their patches) from a tier. */
export function deleteBankImages(tier: Tier, names: string[], label?: string): Promise<BankState> {
  return jpost<BankState>("/bank/images/delete", { tier, names, label: label ?? null });
}

// ---- persisted inspection results (Operator tab reload survival) ----------

export type InspectionEntry = {
  id: string;
  name: string;
  ts: string;
  topk_score: number;
  max_score: number;
  p99_score: number;
  n_exemplar_rows: number;
  alpha: number;
  server_ms: number;
  orig: string;
  heat: string;
};

/** Inspection results persisted server-side for the active project, oldest first. */
export function fetchInspections(): Promise<{ entries: InspectionEntry[] }> {
  return jget<{ entries: InspectionEntry[] }>("/inspections");
}

/** Absolute URL of a stored inspection preview / heatmap file. */
export function inspectionFileUrl(name: string): string {
  return `${API_BASE}/inspections/file/${encodeURIComponent(name)}`;
}

/** Drop the persisted inspection log (Operator tab's clear). */
export async function clearInspections(): Promise<void> {
  const res = await fetch(`${API_BASE}/inspections`, { method: "DELETE" });
  if (!res.ok) throw await parseApiError(res);
}

/** Drop one persisted inspection entry (Operator list's Delete key). */
export async function deleteInspection(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/inspections/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw await parseApiError(res);
}

// ---------------------------------------------------------------------------
// Bank compression settings (int8 quantisation / IVF routing)
// ---------------------------------------------------------------------------

export type CompressionSettings = {
  int8: boolean;
  ivf: boolean;
  ivf_nprobe: number;
};

/** Server-wide bank-compression settings (both transforms default to on). */
export function fetchCompressionSettings(): Promise<CompressionSettings> {
  return jget<CompressionSettings>("/system/compression");
}

/** Persist compression settings; applies to the next score/eval, no restart. */
export function updateCompressionSettings(
  settings: CompressionSettings,
): Promise<CompressionSettings> {
  return apiPost<CompressionSettings>("/system/compression", settings, { method: "PUT" });
}


// ---- feature store / label sets --------------------------------------------
// The split shape of teaching: ingest once (expensive), assign a tier as often
// as the operator changes their mind (free), assemble the bank on demand.
// See clscore.store for why the two halves are separate at all.

export type StoreImageInfo = {
  id: string;
  name: string;
  rows: number;
  /** Patches the image's SW grid yields — differs from `rows` for a capped image. */
  grid_rows: number;
  width: number;
  height: number;
  has_image: boolean;
  /** "" when the image is ingested but not assigned to a tier yet. */
  tier: "" | Tier;
  label: string;
  severity: number;
  marks: number;
  /** The rectangles the marks were drawn from, normalised to the image. */
  rects: AnnotationRect[];
  group: string;
};

export type StoreListResponse = {
  images: StoreImageInfo[];
  total_rows: number;
  dim: number;
  model: string;
  labelset_id: string;
};

export type AssemblyStatus = {
  labelset_id: string;
  labelset_name: string;
  store_images: number;
  store_rows: number;
  assigned: number;
  unassigned: number;
  counts: Record<string, number>;
  fingerprint: string;
  /** Assignments changed since the bank was last assembled. */
  stale: boolean;
  assembled_from: string;
  migrated: boolean;
};

export type IngestResult = { ingested: number; rows: number; failed: string[]; status: AssemblyStatus };
export type MigrationResult = { images: number; rows: number; labelset_id: string; status: AssemblyStatus };
export type LabelSetInfo = {
  id: string; name: string; description: string;
  counts: Record<string, number>; updated_at: string;
};
export type LabelSetList = { labelsets: LabelSetInfo[]; active_id: string };
export type AssignResult = { changed: number; status: AssemblyStatus };
export type MarkResult = { id: string; marks: number; status: AssemblyStatus };
export type AssembleResult = { bank: BankState; status: AssemblyStatus };
export type GroupPreview = {
  mode: string;
  groups: Record<string, string[]>;
  grouped: number;
  ungrouped: number;
};

export type GroupMode = "none" | "datetime" | "prefix" | "manual";

/** How the separation sweep holds images out. "none" excludes only the image
 *  being scored, which reads optimistically while the bank still holds its
 *  twins from the same lot; every other mode holds the whole group out. */
export type Grouping = { mode: GroupMode; sep: string; fields: number };

export function fetchStore(): Promise<StoreListResponse> {
  return jget<StoreListResponse>("/store");
}

/** URL for one store entry's image.
 *
 * Defaults to a downscaled rendition, not the original: the store keeps the
 * source image because a re-ingest needs it, and pointing a picker at those
 * pulls hundreds of megabytes to fill a list of 40px rows. */
export function storeImageUrl(id: string, size: "thumb" | "preview" | "full" = "preview"): string {
  return `${API_BASE}/store/image/${encodeURIComponent(id)}?size=${size}`;
}

/** Extract features for images into the store. No tier: an ingested image
 *  starts unassigned and waits in the labelling grid for a decision. */
export async function ingestImages(files: File[], tier: Tier | "" = ""): Promise<IngestResult> {
  const fd = new FormData();
  for (const f of files) fd.append("images", f, f.name);
  fd.append("tier", tier);
  const res = await fetch(`${API_BASE}/store/ingest`, {
    method: "POST", body: fd, headers: bindingHeaders(),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json() as Promise<IngestResult>;
}

/** Ingest every image inside a .zip. One request per archive: the server
 *  streams it to disk and reads members one at a time, so the browser never
 *  unpacks anything. Same result shape as ingestImages. */
export async function ingestZip(file: File, tier: Tier | "" = ""): Promise<IngestResult> {
  const fd = new FormData();
  fd.append("archive", file, file.name);
  fd.append("tier", tier);
  const res = await fetch(`${API_BASE}/store/ingest_zip`, {
    method: "POST", body: fd, headers: bindingHeaders(),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json() as Promise<IngestResult>;
}

export function deleteFromStore(ids: string[]): Promise<StoreListResponse> {
  return jpost<StoreListResponse>("/store/delete", { ids });
}

/** Carve this bank's existing rows into a store + "standard" label set.
 *  Nothing is re-extracted and nothing the bank holds is modified. */
export function migrateStore(): Promise<MigrationResult> {
  return jpost<MigrationResult>("/store/migrate", {});
}

export function fetchLabelSets(): Promise<LabelSetList> {
  return jget<LabelSetList>("/labelsets");
}

export function createLabelSet(name: string, copyActive = true): Promise<LabelSetList> {
  return jpost<LabelSetList>("/labelsets/create", { name, copy_active: copyActive });
}

export function selectLabelSet(id: string): Promise<LabelSetList> {
  return jpost<LabelSetList>("/labelsets/select", { id });
}

export function deleteLabelSet(id: string): Promise<LabelSetList> {
  return jpost<LabelSetList>("/labelsets/delete", { id });
}

export function assignImages(
  ids: string[], tier: Tier, label = "", severity = 2,
): Promise<AssignResult> {
  return jpost<AssignResult>("/labelsets/assign", { ids, tier, label, severity });
}

export function unassignImages(ids: string[]): Promise<AssignResult> {
  return jpost<AssignResult>("/labelsets/unassign", { ids });
}

export function markStoreImage(id: string, rects: AnnotationRect[]): Promise<MarkResult> {
  return jpost<MarkResult>("/labelsets/mark", { id, rects });
}

export function setStoreGroup(ids: string[], group: string): Promise<StoreListResponse> {
  return jpost<StoreListResponse>("/store/group", { ids, group });
}

export function fetchGroupPreview(
  mode: GroupMode, sep = "_", fields = 1,
): Promise<GroupPreview> {
  const q = new URLSearchParams({ mode, sep, fields: String(fields) });
  return jget<GroupPreview>(`/store/groups?${q.toString()}`);
}

export function fetchAssemblyStatus(): Promise<AssemblyStatus> {
  return jget<AssemblyStatus>("/bank/assembly");
}

/** Rebuild the bank from the store + active label set. Explicit on purpose:
 *  re-labelling is free and repeated, rebuilding is not. */
export function assembleBank(): Promise<AssembleResult> {
  return jpost<AssembleResult>("/bank/assemble", {});
}

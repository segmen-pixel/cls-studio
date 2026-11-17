// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
export const API_BASE = window.location.origin + "/api/v1";
// v2 streaming surface (/v2/*, /ws/v2/*) is mounted at the root, not under /api/v1.
export const API_ORIGIN = window.location.origin;

export const MAX_UPLOAD_BYTES = 200 * 1024 * 1024; // 200 MB — matches server limit

export function assertFileSize(file: File, max: number = MAX_UPLOAD_BYTES): void {
  if (file.size > max) {
    const maxMB = Math.round(max / (1024 * 1024));
    throw new Error(`File "${file.name}" is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed: ${maxMB} MB.`);
  }
}

// ---------------------------------------------------------------------------
// Structured API Error
// ---------------------------------------------------------------------------

/** Structured error from the backend's unified error response. */
export class ApiError extends Error {
  /** Error code (e.g. "CLS-3004") or null for legacy responses */
  readonly code: string | null;
  /** HTTP status code */
  readonly status: number;
  /** Correlation ID for log tracing */
  readonly correlationId: string | null;
  /** Hint from backend (user-actionable suggestion) */
  readonly hint: string | null;
  /** Raw response body */
  readonly raw: string;

  constructor(
    status: number,
    message: string,
    opts: { code?: string | null; correlationId?: string | null; hint?: string | null; raw?: string } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = opts.code ?? null;
    this.correlationId = opts.correlationId ?? null;
    this.hint = opts.hint ?? null;
    this.raw = opts.raw ?? "";
  }

  /** True for transient errors that may succeed on retry */
  get isTransient(): boolean {
    return this.status === 0 || this.status === 408 || this.status === 429
      || this.status === 502 || this.status === 503 || this.status === 504;
  }

  /** User-facing display: code + message */
  get displayMessage(): string {
    const parts: string[] = [];
    if (this.code) parts.push(`[${this.code}]`);
    parts.push(this.message);
    if (this.hint) parts.push(`— ${this.hint}`);
    return parts.join(" ");
  }
}

/** Broadcast when the server answers 401.
 *
 * Every API call funnels through parseApiError, and a browser that reached a
 * LAN-bound server without a session gets 401 on all of them. The shell
 * listens for this and raises the sign-in gate; without it the app sat on a
 * spinner that never resolved, which reads as a dead server. */
export const UNAUTHORIZED_EVENT = "cls-unauthorized";

/**
 * Parse a non-ok Response into an ApiError.
 * Handles both new structured format `{error: {code, message, ...}}`
 * and legacy format `{detail: "..."}`.
 */
export async function parseApiError(res: Response): Promise<ApiError> {
  if (res.status === 401) window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
  const raw = await res.text();
  let message = raw || res.statusText;
  let code: string | null = null;
  let correlationId: string | null = null;
  let hint: string | null = null;

  try {
    const parsed = JSON.parse(raw);
    if (parsed.error) {
      // New structured format
      code = parsed.error.code ?? null;
      message = parsed.error.message ?? message;
      correlationId = parsed.error.correlation_id ?? null;
      hint = parsed.error.hint ?? null;
    } else if (parsed.detail) {
      // Legacy FastAPI format
      message = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
    }
  } catch {
    // Not JSON — use raw text as message
  }

  return new ApiError(res.status, message, { code, correlationId, hint, raw });
}

// ---------------------------------------------------------------------------
// Small JSON request helpers
// ---------------------------------------------------------------------------

export type JsonRequestOpts = {
  /** HTTP verb; "POST" unless overridden. */
  method?: "POST" | "PUT";
  /** Extra request headers (e.g. X-Bank-Binding). */
  headers?: Record<string, string>;
  /** Abort the in-flight request. Cancel buttons pass one so the wait ends at
   * the click instead of at the next loop iteration. */
  signal?: AbortSignal;
};

/** True for the rejection `fetch` produces when its AbortSignal fires.
 * Callers use it to tell "the user cancelled" apart from "the request failed":
 * an abort must not raise a toast or mark the item as errored. */
export function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

/** GET `${API_BASE}${path}` and parse the JSON response.
 * Throws ApiError (ANL codes preserved) on non-2xx. */
export async function apiGet<T>(path: string, headers?: Record<string, string>, signal?: AbortSignal): Promise<T> {
  const init: RequestInit = {};
  if (headers) init.headers = headers;
  if (signal) init.signal = signal;
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) throw await parseApiError(res);
  return res.json() as Promise<T>;
}

/** POST (or PUT) a JSON body to `${API_BASE}${path}` and parse the JSON
 * response. Pass `body === undefined` for a body-less request (no
 * Content-Type header is sent, matching the previous per-module fetches).
 * Throws ApiError (ANL codes preserved) on non-2xx. */
export async function apiPost<T>(path: string, body?: unknown, opts: JsonRequestOpts = {}): Promise<T> {
  const init: RequestInit = { method: opts.method ?? "POST" };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json", ...(opts.headers ?? {}) };
    init.body = JSON.stringify(body);
  } else if (opts.headers) {
    init.headers = opts.headers;
  }
  if (opts.signal) init.signal = opts.signal;
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) throw await parseApiError(res);
  return res.json() as Promise<T>;
}

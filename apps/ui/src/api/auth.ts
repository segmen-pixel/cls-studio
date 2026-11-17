// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// Browser sign-in for a server bound to the LAN.
//
// A non-loopback bind refuses every state-changing call without a token, which
// is the point — but the UI knew nothing about it, so a browser on the LAN got
// 401 on every request and sat on "connecting to the server" forever, with the
// real problem ("you are not signed in") never said out loud.
import { apiGet, apiPost } from "./shared";

export type AuthStatus = {
  /** The server is bound somewhere that requires a token. */
  token_required: boolean;
  /** This browser has a valid session cookie. */
  authenticated: boolean;
};

/** Readable without a session — the gate could not open otherwise. */
export function fetchAuthStatus(): Promise<AuthStatus> {
  return apiGet<AuthStatus>("/auth/status");
}

/** Exchange the shared token for a session cookie. Throws on a bad token. */
export function signIn(token: string): Promise<AuthStatus> {
  return apiPost<AuthStatus>("/auth/session", { token });
}


/** The shared token, readable only by a browser running ON the server: the
 *  route requires a loopback peer AND a same-origin request, so a page served
 *  from the LAN address cannot read it even in the operator's own browser.
 *  Anywhere else the server answers 403 — "not available here" is the normal
 *  answer, not a failure, so callers are expected to swallow it. */
export async function fetchLanToken(): Promise<string> {
  const r = await apiGet<{ token: string }>("/auth/token");
  return r.token ?? "";
}

export function signOut(): Promise<AuthStatus> {
  return apiPost<AuthStatus>("/auth/logout", {});
}

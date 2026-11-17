# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
from __future__ import annotations

import hashlib as _hashlib
import hmac as _hmac
import os as _os
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs
from urllib.parse import urlsplit as _urlsplit

from fastapi import HTTPException, UploadFile

from .config import MAX_UPLOAD_BYTES, MAX_UPLOAD_FILES, MAX_UPLOAD_TOTAL_BYTES


def _format_max_size(nbytes: int) -> str:
    """Render a byte cap for 413 details without misleading truncation.

    ``nbytes // (1024*1024)`` floors, so a 512 KiB cap rendered as
    "max 0 MB" and a 1.9 MiB cap as "max 1 MB". Report exact bytes below
    1 MiB and keep one decimal above it; whole-MiB caps (every default and
    every value the MB-denominated env vars can produce) render exactly as
    before.
    """
    mib = 1024 * 1024
    if nbytes < mib:
        return f"{nbytes} bytes"
    return f"{f'{nbytes / mib:.1f}'.removesuffix('.0')} MB"


def _sanitize_filename(name: str) -> str:
    """Strip directory components and leading/trailing whitespace from a user-supplied filename."""
    return Path(name).name.strip()


def _safe_child(parent: Path, child_name: str) -> Path:
    """Resolve child_name under parent and ensure it stays within parent."""
    resolved = (parent / child_name).resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise HTTPException(status_code=400, detail="invalid path")
    return resolved


def _safe_dir(base: Path, user_path: str) -> Path:
    """Validate that user_path resolves inside base."""
    resolved = Path(user_path).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise HTTPException(status_code=400, detail="invalid directory path")
    return resolved


async def _read_upload(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read an uploaded file with a streaming size cap to prevent DoS."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"file too large (max {_format_max_size(max_bytes)})",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _stream_upload_to_disk(file: UploadFile, dest: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    """Stream uploaded file directly to disk. Returns bytes written."""
    import os
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    total = 0
    try:
        while True:
            chunk = await file.read(262144)  # 256 KB chunks
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                os.close(tmp_fd)
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"file too large (max {_format_max_size(max_bytes)})",
                )
            os.write(tmp_fd, chunk)
        os.close(tmp_fd)
        Path(tmp_path).replace(dest)
    except HTTPException:
        raise
    except Exception:
        try:
            os.close(tmp_fd)
        except OSError:
            pass
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return total


def _check_upload_batch(count: int, total_bytes: int) -> None:
    """Aggregate guard for multi-file uploads — same 413 shape as _read_upload.

    MAX_UPLOAD_BYTES bounds ONE part only, so an N-part request is otherwise
    unbounded. Call once with the part count before reading anything, then
    again after each part with the running byte total.
    """
    if count > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"too many files in one request (max {MAX_UPLOAD_FILES})",
        )
    if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload batch too large (max {_format_max_size(MAX_UPLOAD_TOTAL_BYTES)} total)",
        )


# Archives get their own guards. The bank package, the project package and the
# store's zip ingest all read a zip a client uploaded, and all three need the
# same two answers before a single member is read: does the central directory
# claim an implausible expansion, and does any member name try to escape.
ARCHIVE_EXPANSION_FLOOR_BYTES = 64 * 1024 * 1024


def _check_archive_bounds(zf: zipfile.ZipFile, uploaded_bytes: int) -> None:
    """Zip-bomb guard, as a RATIO rather than an absolute size.

    Our own packages are written STORED, so a legitimate one expands ~1x; 8x
    still leaves room for an archive someone recompressed on the way. The floor
    keeps small archives out of the guard entirely.
    """
    declared_total = sum(info.file_size for info in zf.infolist())
    if declared_total > max(8 * uploaded_bytes, ARCHIVE_EXPANSION_FLOOR_BYTES):
        raise HTTPException(
            status_code=413,
            detail=(
                f"archive expands to {declared_total // (1024*1024)} MB "
                f"from {uploaded_bytes // (1024*1024)} MB — refusing to extract"
            ),
        )


def _check_archive_paths(names: Iterable[str]) -> None:
    """Reject absolute, parent-escaping, or drive-lettered member names.

    PurePosixPath is the right reader for zip member names. The ``":" in
    parts[0]`` arm catches a Windows-authored ``C:/...`` member, whose
    backslashes collapse into one component that ``is_absolute()`` misses.
    """
    for n in names:
        pp = PurePosixPath(n)
        if pp.is_absolute() or ".." in pp.parts or (pp.parts and ":" in pp.parts[0]):
            raise HTTPException(status_code=422, detail=f"unsafe path in archive: {n}")


def _read_zip_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo,
                     max_bytes: int | None = None) -> bytes:
    """Read one member with the same per-part cap a multipart upload gets.

    ``_check_archive_bounds`` trusts the central directory, and ``zipfile``
    does not truncate at ``file_size`` — it reads the compressed stream to its
    end and only then fails the CRC, by which time the bytes are already in
    RAM. This is the bound that survives a lying header.

    The cap is read at call time rather than bound as a default argument: a
    default would freeze the value at import and quietly ignore both a test's
    monkeypatch and any later reconfiguration.
    """
    cap = MAX_UPLOAD_BYTES if max_bytes is None else max_bytes
    chunks: list[bytes] = []
    total = 0
    with zf.open(info) as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > cap:
                raise HTTPException(
                    status_code=413,
                    detail=f"file too large (max {_format_max_size(cap)})",
                )
            chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Public aliases — routers and other callers should use the un-underscored
# names. The underscored variants remain as the canonical definitions so
# in-module references keep working.
# ---------------------------------------------------------------------------
sanitize_filename = _sanitize_filename
safe_child = _safe_child
safe_dir = _safe_dir
read_upload = _read_upload
stream_upload_to_disk = _stream_upload_to_disk
check_upload_batch = _check_upload_batch
check_archive_bounds = _check_archive_bounds
check_archive_paths = _check_archive_paths
read_zip_member = _read_zip_member
format_max_size = _format_max_size




# ---------------------------------------------------------------------------
# Request authentication and browser-origin guards.
#
# Ported from seg-studio (2026-08-03) after a parity audit found cls-studio had
# no request guard at all in its default, token-less configuration: every guard
# lived inside `if API_TOKEN:` in main.py, so a fresh install shipped with none.
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1", "[::1]", ""})
_MUTATING_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

#: Name of the browser session cookie issued by POST /api/v1/auth/session.
SESSION_COOKIE_NAME = "cls_session"
_SESSION_DERIVATION = b"cls-studio session cookie v1"


#: Headers a proxy adds when it relays someone else's request. Their presence
#: means the loopback peer is a front door, not the user's own browser.
_FORWARDING_HEADERS = ("x-forwarded-for", "x-forwarded-host", "x-real-ip", "forwarded")

_LOOPBACK_PEERS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})


def secrets_equal(supplied: str, expected: str) -> bool:
    """Constant-time comparison of two credentials.

    hmac.compare_digest refuses str arguments containing non-ASCII and raises
    TypeError, which in middleware turns a wrong password into a 500 instead of
    a 401 -- and tells the caller their guess was unusual. Comparing the UTF-8
    bytes keeps the timing property and treats every input as simply wrong.
    """
    return _hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def is_same_origin(origin_header: str, host_header: str) -> bool:
    """True when a browser request did not come from another site.

    An absent Origin means a same-origin GET or a non-browser client; both are
    fine. A present one must be the exact origin being served -- host AND port.

    Deliberately stricter than the CSRF check in evaluate_request_guard, which
    also accepts anything named in CLS_ALLOWED_HOSTS and compares that entry by
    hostname only. That is the right latitude for "may this request change
    state", because the operator listed those names. It is the wrong latitude
    for handing out the shared secret: an allow-listed name matches on any
    port, so a page on a different port of the same machine would qualify, and
    whoever reads the token can then replay it as X-API-Token from anywhere on
    the network.
    """
    origin = (origin_header or "").strip()
    if not origin:
        return True
    origin_netloc = _urlsplit(origin).netloc.lower()
    return bool(origin_netloc) and origin_netloc == (host_header or "").strip().lower()


def is_local_peer(client_host: str | None, forwarding_headers: Iterable[str] = ()) -> bool:
    """True when the TCP peer is this machine and nothing relayed the request.

    Unlike the Host header, the peer address cannot be forged by the client, so
    this survives a bind to 0.0.0.0: a LAN caller claiming ``Host: localhost``
    still connects from its own address. Any forwarding header disqualifies the
    peer, because a reverse proxy on this host also connects from loopback and
    the request behind it is not local at all.
    """
    if not client_host:
        return False
    if any(h.lower() in _FORWARDING_HEADERS for h in forwarding_headers):
        return False
    return client_host.strip().strip("[]") in _LOOPBACK_PEERS


def session_cookie_value(configured_token: str) -> str:
    """The cookie value that proves knowledge of ``configured_token``.

    A hash rather than the token itself, so the secret never appears in a
    browser's cookie jar, a proxy log, or an HAR export. It is derived
    deterministically so sessions survive a server restart — there is no
    server-side session store to keep in sync.
    """
    if not configured_token:
        return ""
    return _hmac.new(_SESSION_DERIVATION, configured_token.encode("utf-8"), _hashlib.sha256).hexdigest()


def _hostname_only(netloc: str) -> str:
    """Lowercase hostname (no port) from a Host header or an Origin netloc."""
    s = (netloc or "").strip().lower()
    if "://" in s:
        s = _urlsplit(s).netloc
    if s.startswith("["):  # bracketed IPv6, e.g. [::1]:8002
        return s[: s.index("]") + 1] if "]" in s else s
    return s.rsplit(":", 1)[0] if ":" in s else s


def _extra_allowed_hosts() -> frozenset[str]:
    raw = _os.getenv("CLS_ALLOWED_HOSTS", "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def is_allowed_host(host_header: str) -> bool:
    """True when the Host header is loopback or an explicitly allowed name.

    A DNS-rebinding request arrives with the attacker's domain in Host, which
    is neither loopback nor allow-listed, so it is rejected.
    """
    h = _hostname_only(host_header)
    return h in _LOOPBACK_HOSTNAMES or h in _extra_allowed_hosts()


def _origin_matches_host(origin_header: str, host_header: str) -> bool:
    """True when the request's Origin is the same origin as its Host (or allowed)."""
    origin_netloc = _urlsplit((origin_header or "").strip()).netloc.lower()
    if not origin_netloc:
        return False
    if origin_netloc == (host_header or "").strip().lower():
        return True
    return _hostname_only(origin_netloc) in _extra_allowed_hosts()


def evaluate_request_guard(
    *,
    method: str,
    host_header: str,
    origin_header: str,
    supplied_token: str,
    configured_token: str,
    supplied_cookie: str = "",
    local_peer: bool = False,
) -> tuple[str, str]:
    """Decide a guarded request: returns (verdict, reason).

    verdict is one of "allow", "unauthorized" (401), "forbidden" (403). The
    caller is responsible for only invoking this on guarded paths.
    """
    m = (method or "GET").upper()
    if m == "OPTIONS":
        return "allow", ""
    if configured_token:
        if supplied_token and secrets_equal(supplied_token, configured_token):
            return "allow", ""
        if supplied_cookie and secrets_equal(supplied_cookie, session_cookie_value(configured_token)):
            # A session cookie is an *ambient* credential: the browser attaches
            # it to any request aimed at this origin, including one triggered by
            # another site. So the CSRF check that guards the tokenless bind
            # applies here too. The DNS-rebinding host allowlist deliberately
            # does not: cookies are scoped to the hostname the user actually
            # browsed to, so a rebound attacker origin is never sent this cookie
            # to begin with, and requiring loopback here would break the LAN
            # deployment this cookie exists to serve.
            if m in _MUTATING_METHODS and origin_header and not _origin_matches_host(origin_header, host_header):
                return "forbidden", "Cross-origin request rejected (CSRF protection)."
            return "allow", ""
        if not local_peer:
            return "unauthorized", "Missing or invalid X-API-Token header."
        # A request from this machine itself. The token exists to authenticate
        # the *network*, and this peer is not on it: binding to 0.0.0.0 should
        # not make the operator's own browser log in to their own desktop app.
        # Falling through applies exactly the rules that protect the default
        # loopback install, so this is no weaker than a local-only deployment.
    # Tokenless (or a local peer on a token-protected server): only defensible
    # for a loopback, same-origin browser client.
    #
    # "Loopback" has to mean the peer, not the Host header. The header is
    # written by the caller, so a LAN client can send `Host: localhost` and
    # satisfy is_allowed_host from anywhere on the network. The startup gate
    # that is supposed to stop a tokenless off-box bind resolves the interface
    # from CLS_HOST and the persisted lan_access flag, so it never sees
    # `uvicorn --host 0.0.0.0` and does not fire for it. That left one
    # unauthenticated path onto the whole API, reachable by anything that could
    # open the port.
    #
    # Every deployment that legitimately serves the network already carries a
    # token: CLS_HOST and the lan_access flag both route through the startup
    # gate, which refuses to start without one. So a tokenless server reaching
    # this point is local-only by definition, and saying so costs nothing.
    if not local_peer:
        return "unauthorized", (
            "This server has no API token configured, so it serves only "
            "clients on the machine it runs on. Set CLS_API_TOKEN to allow "
            "access over the network."
        )
    if not is_allowed_host(host_header):
        return "forbidden", "Host header not allowed (possible DNS rebinding)."
    if m in _MUTATING_METHODS and origin_header and not _origin_matches_host(origin_header, host_header):
        return "forbidden", "Cross-origin request rejected (CSRF protection)."
    return "allow", ""


def _session_cookie_from_header(cookie_header: str) -> str:
    """Extract the session cookie from a raw ``Cookie:`` header value.

    Starlette parses cookies for HTTP requests, but a WebSocket handshake is
    judged in raw-ASGI territory where only the byte headers are available.
    """
    for part in (cookie_header or "").split(";"):
        name, _, value = part.strip().partition("=")
        if name == SESSION_COOKIE_NAME:
            return value.strip()
    return ""


class WebSocketTokenGate:
    """Reject unauthenticated WebSocket handshakes when a shared token is set.

    Starlette's ``@app.middleware("http")`` only sees HTTP scopes, so the
    ``X-API-Token`` check for the REST surface never fires for WebSocket
    connects. This pure-ASGI middleware closes guarded WebSocket handshakes
    with code 4401 unless the client supplies the token via an
    ``X-API-Token`` header or an ``api_token`` query parameter (browsers
    cannot set custom headers on WebSocket connections).
    """

    def __init__(self, app, token: str = "", guard: Callable[[str], bool] | None = None):
        self.app = app
        self.token = token
        self.guard = guard or (lambda _path: False)

    async def __call__(self, scope, receive, send):
        if self.token and scope["type"] == "websocket" and self.guard(scope.get("path", "")):
            supplied = ""
            for key, value in scope.get("headers") or []:
                if key == b"x-api-token":
                    supplied = value.decode("latin-1")
                    break
            if not supplied:
                query = parse_qs((scope.get("query_string") or b"").decode("latin-1"))
                supplied = (query.get("api_token") or [""])[0]
            if supplied != self.token:
                await receive()  # consume the websocket.connect event
                await send({"type": "websocket.close", "code": 4401})
                return
        await self.app(scope, receive, send)

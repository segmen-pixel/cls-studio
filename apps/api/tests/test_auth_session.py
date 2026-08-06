# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Browser sign-in: the cookie flow that lets a LAN browser use the API.

The shared TestClient is deliberately session-scoped (see conftest: a second
client re-runs startup and duplicates every route), so the LAN-peer decision
matrix is exercised against evaluate_request_guard directly, and the routes
are exercised through the shared client.
"""
from __future__ import annotations

import pytest

TOKEN = "test-shared-secret"


@pytest.fixture
def token_set(monkeypatch):
    """Run the auth router as a token-protected deployment."""
    monkeypatch.setattr("app.routers.auth.API_TOKEN", TOKEN)
    return TOKEN


# --- the LAN decision matrix (pure function; no app, no second client) ------

def _guard(**kw):
    from app.core.security import evaluate_request_guard
    base = dict(method="GET", host_header="testserver", origin_header="",
                supplied_token="", configured_token=TOKEN, supplied_cookie="",
                local_peer=False)
    base.update(kw)
    return evaluate_request_guard(**base)


def test_lan_peer_without_credential_is_unauthorized():
    assert _guard()[0] == "unauthorized"


def test_lan_peer_with_token_header_is_allowed():
    assert _guard(supplied_token=TOKEN)[0] == "allow"


def test_lan_peer_with_session_cookie_is_allowed():
    from app.core.security import session_cookie_value
    assert _guard(supplied_cookie=session_cookie_value(TOKEN))[0] == "allow"


def test_session_cookie_does_not_bypass_csrf():
    """The cookie is ambient, so a cross-origin mutation must still fail."""
    from app.core.security import session_cookie_value
    verdict, reason = _guard(method="POST", origin_header="https://evil.example",
                             supplied_cookie=session_cookie_value(TOKEN))
    assert verdict == "forbidden" and "CSRF" in reason


def test_a_stale_cookie_is_not_a_credential():
    assert _guard(supplied_cookie="deadbeef")[0] == "unauthorized"


# --- the routes -------------------------------------------------------------

def test_status_is_open_and_honest_without_a_token(client):
    r = client.get("/api/v1/auth/status")
    assert r.status_code == 200
    assert r.json() == {"token_required": False, "authenticated": True}


def test_status_reports_a_token_is_required(client, token_set):
    r = client.get("/api/v1/auth/status")
    assert r.status_code == 200 and r.json()["token_required"] is True


def test_wrong_token_is_rejected_and_sets_no_cookie(client, token_set):
    from app.core.security import SESSION_COOKIE_NAME
    r = client.post("/api/v1/auth/session", json={"token": "wrong"})
    assert r.status_code == 401
    assert r.json()["authenticated"] is False
    assert SESSION_COOKIE_NAME not in r.cookies


def test_correct_token_mints_a_hardened_cookie(client, token_set):
    from app.core.security import SESSION_COOKIE_NAME, session_cookie_value
    r = client.post("/api/v1/auth/session", json={"token": TOKEN})
    assert r.status_code == 200 and r.json()["authenticated"] is True
    assert r.cookies[SESSION_COOKIE_NAME] == session_cookie_value(TOKEN)
    raw = r.headers["set-cookie"].lower()
    assert "httponly" in raw and "samesite=strict" in raw
    # Plain-HTTP LAN deployments must not get a Secure cookie they cannot send.
    assert "secure" not in raw
    assert TOKEN not in raw  # the secret never reaches the cookie jar
    client.post("/api/v1/auth/logout")


def test_session_without_a_token_configured_mints_nothing(client):
    r = client.post("/api/v1/auth/session", json={"token": "anything"})
    assert r.status_code == 200
    assert r.json() == {"token_required": False, "authenticated": True}


def test_logout_clears_the_cookie(client, token_set):
    client.post("/api/v1/auth/session", json={"token": TOKEN})
    r = client.post("/api/v1/auth/logout")
    assert r.status_code == 200
    assert r.headers["set-cookie"].startswith("cls_session=")
    assert "max-age=0" in r.headers["set-cookie"].lower() or 'cls_session=""' in r.headers["set-cookie"]


def test_token_reveal_needs_local_peer_and_same_origin(client, token_set):
    """Loopback alone is not enough: any page on this box is a local peer."""
    r = client.get("/api/v1/auth/token", headers={"Origin": "http://evil.local"})
    assert r.status_code == 403 and r.json()["token"] == ""


def test_token_reveal_serves_the_operator_at_the_machine(client, token_set):
    r = client.get("/api/v1/auth/token", headers={"Origin": "http://testserver"})
    assert r.status_code == 200
    assert r.json()["token"] == TOKEN
    assert r.headers.get("Vary") == "Origin"

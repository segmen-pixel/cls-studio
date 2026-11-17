# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""The request guard must hold in the default, token-less configuration.

These assert the attacks the guard exists to stop, so a regression that makes
the middleware permissive fails here rather than shipping.
"""
from __future__ import annotations

import pytest


def test_cross_origin_mutation_is_rejected(client, project_id):
    """A page on another site cannot make the local server change state."""
    resp = client.delete(
        f"/api/v1/projects/{project_id}",
        headers={"Origin": "https://evil.example"},
    )
    assert resp.status_code == 403
    assert "CSRF" in resp.json()["detail"]


def test_cross_origin_read_is_allowed(client):
    """GETs are not blocked: CORS already stops the attacker reading them."""
    resp = client.get("/api/v1/projects", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 200


def test_same_origin_mutation_is_allowed(client):
    resp = client.post(
        "/api/v1/projects",
        json={"name": "guard-same-origin"},
        headers={"Origin": "http://testserver"},
    )
    assert resp.status_code == 200
    client.delete(f"/api/v1/projects/{resp.json()['id']}")


def test_rebound_host_is_rejected(client):
    """A DNS-rebinding request carries the attacker's name in Host."""
    resp = client.get("/api/v1/projects", headers={"Host": "attacker.example"})
    assert resp.status_code == 403
    assert "rebinding" in resp.json()["detail"].lower()


def test_no_origin_mutation_is_allowed(client):
    """Non-browser clients (curl, the launcher) send no Origin and still work."""
    resp = client.post("/api/v1/projects", json={"name": "guard-no-origin"})
    assert resp.status_code == 200
    client.delete(f"/api/v1/projects/{resp.json()['id']}")


def test_unguarded_paths_stay_open(client):
    assert client.get("/api/v1/health").status_code == 200


@pytest.mark.parametrize("supplied,expected", [("", False), ("x", False)])
def test_secrets_equal_is_total(supplied, expected):
    """Non-ASCII must compare False, not raise (a 500 leaks that it was odd)."""
    from app.core.security import secrets_equal
    assert secrets_equal(supplied, "secret") is expected
    assert secrets_equal("パスワード", "secret") is False


def test_local_peer_rejects_relayed_requests():
    from app.core.security import is_local_peer
    assert is_local_peer("127.0.0.1", []) is True
    assert is_local_peer("127.0.0.1", ["X-Forwarded-For"]) is False
    assert is_local_peer("192.168.1.50", []) is False
    assert is_local_peer(None, []) is False


def test_nonlocal_bind_without_token_is_refused(monkeypatch):
    from app.core import config
    monkeypatch.setattr(config, "API_TOKEN", "")
    monkeypatch.setenv("CLS_HOST", "0.0.0.0")
    with pytest.raises(SystemExit) as exc:
        config.require_token_for_nonlocal_bind()
    assert "Refusing to start" in str(exc.value)


def test_nonlocal_bind_with_token_is_allowed(monkeypatch):
    from app.core import config
    monkeypatch.setattr(config, "API_TOKEN", "a-shared-secret")
    monkeypatch.setenv("CLS_HOST", "0.0.0.0")
    config.require_token_for_nonlocal_bind()


def test_loopback_bind_needs_no_token(monkeypatch):
    from app.core import config
    monkeypatch.setattr(config, "API_TOKEN", "")
    monkeypatch.setenv("CLS_HOST", "127.0.0.1")
    config.require_token_for_nonlocal_bind()

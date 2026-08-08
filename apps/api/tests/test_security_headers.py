# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Security-header middleware.

Framing is denied outright: nothing this app serves is meant to be embedded.
"""
from __future__ import annotations


def test_regular_api_response_denies_framing(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    assert resp.headers.get("x-frame-options") == "DENY"


def test_error_responses_carry_the_headers_too(client):
    # The middleware runs on the way out, so a 404 is covered as well.
    resp = client.get("/api/v1/projects/_none")
    assert resp.headers.get("x-frame-options") == "DENY"
    assert resp.headers.get("x-content-type-options") == "nosniff"

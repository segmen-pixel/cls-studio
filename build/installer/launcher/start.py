# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Entry point baked into the Windows installer bundle.

The bundle ships an embedded CPython, so this runs with the repository root as
the working directory and starts the same ASGI app the dev scripts do. It is
copied into the staging tree by ``scripts/build_installer.py``; keep it
dependency-free beyond what the bundle installs.
"""
from __future__ import annotations

import threading
import time
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8791
URL = f"http://localhost:{PORT}/ui/"


def _open_browser() -> None:
    # The server needs a moment before the UI route answers; opening too early
    # shows the browser's own error page and users assume the install failed.
    time.sleep(2.0)
    try:
        webbrowser.open(URL)
    except Exception:
        pass


def main() -> None:
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(
        "apps.api.app.main:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()

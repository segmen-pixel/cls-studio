# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Windows scripts must carry CRLF in the repository, not just after checkout.

cmd.exe seeks a ``goto`` label by byte offset. Given an LF-only batch file it
can land mid-line, so the line it thinks it jumped to is never run -- the
symptom is a variable that is silently empty two lines later, not a syntax
error. Measured on a Surface Book 3 on 2026-08-18: the same
``install_windows.bat`` failed at the venv step as LF and completed as CRLF,
with line endings the only difference between the two runs.

``.gitattributes`` therefore marks these ``-text`` rather than
``text eol=crlf``: ``eol`` is a checkout filter, so it fixes a clone and does
nothing for ``git archive`` -- which is exactly what GitHub serves as a
release's "Source code (zip)", and how most users get these files. The blob
itself has to be CRLF, and this test is what keeps it that way after someone
edits a script on Linux.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PATTERNS = ("*.bat", "*.ps1", "*.cmd")


def _tracked_windows_scripts() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", *PATTERNS],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        return []
    return [REPO / line for line in out.stdout.split() if line]


def test_every_windows_script_is_crlf():
    scripts = _tracked_windows_scripts()
    if not scripts:
        return  # not a git checkout (source zip / vendored tree) -- nothing to assert
    offenders = []
    for p in scripts:
        if not p.is_file():
            continue
        raw = p.read_bytes()
        bare_lf = raw.count(b"\n") - raw.count(b"\r\n")
        if bare_lf:
            offenders.append(f"{p.relative_to(REPO)} ({bare_lf} bare LF)")
    assert not offenders, (
        "Windows scripts must be CRLF; cmd.exe mis-seeks goto labels otherwise:\n  "
        + "\n  ".join(offenders)
    )


def test_gitattributes_does_not_normalise_windows_scripts():
    """`text eol=crlf` would leave the blob LF and only fix checkouts."""
    text = (REPO / ".gitattributes").read_text(encoding="utf-8")
    for pat in ("*.bat", "*.ps1"):
        line = next(
            (ln for ln in text.splitlines() if ln.split("#")[0].strip().startswith(pat)),
            None,
        )
        assert line is not None, f"{pat} has no .gitattributes rule"
        assert "-text" in line, (
            f"{pat} must be -text so the blob keeps CRLF; found: {line!r}"
        )

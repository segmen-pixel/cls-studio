# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Custom exception classes for the cls-studio API.

All domain exceptions inherit from :class:`AppError`, which carries a
structured ``code`` (CLS-XXXX) used in API responses and logs.
Existing exception names are preserved for backward compatibility.

Security: ``detail`` is logged server-side only — never sent to the client.
"""
from __future__ import annotations

from typing import Any

from . import error_codes as E


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class AppError(Exception):
    """Base exception for all application errors.

    Attributes:
        code:  ANL error code string (e.g. ``"CLS-9002"``).
        user_message:  Safe message returned to the client.
        detail:  Internal-only context (logged, **never** in API response).
        context:  Structured metadata (project_id, bank_id, …).
    """

    code: str = E.SYSTEM_INTERNAL

    def __init__(
        self,
        user_message: str = "",
        *,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        if not user_message:
            user_message = E.get_meta(self.code).message_en
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail
        self.context = context or {}

    @property
    def http_status(self) -> int:
        return E.get_meta(self.code).http_status

    @property
    def log_level(self) -> str:
        return E.get_meta(self.code).log_level


# ---------------------------------------------------------------------------
# Project / Resource (CLS-2xxx)
# ---------------------------------------------------------------------------
class ProjectNotFoundError(AppError):
    """The requested project does not exist on disk."""
    code = E.NOT_FOUND_PROJECT


class ImageNotFoundError(AppError):
    """The requested image does not exist."""
    code = E.NOT_FOUND_IMAGE


class MaskNotFoundError(AppError):
    """The requested mask does not exist."""
    code = E.NOT_FOUND_MASK


class BankNotFoundError(AppError):
    """The requested memory bank does not exist (or is tombstoned)."""
    code = E.NOT_FOUND_BANK


# ---------------------------------------------------------------------------
# Validation (CLS-1xxx)
# ---------------------------------------------------------------------------
class ValidationError(AppError):
    """Generic validation error."""
    code = E.VALIDATION_REQUIRED_PARAM


class ImageDecodeError(AppError):
    """Uploaded image bytes could not be decoded."""
    code = E.VALIDATION_IMAGE_DECODE


# ---------------------------------------------------------------------------
# Scoring (CLS-4xxx)
# ---------------------------------------------------------------------------
class PredictError(AppError):
    """Scoring failed (e.g. the overlay could not be rendered)."""
    code = E.INFER_FAILED


# ---------------------------------------------------------------------------
# System (CLS-7xxx)
# ---------------------------------------------------------------------------
class GPUDeviceError(AppError):
    """GPU device configuration failed."""
    code = E.SYSTEM_GPU_DEVICE


class FileIOError(AppError):
    """File I/O error."""
    code = E.SYSTEM_FILE_IO


# ---------------------------------------------------------------------------
# Security (CLS-8xxx)
# ---------------------------------------------------------------------------
class PathTraversalError(AppError):
    """Path traversal detected."""
    code = E.SECURITY_PATH_TRAVERSAL


# ---------------------------------------------------------------------------
# Bank / Active state (CLS-9xxx)
# ---------------------------------------------------------------------------
class BankCorruptError(AppError):
    """A memory bank's on-disk data failed to load (corrupt files)."""
    code = E.BANK_DATA_CORRUPT


class NoActiveProjectError(AppError):
    """A bank route was called before any project was selected."""
    code = E.BANK_NO_ACTIVE_PROJECT


class ActiveProjectDeletedError(AppError):
    """The active project was deleted while the request was in flight."""
    code = E.BANK_ACTIVE_PROJECT_DELETED

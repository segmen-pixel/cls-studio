# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Structured error code registry for cls-studio.

Every user-facing error MUST have a code from this module.
Internal details (paths, stack traces) are NEVER included in responses.

Categories:
    CLS-1xxx  Validation        400  Request parameter issues
    CLS-2xxx  Not Found         404  Resource lookup failures
    CLS-4xxx  Scoring           5xx  Scoring / overlay rendering
    CLS-7xxx  System            5xx  Internal / hardware
    CLS-8xxx  Security          4xx  Path traversal, access
    CLS-9xxx  Bank / State      4xx  Memory-bank lookup / active-state

The numbering has gaps: CLS-3xxx (training), CLS-5xxx (AI assist) and
CLS-6xxx (dataset import/export) covered features that this product does not
have, so those codes were removed rather than shipped as dead advertising.
Retired numbers are never reused — a code in a log always means one thing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorMeta:
    """Immutable metadata for a single error code."""

    http_status: int
    message_en: str
    message_ja: str
    hint_en: str | None = None
    hint_ja: str | None = None
    log_level: str = "WARNING"


# ---------------------------------------------------------------------------
# Registry — code string → ErrorMeta
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, ErrorMeta] = {}


def _reg(code: str, meta: ErrorMeta) -> str:
    """Register a code and return the code string for assignment."""
    _REGISTRY[code] = meta
    return code


def get_meta(code: str) -> ErrorMeta:
    """Look up metadata for an error code. Falls back to CLS-7004."""
    return _REGISTRY.get(code, _REGISTRY["CLS-7004"])


# ── CLS-1xxx: Validation ─────────────────────────────────────────────────
VALIDATION_CLASS_ID_RANGE = _reg("CLS-1001", ErrorMeta(
    400, "Class ID must be in 0..254.",
    "クラスIDは0〜254の範囲で指定してください。"))
VALIDATION_IGNORE_INDEX = _reg("CLS-1002", ErrorMeta(
    400, "ignore_index must be 255.",
    "ignore_indexは255でなければなりません。"))
VALIDATION_DUPLICATE_IDS = _reg("CLS-1003", ErrorMeta(
    400, "Duplicate class IDs are not allowed.",
    "クラスIDが重複しています。"))
VALIDATION_BG_DELETE = _reg("CLS-1005", ErrorMeta(
    400, "Cannot delete the background class.",
    "背景クラスは削除できません。"))
VALIDATION_REQUIRED_PARAM = _reg("CLS-1006", ErrorMeta(
    400, "A required parameter is missing.",
    "必須パラメータが不足しています。"))
VALIDATION_JSON_PARSE = _reg("CLS-1007", ErrorMeta(
    400, "Invalid JSON in request body.",
    "リクエストのJSONが不正です。"))
VALIDATION_FILE_FORMAT = _reg("CLS-1008", ErrorMeta(
    400, "Unsupported file format.",
    "対応していないファイル形式です。"))
VALIDATION_EMPTY_FILE = _reg("CLS-1009", ErrorMeta(
    400, "Uploaded file is empty.",
    "アップロードされたファイルが空です。"))
VALIDATION_IMAGE_DECODE = _reg("CLS-1011", ErrorMeta(
    422, "Could not decode image bytes.",
    "画像データをデコードできませんでした。"))

# ── CLS-2xxx: Not Found ──────────────────────────────────────────────────
NOT_FOUND_PROJECT = _reg("CLS-2001", ErrorMeta(
    404, "Project not found.",
    "プロジェクトが見つかりません。"))
NOT_FOUND_IMAGE = _reg("CLS-2002", ErrorMeta(
    404, "Image not found.",
    "画像が見つかりません。"))
NOT_FOUND_MASK = _reg("CLS-2003", ErrorMeta(
    404, "Mask not found.",
    "マスクが見つかりません。"))
NOT_FOUND_CLASSES = _reg("CLS-2008", ErrorMeta(
    404, "classes.json not found or has no classes defined.",
    "classes.jsonが見つからないか、クラスが未定義です。"))
NOT_FOUND_BANK = _reg("CLS-2010", ErrorMeta(
    404, "Bank not found.",
    "バンクが見つかりません。"))

# ── CLS-4xxx: Scoring ────────────────────────────────────────────────────
INFER_FAILED = _reg("CLS-4002", ErrorMeta(
    500, "Prediction failed.",
    "推論に失敗しました。",
    log_level="ERROR"))

# ── CLS-7xxx: System ─────────────────────────────────────────────────────
SYSTEM_GPU_DEVICE = _reg("CLS-7001", ErrorMeta(
    500, "GPU device configuration failed.",
    "GPUデバイスの設定に失敗しました。",
    log_level="ERROR"))
SYSTEM_INTERNAL = _reg("CLS-7004", ErrorMeta(
    500, "An internal error occurred.",
    "内部エラーが発生しました。",
    hint_en="Check server logs or contact support with the error code.",
    hint_ja="サーバーログを確認するか、エラーコードを添えてサポートに連絡してください。",
    log_level="ERROR"))
SYSTEM_COREML_UNAVAILABLE = _reg("CLS-7006", ErrorMeta(
    501, "Core ML export is not available on this machine.",
    "\u3053\u306e\u30de\u30b7\u30f3\u3067\u306f Core ML \u30a8\u30af\u30b9\u30dd\u30fc\u30c8\u3092\u5b9f\u884c\u3067\u304d\u307e\u305b\u3093\u3002",
    hint_en=("Conversion needs macOS: coremltools publishes no Windows build, and a "
             "converted encoder cannot be run - and therefore cannot be checked "
             "against the bank - anywhere else. Export from a Mac instead."),
    hint_ja=("\u5909\u63db\u306b\u306f macOS \u304c\u5fc5\u8981\u3067\u3059\u3002coremltools \u306b Windows \u7248\u306f\u306a\u304f\u3001"
             "\u5909\u63db\u5f8c\u306e\u30a8\u30f3\u30b3\u30fc\u30c0\u3092\u5b9f\u884c\u3067\u304d\u306a\u3044\u305f\u3081 "
             "\u30d0\u30f3\u30af\u3068\u306e\u7167\u5408\u3082\u3067\u304d\u307e\u305b\u3093\u3002Mac \u3067\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002"),
    log_level="INFO"))
SYSTEM_FILE_IO = _reg("CLS-7005", ErrorMeta(
    500, "File I/O error.",
    "ファイルI/Oエラーが発生しました。",
    log_level="ERROR"))

# ── CLS-8xxx: Security ───────────────────────────────────────────────────
SECURITY_PATH_TRAVERSAL = _reg("CLS-8001", ErrorMeta(
    400, "Invalid path detected.",
    "不正なパスが検出されました。"))
SECURITY_INVALID_PROJECT_ID = _reg("CLS-8002", ErrorMeta(
    400, "Invalid project ID.",
    "不正なプロジェクトIDです。"))

# ── CLS-9xxx: Bank / Active state ────────────────────────────────────────
BANK_DATA_CORRUPT = _reg("CLS-9001", ErrorMeta(
    422, "Bank data is corrupt.",
    "バンクデータが破損しています。",
    log_level="ERROR"))
BANK_NO_ACTIVE_PROJECT = _reg("CLS-9002", ErrorMeta(
    409, "No active project — select a project first.",
    "アクティブなプロジェクトがありません — 先にプロジェクトを選択してください。"))
BANK_ACTIVE_PROJECT_DELETED = _reg("CLS-9003", ErrorMeta(
    409, "Active project was deleted — select another project.",
    "アクティブなプロジェクトは削除されました — 別のプロジェクトを選択してください。"))

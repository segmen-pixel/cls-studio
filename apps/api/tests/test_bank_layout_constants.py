# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""``bank_images`` restates clscore's on-disk layout; this keeps it honest.

``bank_images`` cannot import clscore: ``clscore/__init__.py`` imports
``clscore.bank`` which imports torch, and ``routers/projects.py`` — one of the
module's two consumers — must stay free of torch-adjacent modules. So the
layout names are spelled out there and checked against their owners here,
where importing clscore (and therefore torch) costs nothing.

This supersedes the single-constant check that lived in
``test_project_thumbnail.py``. That one pinned ``IMAGES_SUBDIR`` and not
``STORE_SUBDIR``, so renaming half the path would have broken the project
card's thumbnail with a green suite.
"""

from __future__ import annotations

from app.core import bank_images as bi


def test_the_store_layout_matches_clscore():
    from clscore import store as store_mod

    assert bi.STORE_SUBDIR == store_mod.STORE_SUBDIR
    assert bi.STORE_IMAGES_SUBDIR == store_mod.IMAGES_SUBDIR
    assert bi.STORE_INDEX_FILE == store_mod.STORE_INDEX_FILE


def test_the_bank_layout_matches_clscore():
    from clscore.bank import INDEX_ENTRY_ID_KEY, SOURCE_IMAGES_SUBDIR, Bank

    assert bi.BANK_META_FILE == Bank.META_FILE
    assert bi.IMAGES_SUBDIR == SOURCE_IMAGES_SUBDIR
    assert bi.INDEX_ENTRY_ID_KEY == INDEX_ENTRY_ID_KEY


def test_the_labelset_layout_matches_clscore():
    from clscore import labelset as ls_mod

    assert bi.LABELSETS_SUBDIR == ls_mod.LABELSETS_SUBDIR
    assert bi.LABELSET_ACTIVE_MARKER == ls_mod.ACTIVE_MARKER


def test_the_tiers_match_the_type_clscore_declares():
    from clscore.bank import Tier

    assert bi.TIERS == Tier.__args__


def test_the_app_side_layout_matches_its_owners():
    from app.core import cls_state, cls_store
    from app.routers import staging

    assert bi.BANKS_SUBDIR == cls_state.BANKS_SUBDIR
    assert bi.IMAGES_SUBDIR == cls_state.IMAGES_SUBDIR
    assert bi.STORE_RENDER_SUBDIR == cls_store.RENDER_SUBDIR
    assert bi.STAGING_SUBDIR == staging.STAGING_SUBDIR


def test_the_displayable_set_is_what_the_writers_have_always_meant():
    """A literal, so widening the browser-native set stays a conscious act.

    Both writers transcode anything outside this to PNG; adding a format here
    without teaching them means a file that is served but cannot be drawn.
    """
    assert bi.DISPLAYABLE_EXTS == frozenset(
        {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    )


def test_the_decodable_set_is_the_zip_import_allowlist():
    assert bi.DECODABLE_EXTS == frozenset(
        {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    )

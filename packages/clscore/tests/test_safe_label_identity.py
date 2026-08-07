# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""A defect class's on-disk stem is its PRIMARY KEY, so it has to be unique.

The stem keys ``bank.critical[label]``, ``BankMeta.*_image_index[label]``, the
per-row metadata, the eval-cache namespace and the ``<label>.npy`` file. The
old rule collapsed every run of non-ASCII to "_" and then stripped it, so

    safe_label("傷")   -> DEFAULT_LABEL
    safe_label("汚れ") -> DEFAULT_LABEL      same bucket
    safe_label("傷A")  -> "A"
    safe_label("汚れA") -> "A"               same bucket

merged two defect classes into one tensor, one row index and one metadata
array. The UI actively invites Japanese defect names (the placeholder is
"新しい種類"), so this is the ordinary case here, not an edge case.
"""

from __future__ import annotations

import numpy as np

from clscore.assemble import assemble_bank
from clscore.bank import DEFAULT_LABEL, safe_label
from clscore.labelset import LabelSet
from clscore.store import FeatureStore, StoreMeta

DIM = 16


# ---- the rule itself --------------------------------------------------------


def test_ascii_labels_are_untouched():
    """No migration for the banks that never hit the bug.

    Their stems, their assembly fingerprint and their eval-cache keys all stay
    byte-identical, so only affected projects are asked to re-assemble.
    """
    for label in ("scratch", "dent", "lot-3", "A", "_default", "a_b-c"):
        assert safe_label(label) == label


def test_a_label_that_is_already_a_legal_stem_passes_through_verbatim():
    """Underscores included. ``.strip("_")`` would have turned ``_default``
    into ``default-3bf30573`` and split the bucket the codebase names by hand.
    """
    for label in ("_default", "_leading", "trailing_", "__both__"):
        assert safe_label(label) == label


def test_a_space_and_an_underscore_are_no_longer_the_same_class():
    """Both used to reduce to "a_b" -- an ASCII collision, not just a CJK one."""
    assert safe_label("a b") != safe_label("a_b")
    assert safe_label("a_b") == "a_b", "the already-legal one keeps its name"


def test_empty_and_blank_still_fall_back_to_the_default_label():
    for label in (None, "", "   ", "\t"):
        assert safe_label(label) == DEFAULT_LABEL


def test_two_japanese_classes_no_longer_share_a_bucket():
    a, b = safe_label("傷"), safe_label("汚れ")
    assert a != b
    assert a != DEFAULT_LABEL and b != DEFAULT_LABEL


def test_japanese_classes_with_a_shared_ascii_suffix_stay_apart():
    """The subtler half: "傷A" and "汚れA" both used to reduce to "A"."""
    assert safe_label("傷A") != safe_label("汚れA")
    assert safe_label("キズ2") != safe_label("汚れ2")


def test_the_stem_is_filename_safe():
    import re

    for label in ("傷", "傷A", "スクラッチ 3", "a/b", "../etc", "50%", "  x  "):
        assert re.fullmatch(r"[A-Za-z0-9_\-]+", safe_label(label)), label


def test_the_stem_is_stable_and_idempotent():
    once = safe_label("傷A")
    assert safe_label("傷A") == once, "same input, same stem, every run"
    assert safe_label(once) == once, "already-safe stems pass through unchanged"


def test_whitespace_around_a_label_does_not_make_a_new_class():
    assert safe_label("  傷A  ") == safe_label("傷A")


# ---- through an assemble ----------------------------------------------------


def _store(tmp_path, n: int) -> FeatureStore:
    st = FeatureStore(tmp_path / "store", meta=StoreMeta(dim=DIM))
    rng = np.random.default_rng(0)
    for i in range(n):
        st.add(rng.random((5, DIM), dtype=np.float32), name=f"ng{i}.png", grid_rows=5)
    st.save_index()
    return st


def test_two_japanese_defect_classes_assemble_into_two_tensors(tmp_path):
    st = _store(tmp_path, 2)
    ls = LabelSet(id="standard", name="standard")
    ls.assign(st.entries[0].id, tier="critical", label="傷")
    ls.assign(st.entries[1].id, tier="critical", label="汚れ")

    bank = assemble_bank(st, ls)

    assert len(bank.critical) == 2, "one tensor per defect class"
    assert len(bank.meta.critical_image_index) == 2
    assert DEFAULT_LABEL not in bank.critical, "neither collapsed into the bucket"
    for arr in bank.critical.values():
        assert arr.shape[0] == 5, "5 rows each, not 10 in one"


def test_the_bank_remembers_what_the_operator_typed(tmp_path):
    st = _store(tmp_path, 1)
    ls = LabelSet(id="standard", name="standard")
    ls.assign(st.entries[0].id, tier="critical", label="傷A")

    bank = assemble_bank(st, ls)

    stem = safe_label("傷A")
    assert list(bank.critical) == [stem]
    assert bank.meta.label_display[stem] == "傷A", "the stem alone is unreadable"


def test_an_ascii_label_needs_no_display_entry(tmp_path):
    st = _store(tmp_path, 1)
    ls = LabelSet(id="standard", name="standard")
    ls.assign(st.entries[0].id, tier="critical", label="scratch")

    bank = assemble_bank(st, ls)

    assert list(bank.critical) == ["scratch"]
    assert bank.meta.label_display == {}, "the stem already is the name"


def test_label_display_survives_the_meta_round_trip(tmp_path):
    from clscore.bank import BankMeta

    meta = BankMeta(dim=DIM, label_display={"A-deadbeef": "傷A"})
    path = tmp_path / "bank_meta.json"
    path.write_text(meta.to_json(), encoding="utf-8")
    assert BankMeta.from_path(path).label_display == {"A-deadbeef": "傷A"}


def test_a_legacy_bank_without_label_display_still_loads(tmp_path):
    import json

    from clscore.bank import BankMeta

    path = tmp_path / "bank_meta.json"
    path.write_text(json.dumps({"dim": DIM, "critical_images": {}}), encoding="utf-8")
    assert BankMeta.from_path(path).label_display == {}

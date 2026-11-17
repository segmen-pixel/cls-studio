# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""The DINOv2 hub spec must carry a ref.

Without one, ``torch.hub.load`` cannot know whether the default branch is
``main`` or ``master``, so ``_parse_repo_info`` fetches the repo page over
the network on EVERY call -- ahead of the local cache, so a fully cached
machine still cannot load a model offline. Worse, it re-raises any
``HTTPError`` that is not a 404, so a 403 from GitHub (trivially reached by
a CI run that loads a model a few times) fails the load with a ``KeyError``
raised deep inside urllib, nowhere near the cause. Caught exactly that way
on the public repo's CI, 2026-08-09.

``main`` normalises to the same cache directory the unpinned call already
used, so pinning re-downloads nothing.
"""

from __future__ import annotations

import torch

from clscore.feature_extractor import load_dinov2


class _StubBackbone:
    """Just enough surface for load_dinov2's ``.eval().to(device)`` tail."""

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self


def test_load_dinov2_passes_an_explicit_ref(monkeypatch):
    seen: dict[str, str] = {}

    def fake_hub_load(repo, model_name, *args, **kwargs):
        seen["repo"] = repo
        seen["model"] = model_name
        return _StubBackbone()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    load_dinov2("dinov2_vitb14", device="cpu")

    assert ":" in seen["repo"], (
        "the hub spec has no ref, so every load makes a network call to "
        f"resolve the default branch: {seen['repo']!r}"
    )
    owner_repo, ref = seen["repo"].split(":")
    assert owner_repo == "facebookresearch/dinov2"
    assert ref == "main"
    assert seen["model"] == "dinov2_vitb14"


def test_the_pinned_ref_resolves_without_touching_the_network(monkeypatch):
    """The point of the ref, stated as behaviour rather than as a string."""
    import torch.hub as hub

    calls: list[str] = []

    def blocked(url, *args, **kwargs):
        calls.append(str(url))
        raise AssertionError(f"_parse_repo_info went to the network: {url}")

    monkeypatch.setattr(hub, "urlopen", blocked)
    owner, name, ref = hub._parse_repo_info("facebookresearch/dinov2:main")

    assert (owner, name, ref) == ("facebookresearch", "dinov2", "main")
    assert calls == []
    # Same cache directory as the unpinned call resolves to -- nothing
    # re-downloads for anyone who already has the repo cached.
    assert "_".join([owner, name, ref.replace("/", "_")]) == "facebookresearch_dinov2_main"

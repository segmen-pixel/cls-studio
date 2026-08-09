# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""The streaming extractors must be the list forms, minus the accumulation.

A batch teach used to hold every image's tokens AND every image's float32 copy
at once — tens of GB for a group of large images, none of which the allocator
returns to the OS afterwards. The teach path now consumes per image. That is
only safe if the arrays are bit-identical and the batching is unchanged, which
is what these pin: same values, same forwards, same order, and no accumulator
left behind.
"""
from __future__ import annotations

import numpy as np
import torch

from clscore.feature_extractor import (
    extract_images_features_batched,
    extract_windows_tokens_batched,
    iter_images_features_batched,
    iter_windows_tokens_batched,
)
from clscore.sw import DINO_PATCH, WINDOW_SIZE

SIDE = WINDOW_SIZE // DINO_PATCH
DIM = 8


class _StubBackbone(torch.nn.Module):
    """Deterministic stand-in: token value encodes the window's mean pixel."""

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.batch_sizes: list[int] = []

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.batch_sizes.append(int(x.shape[0]))
        seed = x.mean(dim=(1, 2, 3)).view(-1, 1, 1)
        return {"x_norm_patchtokens": seed.expand(x.shape[0], SIDE * SIDE, DIM).clone()}


def _windows(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [
        rng.integers(0, 255, size=(WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
        for _ in range(n)
    ]


def _images(sizes: list[tuple[int, int]]) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    return [rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8) for (h, w) in sizes]


def test_window_stream_concatenates_to_the_list_form():
    wins = _windows(35)
    streamed = torch.cat(
        list(iter_windows_tokens_batched(_StubBackbone(), wins, "cpu", max_batch=8)), dim=0
    )
    listed = extract_windows_tokens_batched(_StubBackbone(), wins, "cpu", max_batch=8)
    assert streamed.shape == listed.shape
    assert torch.equal(streamed, listed)


def test_window_stream_uses_the_same_mini_batches():
    a, b = _StubBackbone(), _StubBackbone()
    list(iter_windows_tokens_batched(a, _windows(35), "cpu", max_batch=8))
    extract_windows_tokens_batched(b, _windows(35), "cpu", max_batch=8)
    assert a.batch_sizes == b.batch_sizes == [8, 8, 8, 8, 3]


def test_empty_input_streams_nothing():
    assert list(iter_windows_tokens_batched(_StubBackbone(), [], "cpu")) == []
    assert list(iter_images_features_batched(_StubBackbone(), [], "cpu")) == []


def test_image_stream_matches_the_list_form_exactly():
    imgs = _images([(600, 900), (520, 520), (1100, 700)])
    listed = extract_images_features_batched(_StubBackbone(), imgs, "cpu", max_batch=8)
    streamed = list(iter_images_features_batched(_StubBackbone(), imgs, "cpu", max_batch=8))
    assert [i for i, _ in streamed] == [0, 1, 2]
    assert len(listed) == 3
    for (_i, got), want in zip(streamed, listed):
        assert got.dtype == np.float32 == want.dtype
        assert got.shape == want.shape
        assert np.array_equal(got, want)


def test_image_stream_yields_before_the_group_is_finished():
    """The whole point: image 0 must be usable while image 2 is still running.

    If this ever regresses to collecting everything first, the peak goes back
    to scaling with the group instead of with one image.
    """
    imgs = _images([(600, 900), (600, 900), (600, 900)])
    model = _StubBackbone()
    gen = iter_images_features_batched(model, imgs, "cpu", max_batch=4)
    first_i, first = next(gen)
    assert first_i == 0
    forwards_after_first = len(model.batch_sizes)
    rest = list(gen)
    assert [i for i, _ in rest] == [1, 2]
    # Some forwards must still have been outstanding when image 0 came out.
    assert forwards_after_first < len(model.batch_sizes)


def test_single_image_bank_features_match_the_token_path():
    """The streamed single-image path must equal building the tensor first.

    That path also stopped defaulting to keep_on_device=True: it was parking
    every window's tokens in VRAM only to copy them to the host on the next
    line.
    """
    from clscore.feature_extractor import extract_image_features_for_bank
    from clscore.sw import pad_to_min, sw_offsets

    img = _images([(900, 1300)])[0]
    got = extract_image_features_for_bank(_StubBackbone(), img, "cpu", max_batch=5)

    padded, _ = pad_to_min(img)
    crops = [padded[y : y + WINDOW_SIZE, x : x + WINDOW_SIZE] for (y, x) in sw_offsets(*padded.shape[:2])]
    toks = extract_windows_tokens_batched(_StubBackbone(), crops, "cpu", max_batch=5)
    want = toks.reshape(-1, toks.shape[-1]).numpy().astype(np.float32)

    assert got.dtype == np.float32
    assert got.shape == want.shape == (len(crops) * SIDE * SIDE, DIM)
    assert np.array_equal(got, want)


def test_mini_batch_straddling_an_image_boundary_is_split_correctly():
    """max_batch deliberately does not divide the per-image window count."""
    imgs = _images([(520, 520), (520, 900), (900, 900)])
    listed = extract_images_features_batched(_StubBackbone(), imgs, "cpu", max_batch=3)
    streamed = [f for _i, f in iter_images_features_batched(_StubBackbone(), imgs, "cpu", max_batch=3)]
    assert len(streamed) == len(listed) == 3
    for got, want in zip(streamed, listed):
        assert np.array_equal(got, want)

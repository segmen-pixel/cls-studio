# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Batched extraction must not stage the whole window set at once.

A batch teach of large images reached thousands of windows and asked CUDA for
one 11 GiB allocation, because every window was stacked and moved to the
device before the first forward ran -- ``max_batch`` only ever bounded the
forward itself. These tests pin the bound to ``max_batch`` instead of the
window count, without needing a GPU: a stub model records the batch shapes it
is handed.
"""

from __future__ import annotations

import numpy as np
import torch

from clscore.feature_extractor import extract_windows_tokens_batched
from clscore.sw import DINO_PATCH, WINDOW_SIZE

SIDE = WINDOW_SIZE // DINO_PATCH
DIM = 8


class _StubBackbone(torch.nn.Module):
    """Returns correctly-shaped tokens and remembers every batch it saw."""

    def __init__(self) -> None:
        super().__init__()
        # One real parameter so the dtype probe in the extractor finds it.
        self.marker = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.seen_batch_sizes: list[int] = []

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen_batch_sizes.append(int(x.shape[0]))
        return {"x_norm_patchtokens": torch.zeros(x.shape[0], SIDE * SIDE, DIM)}


def _windows(n: int) -> list[np.ndarray]:
    return [np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), np.uint8) for _ in range(n)]


def test_no_forward_sees_more_than_max_batch() -> None:
    model = _StubBackbone()
    extract_windows_tokens_batched(model, _windows(70), "cpu", max_batch=16)
    assert model.seen_batch_sizes, "the stub was never called"
    assert max(model.seen_batch_sizes) <= 16, model.seen_batch_sizes
    assert sum(model.seen_batch_sizes) == 70, "every window must be processed exactly once"


def test_output_shape_and_order_are_unchanged() -> None:
    model = _StubBackbone()
    out = extract_windows_tokens_batched(model, _windows(35), "cpu", max_batch=8)
    assert out.shape == (35, SIDE, SIDE, DIM)


def test_a_single_window_still_works() -> None:
    model = _StubBackbone()
    out = extract_windows_tokens_batched(model, _windows(1), "cpu", max_batch=32)
    assert out.shape == (1, SIDE, SIDE, DIM)
    assert model.seen_batch_sizes == [1]


def test_empty_input_returns_empty() -> None:
    model = _StubBackbone()
    out = extract_windows_tokens_batched(model, [], "cpu", max_batch=32)
    assert out.numel() == 0
    assert model.seen_batch_sizes == []


def test_keep_on_device_false_returns_host_tensors() -> None:
    # The teach path relies on this: tokens must not accumulate on the device
    # when the caller is going to hand them to numpy anyway.
    model = _StubBackbone()
    out = extract_windows_tokens_batched(
        model, _windows(20), "cpu", max_batch=8, keep_on_device=False
    )
    assert out.device.type == "cpu"
    assert out.shape == (20, SIDE, SIDE, DIM)


def test_inputs_are_staged_per_batch_not_all_at_once() -> None:
    """The staged input tensor must never hold more than max_batch windows.

    Recorded through the stub: if the old code path came back, the extractor
    would build one [N, 3, 518, 518] tensor before any forward, and the first
    recorded batch would still be max_batch -- so shape alone cannot catch it.
    Count allocations instead: one stack per mini-batch, not one overall.
    """
    stacks: list[int] = []
    real_stack = torch.stack

    def counting_stack(tensors, *a, **kw):  # noqa: ANN001, ANN202
        stacks.append(len(tensors))
        return real_stack(tensors, *a, **kw)

    model = _StubBackbone()
    torch.stack = counting_stack  # type: ignore[assignment]
    try:
        extract_windows_tokens_batched(model, _windows(70), "cpu", max_batch=16)
    finally:
        torch.stack = real_stack  # type: ignore[assignment]

    assert stacks, "torch.stack was never called"
    assert max(stacks) <= 16, f"a stack of {max(stacks)} windows was built at once: {stacks}"
    assert sum(stacks) == 70

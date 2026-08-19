# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
"""Feature extraction must survive a CUDA OOM by backing off, not by dying.

Peak VRAM is bounded by max_batch now, but a bound chosen up front is still a
guess: another process can take the card between two batches, and a forward's
peak is not analytically predictable the way the distance matrix is. So the
extractor halves the batch and retries. These tests inject the OOM, so they
need no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from clscore.feature_extractor import extract_windows_tokens_batched, is_cuda_oom
from clscore.sw import DINO_PATCH, WINDOW_SIZE

SIDE = WINDOW_SIZE // DINO_PATCH
DIM = 8


class _OomingBackbone(torch.nn.Module):
    """Refuses any batch larger than ``fits``, the way a full card would."""

    def __init__(self, fits: int) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.fits = fits
        self.seen: list[int] = []

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n = int(x.shape[0])
        self.seen.append(n)
        if n > self.fits:
            raise torch.OutOfMemoryError(
                f"CUDA out of memory. Tried to allocate {n * 3.2:.2f} GiB"
            )
        return {"x_norm_patchtokens": torch.zeros(n, SIDE * SIDE, DIM)}


def _windows(n: int) -> list[np.ndarray]:
    return [np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), np.uint8) for _ in range(n)]


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 11.23 GiB"), True),
        # cuBLAS workspace failures arrive as a plain RuntimeError.
        (RuntimeError("CUDA error: out of memory"), True),
        (RuntimeError("cublas runtime error: out of memory"), True),
        # A host allocation failure needs a different remedy, so it must not match.
        (MemoryError("out of memory"), False),
        (RuntimeError("shape mismatch"), False),
        (ValueError("out of memory in a filename, oddly"), False),
    ],
)
def test_oom_detection_discriminates(exc: BaseException, expected: bool) -> None:
    assert is_cuda_oom(exc) is expected


def test_backs_off_and_still_processes_every_window() -> None:
    model = _OomingBackbone(fits=4)
    out = extract_windows_tokens_batched(model, _windows(20), "cpu", max_batch=32)
    assert out.shape == (20, SIDE, SIDE, DIM), "no window may be dropped by the retry"
    # batch_size halves 32 -> 16 -> 8 -> 4; each attempt takes
    # min(batch_size, remaining), so the first is 20 rather than 32.
    assert model.seen[:4] == [20, 16, 8, 4], model.seen
    # Everything after the first success is at the reduced size, and the 20
    # windows are covered exactly once by the batches that did not raise.
    assert sum(n for n in model.seen if n <= 4) == 20, model.seen


def test_the_reduced_batch_sticks() -> None:
    # Going straight back up would re-trigger the same OOM on the next batch.
    model = _OomingBackbone(fits=4)
    extract_windows_tokens_batched(model, _windows(40), "cpu", max_batch=16)
    succeeded = [n for n in model.seen if n <= 4]
    assert len(succeeded) >= 2, model.seen
    assert all(n <= 4 for n in model.seen[-len(succeeded):]), model.seen


def test_a_card_that_cannot_fit_one_window_reports_plainly() -> None:
    model = _OomingBackbone(fits=0)
    with pytest.raises(RuntimeError, match="even one window at a time") as ei:
        extract_windows_tokens_batched(model, _windows(3), "cpu", max_batch=8)
    # The remedy must be in the message, and the original error preserved.
    assert "CPU" in str(ei.value)
    assert isinstance(ei.value.__cause__, torch.OutOfMemoryError)


def test_non_oom_errors_are_not_swallowed() -> None:
    class _Broken(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(1))

        def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            raise RuntimeError("weights are corrupt")

    with pytest.raises(RuntimeError, match="weights are corrupt"):
        extract_windows_tokens_batched(_Broken(), _windows(4), "cpu", max_batch=2)

# Benchmarks

Memory-bank search benchmarks for Cls-Studio. These numbers are produced by
[`scripts/benchmark_knn.py`](scripts/benchmark_knn.py), which runs on
**synthetic input and requires no dataset**, so anyone can reproduce them on
their own hardware:

```bash
python scripts/benchmark_knn.py --device cuda:0    # GPU
python scripts/benchmark_knn.py --device cpu       # CPU (use smaller --bank-sizes)
```

## What is measured

Cls-Studio scores an image by searching its patch features against the normal
memory bank (top-k mean distance). That search is the only part of inference
that **scales with bank size** — the DINOv2 forward pass is bank-independent
(roughly 1 s per image on an RTX 3090 regardless of bank size) and is not
included here.

Two production code paths are compared, both exactly as shipped:

- **Full scan** — fp16 bank resident on the device, dense distance matrix.
- **int8 + IVF** (the default) — bank resident as int8 codes (half the
  memory), searched through IVF cluster routing (only the clusters nearest
  each query's patches are visited).

The synthetic bank is a mixture of Gaussian appearance modes, and each
measured query set draws from a few modes — mimicking how one inspection
image's patches concentrate in a small part of a real bank. On structureless
uniform-random data IVF has nothing to route; the mixture is the honest
setting.

## Test setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (24 GB) |
| Framework | PyTorch 2.x + CUDA (Windows) |
| Features | 768-dim (DINOv2 ViT-B patch features), fp16 |
| Queries | 2,048 rows per measurement (≈ one taught image after the per-image cap) |
| Search | k=5 neighbours; IVF nprobe=8 |
| Method | 5 timed runs after warmup, **median** reported, CUDA-synchronized |

## Search performance

| Bank rows | fp16 size | int8 size | Clusters | Index build | Full scan | int8+IVF | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50,000 | 73.2 MB | 36.6 MB | 781 | 0.9 s | 8.51 ms | 4.47 ms | 1.9x |
| 250,000 | 366.2 MB | 183.1 MB | 1024 | 4.6 s | 36.47 ms | 9.05 ms | 4.0x |
| 1,000,000 | 1464.8 MB | 732.4 MB | 1024 | 17.5 s | 153.58 ms | 16.87 ms | 9.1x |

**Takeaways**

- **int8 residency halves the bank's device memory** at every size.
- **IVF keeps search time nearly flat as the bank grows** — full scan is
  linear in bank rows, routed search visits only the probed clusters, so the
  advantage widens with bank size (1.9x at 50k rows, 9.1x at 1M rows here).
- The index builds in seconds and is cached on disk next to the bank
  (`ivf_index.npz`); taught images are appended to it incrementally, and a
  full rebuild only happens after substantial growth.
- Real-image banks are less cleanly separable than the synthetic mixture, so
  expect real-world routing gains to fall between the full-scan baseline and
  these numbers. Run the script on your own hardware to see where your banks
  land.

## Accuracy

Compression is enabled by default because it was **verdict-neutral on the
projects we tested** (int8 changed no OK/NG verdict; IVF nprobe=8 was
score-separation-neutral or slightly better on each of them). Those were
our own datasets, so the numbers are not reproducible by you and are
deliberately left out of this document. To evaluate accuracy on *your*
data, run the in-app **separation check**: it scores every taught image
leave-own-image-out and shows the OK/NG score distributions under the
currently active compression settings, which is exactly what the verdict
threshold is picked from. Compression can be switched off in Settings at any
time for an A/B comparison; the on-disk bank is always stored at full
precision.

## Notes

- Timings are search only and exclude image decode, DINOv2 forward, and
  heatmap rendering.
- `--bank-sizes`, `--queries`, `--k`, `--nprobe`, and `--runs` are
  configurable; run `scripts/benchmark_knn.py --help` for all options.

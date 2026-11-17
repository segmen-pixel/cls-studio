# clscore

`clscore` is the inference core of [Cls-Studio](https://github.com/segmen-pixel/cls-studio):
a training-free industrial visual classification engine built on a frozen
foundation-model backbone and a 3-tier patch memory bank.

| Module | Responsibility |
|---|---|
| `clscore.bank` | 3-tier memory bank (normal / critical / negative), append-only HITL primitive, greedy k-Center coreset |
| `clscore.scoring` | Patch kNN distance scoring, per-label attribution, alpha/beta composition |
| `clscore.incident` | Time-aware memory metadata (severity, hit-count, short/mid/long tier, freshness decay) |
| `clscore.feature_extractor` | DINOv2 + distilled BG-aware backbones, sliding-window token extraction |
| `clscore.backends` | Pluggable inference backends (torch, OpenVINO INT8 for CPU / Intel iGPU / NPU) |
| `clscore.runtime` | Host resource probing and worker planning |
| `clscore.postprocess` / `clscore.image_io` | Overlay rendering and Unicode-safe image IO |

## Install

```bash
pip install -e packages/clscore
```

OpenVINO backend: `pip install -e "packages/clscore[openvino]"`.

## Public API

```python
from clscore import Bank, score_image, __version__
```

Internals not re-exported in `clscore/__init__.py` are subject to change.

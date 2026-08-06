# Edge export format

`GET /bank/export/edge` writes a package for running inspection **on a device**
— a phone, a tablet, an embedded box — instead of talking to this server.

It is not the same thing as `GET /bank/export`. That one moves a bank between
Cls-Studio installs: it ships fp16 features, the taught source images and the
eval cache, and it can run to several GB. This one ships only what is needed
to score an image, already quantised, with no images at all.

Reading it needs a `.npy` parser and a distance computation. Nothing else —
no torch, no Python.

## Layout

The zip holds a single top-level directory, so unzipping gives you a folder to
import as a unit rather than loose files:

```
<bank-id>.anomedge.zip
  <bank-id>/
    manifest.json               the cls-studio contract (read this first)
    classes.json                host sidecar: class names and colours
    model_manifest.json         host sidecar: input size, class count, kind
    normal_codes.npy            int8    [N, D]
    normal_scale.npy            float32 [D]
    ivf_centroids.npy           float16 [K, D]     optional
    ivf_row_cluster.npy         int32   [N]        optional
    exemplars/<label>.npy       int8    [M, D]     optional, per defect label
    exemplars/<label>.scale.npy float32 [D]
```

A real export, from a 272,384-row bank at 768 dimensions: 209 MB of int8
codes (fp16 would be 418 MB), 1.6 MB of IVF centroids, 1.1 MB of row
assignments, 188 MB zipped.

Arrays are written in `.npy` format version 1.0: a short ASCII header giving
dtype, shape and order, then raw little-endian data. A reader of about 30
lines covers it.

## manifest.json

```json
{
  "format": "cls-studio-edge/1",
  "bank_id": "line3-connector",
  "encoder": {
    "model": "dinov2_vitb14",
    "dim": 768,
    "window": 518,
    "stride": 256,
    "patch": 14,
    "layers": null
  },
  "normal": {
    "rows": 271934,
    "dim": 768,
    "quantization": "int8-symmetric-per-dim",
    "dequantize": "value = code * scale[dim]"
  },
  "ivf": { "clusters": 1024, "nprobe": 8, "note": "..." },
  "exemplars": { "scratch": { "file": "scratch", "rows": 42 } },
  "verdict": { "k": 3, "threshold": 12.5, "metric": "l2" }
}
```

`encoder` is the part that matters most. The bank rows are patch features
from that specific encoder at that specific window and patch size. **A device
that produces features any other way is comparing points from two different
spaces**, and the distances will be meaningless rather than merely wrong. If
you cannot reproduce the encoder described here, the package is unusable —
check this before anything else.

`ivf` and `exemplars` are absent (`null` / `{}`) when the bank has none.
`verdict` is `null` until a verdict recipe has been saved for the bank; a
package in that state is still valid, the device simply has no threshold to
apply.

## Scoring on the device

1. Encode the input the way `encoder` describes. You get patch features
   shaped `[P, D]`.
2. For each patch, find the `k` nearest normal rows and take the mean of
   those distances. That per-patch value is the anomaly score; the image
   score is the maximum over patches.
3. Compare against `verdict.threshold` if one is present.

Distances can be computed directly against the int8 codes — dequantise a row
as `code * scale[dim]`, or fold the scale into the query once and compare in
the quantised space, which is usually faster.

### Narrowing the search with IVF

Scanning every row is fine for a small bank. When `ivf` is present:

1. Compute the distance from the query patch to each of the `K` centroids.
2. Take the `nprobe` nearest ones.
3. Restrict the search to rows whose `ivf_row_cluster[row]` is among them.

The centroids come from the server's own index, so a device that follows this
narrows to the same candidate rows the server would. Treat `nprobe` as a
starting point: raising it widens the candidate set and costs time,
lowering it does the reverse.

### Exemplars

Each `exemplars/<label>.npy` block holds representative rows for one defect
label, quantised the same way as the normal bank. They answer "which kind of
defect does this look like", not "is this a defect" — the normal bank answers
that. Distance to an exemplar block is a similarity signal, not a verdict.

## Host sidecars

`classes.json` and `model_manifest.json` exist for CoreML inspection apps
that already read those two filenames. A host that knows nothing about
Cls-Studio still gets the class names and overlay colours right, and treats the
arrays it does not understand as files it never opens.

Two things to know if you are writing such a host:

- `model_manifest.json`'s `num_classes` is what the host **displays** — normal
  plus each defect label. It is deliberately *not* the encoder's channel
  width, which lives in `feature_dim`. Reading the feature map as class
  logits produces nonsense.
- `postprocess` is `anomaly-knn`. That is the flag saying scoring means
  "distance to the bank", not "argmax over channels".

Every defect label shares one overlay colour (vermilion, `#D55E00`) rather
than getting its own: the verdict is binary, and the label only says *which
kind* once something is already flagged. Vermilion rather than purple keeps
the overlay legible under red-green colour vision deficiency.

## Size

The normal bank dominates: `rows × dim` bytes at int8, roughly half of what
fp16 would cost. A 270k-row bank at 768 dimensions is about 200 MB, which is
comfortable on a tablet and tight on a phone. Options, in the order worth
trying:

- Teach fewer images, or lower the per-image patch cap, so the bank has fewer
  rows to begin with.
- Use a smaller encoder. The BG backbones distil DINOv2 into encoders down to
  a few MB; the package format does not care which one produced the features,
  as long as `encoder` says so and the device reproduces it.
- Reduce `dim`. It costs the most per byte, and it is the one number every
  row pays for.

## Versioning

`format` is `cls-studio-edge/N`. A reader should refuse a major version it
does not know rather than guess. New optional keys may appear within a
version; ignore what you do not recognise.

# Feature Catalog

One line per feature. Details: [User Guide](user-guide.md).

## Projects & banks
- Multiple projects, one memory bank each
- Per-bank verdict recipes, exports, capacity budgets
- Project cards with live image counts and thumbnails
- Whole-project export / import as a single archive

## Building the bank
- Multi-file import, PNG/JPEG/WebP/BMP/GIF/TIFF
- Import once, relabel freely: the encoder runs once per image and
  labelling re-uses the stored features
- Label sets: a second labelling of the same images, no re-encoding
- 3-tier bank: normal / critical (per defect kind) / negative
- Defect-exemplar rectangle marking on defect images
- Per-image patch cap on every tier (coreset-reduced; marked defect
  exemplars always survive), append-only storage
- Per-image delete with exact row pruning; lossless thumbnails
- Atomic, crash-safe persistence; group-wise batch saves

## Separation check
- Leave-own-image-out scoring of every taught image, cached
- OK/NG distribution histograms with threshold line
- Auto-tuned k and α; manual override respected
- Threshold suggestion; per-bank verdict recipe with staleness flag
- PCA / contrastive projection map colored by score

## Inspection
- Multi-image queue: progress, OK/NG tally, cancel that aborts the
  in-flight request (nothing half-scored is kept or logged)
- Absolute-scale diverging heatmaps anchored at the threshold
- Per-defect-label attribution
- Server-persisted inspection log (reload-proof, capped, deletable)
- TIFF scoring with automatic display transcode
- Unified viewer controls across tabs (zoom / pan / keys)

## Performance & memory
- Bank compression default-on: int8 quantisation + IVF cluster routing
- Half resident memory; near-flat search scaling ([benchmarks](../BENCHMARKS.md))
- Compression settings in-app; on-disk data always full precision
- VRAM-aware chunking (no OOM as banks grow); probed max batch

## Lifecycle
- Time-aware exemplar weighting (severity × freshness), hit tracking
- Decay maintenance with dry-run preview
- Bank export/import as single `.clsbank.zip` incl. verdict recipe
- Whole-project export/import as `.clsproj.zip` (every bank + images +
  inspection log; import always lands as a new project)
- In-place shrink of over-grown banks (`scripts/reduce_bank.py`, dry-run
  first, exemplar-preserving)

## Platform
- Windows / macOS installers; Docker compose; CPU-only mode
- Local-first; LAN opt-in with optional shared-secret auth (`X-API-Token` header)
- Bilingual UI (ja / en); colorblind-safe visual design
- Synthetic demo image generator (`scripts/make_demo_images.py`) — try the
  full workflow with no data at hand
- REST API with OpenAPI docs at `/docs`

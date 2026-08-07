# Changelog

All notable changes to Cls-Studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Cls-Studio versioning starts at 0.1.0. Earlier 0.9.x entries that
previously appeared here belonged to the application shell this
codebase was derived from, not to Cls-Studio itself.

## [0.2.1] - 2026-08-07

### Upgrading

Projects whose defect classes are named in a script other than the Latin
alphabet — Japanese, for one — must **re-assemble** once after this release.
The Bank tab flags them as stale on its own; nothing else is required and no
data is lost. Projects whose class names are already plain ASCII are
unaffected and are not asked to rebuild anything. See the `safe_label` entry
under Fixed for why.

### Added

- A zip of images can be dropped into the bank. Nested folders are flattened,
  `__MACOSX` resource forks and dotfiles are skipped rather than reported as
  failures, and the archive guards (size, member count, path traversal) are
  the ones the project export already used.
- An import dialog: recurse into folders, add a prefix or suffix to every
  name, and review the list before anything is ingested.
- The Teach list's third column is the evaluation result — the score, plus a
  mark for an over-detection or a miss — instead of the patch count, which
  said nothing about whether the bank agrees with the operator.
- The Teach list sorts by score and filters by result, so a pass over the
  worst cases is possible without reading every row.
- Images can be selected in the Teach list and moved to over-detection
  suppression in one action.
- `stale_assignments` on the assembly status, and `label_display` on the bank
  image listing (see below).

### Fixed

- **A defect class named in a non-Latin script shared a bucket with every
  other one.** The on-disk stem is a label's primary key — it keys the feature
  tensor, the row index, the per-row metadata, the evaluation cache and the
  `.npy` file — and it was produced by collapsing every run of non-ASCII to an
  underscore and stripping it. Two Japanese class names therefore resolved to
  the same stem and were merged into one tensor, one index and one metadata
  array; the same happened to any two names sharing a trailing ASCII fragment
  (`傷A` and `汚れA` both became `A`). A digest of the original name is now
  appended whenever the collapse loses information, so the mapping is
  one-to-one. Names that were already legal stems are untouched, which is why
  only affected projects are asked to re-assemble.
- **The project card painted a thumbnail above the words "0 images".** The
  card's two halves asked different questions: the thumbnail lookup knew about
  the feature store, the counter did not, so a project that had imported
  hundreds of images but not yet assembled read as empty. The counter also
  de-duplicated by filename and ignored images left unlabelled after an
  assemble. Both halves now share one module. `.webp` was missing from the
  card's list of image types, so an all-WebP project had no thumbnail at all.
- **Re-labelling an image blanked it in the Teach tab.** The bank freezes an
  image's tier when it is assembled; the label set is live, and the Bank tab
  deliberately does not assemble when you re-label. The image lookup asked the
  live label set and refused anything that disagreed, so the thumbnail, the
  viewer and the heatmap all answered 404 until the next assemble. Unassigning
  an image did the same. The bank's row index now records which store entry a
  row range came from, and the lookup uses that.
- **Two images with the same filename scored as one.** The store allows it — a
  zip with two folders that both hold `img001.png` is ordinary — but the
  leave-own-image-out exclusion was looked up by filename, so both images were
  scored with only one of them held out. The other found its own patches in
  the bank at zero distance and read as unusually normal, which is the one
  outcome this tool exists to prevent.
- **Deleting several labelled images at once dropped only one label
  assignment.** The remaining assignments pointed at images that no longer
  existed, and the "unlabelled" counter subtracted two independently
  maintained totals and clamped the result at zero — so the Bank tab reported
  nothing left to label, ticked the labelling step complete, and assembling
  produced an empty bank. The count is a set difference now and cannot be
  driven negative; the divergence it used to absorb is reported instead.
- **Deleting one image erased the defect marks on the others.** A per-image
  delete rebuilt every surviving index entry from three fields, discarding the
  annotation rectangles and the grid mapping on all of them. Clearing a tier
  left its row index populated, so the cleared images went on being listed
  after their files were gone.
- **A deleted defect went on driving the threshold.** The cached evaluation
  sweep is keyed by name and nothing reconciled it when an image left the bank
  through the route the UI actually uses, and the cache fingerprint could not
  notice because it only covers the normal tier. The separation histogram, the
  AUROC and the auto-tuned threshold all counted images the operator had
  deleted. The endpoint now describes the bank it has, and an assemble tidies
  the file.
- **Deleting a migrated image reclaimed nothing.** The containment rule
  admitted only the store's own image directory, which is narrower than the
  set of paths the app writes, so a migrated project kept its deleted files on
  disk and kept showing one of them on the project card.
- A filename containing `#`, `%` or a space could not be served: the bank
  image URL was assembled without escaping, and the store keeps the operator's
  original filename verbatim.
- The defect-kind chips show the name that was typed. The listing used to
  return the sanitised stem, so a name in a non-Latin script never came back
  as written. The bank image listing keeps the stem as its key and carries the
  readable form alongside it as `label_display`; the edge package's
  `classes.json` carries the readable form too.
- Thumbnails and heatmaps returned to a project whose filenames contain
  characters outside `[A-Za-z0-9._-]`: the writer rewrote such a name and the
  reader rejected it, so every image in an affected project failed to load.
- An assembly can no longer land in a bank that was re-bound underneath it
  while it ran.
- The exemplar weight reaches the Inspect tab, the threshold survives a tab
  switch, and four more routes verify which bank the caller meant.
- The bank grid repaints while an import runs rather than after it, and the
  tab no longer freezes for the duration.

### Changed

- The store listing's `label` is the text the operator typed rather than the
  on-disk stem. Anything addressing a class by key should use the bank image
  listing's `label`, which is unchanged.

## [0.2.0] - 2026-08-06

### Fixed
- The browser-tab icon was still the previous product's mark: a white **A** on
  cyan, in both `favicon.svg` and all four sizes inside `favicon.ico`. The
  rename checked for the old name as text, which a single letter in an SVG
  `<text>` element and a binary `.ico` both slipped past.
- The separation check now actually applies the grouping rule. The server
  has accepted `group_mode` / `group_sep` / `group_fields` on
  `/bank/images/evaluate` and held whole groups out all along, but the UI
  never sent them, so every sweep ran leave-one-image-out however the
  control was set — the optimistic reading that control exists to prevent.
  The rule moved from the Bank tab to the Teach tab, beside the sweep it
  governs, and is passed through. **Separation numbers measured with a
  grouping rule selected before this release were leave-one-image-out
  numbers and read high.**
- The Teach tab called `preventDefault` on the space bar whichever tab you
  were looking at: its key listener was not gated on the tab being
  visible, and the component stays mounted when you leave it.

### Added
- Dropping image files anywhere on the Bank tab imports them. The tab had no
  drop handler at all — importing meant the file picker behind the Import
  button — while the Inspect tab took the same gesture. A gesture that works
  one tab over reads as broken rather than missing, so the tab now shows the
  same "drop to import" overlay while files are over it.
- `CLS_E2E_BASE` points the Playwright suite at a scratch server instead of the
  hardcoded `localhost:8791`. The suite's own setup already advised running
  against "a scratch `CLS_PROJECTS_DIR` instance", which the hardcoded address
  in three files made impossible.
- Documentation refreshed against the current four-tab workflow — the
  Bank tab's three numbered steps, the separation check and feature map on
  the Teach tab, and the validation rule that now reaches the evaluation.
  Screenshots recaptured in both languages (`docs/images/` English,
  `docs/images/ja/` Japanese); the shots of the removed staging area and of
  defect marking on the Teach tab are gone.
- `scripts/make_demo_images.py` — a synthetic demo image set (brushed
  plates, scratch / stain NG variants, probe images), so the First Run
  guide can be followed with no data at hand. The same set backs the
  README hero GIF and the docs screenshots.
- Documentation catch-up: a 10-minute First Run Walkthrough, Japanese
  mirrors for every doc under `docs/ja/`, and real screenshots (captured
  against a scratch server with synthetic demo data —
  `docs/contributing/screenshots.md` has the recapture procedure).
- Whole-project export / import: `GET /projects/{id}/export` writes one
  `.clsproj.zip` (every bank, images, masks, inspection log), and
  `POST /projects/import` restores it as a new project. Export button on
  the project card.
- `scripts/reduce_bank.py` — shrink banks that grew before the labelled
  tiers were capped, in place, keeping every defect mark. Dry run by
  default; `--apply` backs the bank directory up first.
- The Settings capacity gauge now also reports NG / negative patches and
  the all-tier resident total (they were invisible before, while being
  most of some banks' memory).
- `CLS_MAX_ARCHIVE_MB` (default 64 GB): package imports get their
  own size ceiling instead of the 200 MB per-image limit — an exported
  multi-GB bank can be imported again.

### Removed
- Binary distribution. Cls-Studio ships as source, installed with
  `install-windows.bat` / `install-macos.sh`; the bundled installer that
  built a ~3.3 GiB Windows zip has been removed along with its release
  tooling (`scripts/build_installer.py`, `scripts/windows/build_installer.bat`,
  `scripts/macos/build_installer_macos.sh`, `scripts/release/`). No such
  artifact was ever published, so no download disappears with it.
- The LGPL notice bundle under `licenses/third_party/lgpl/`. It existed to
  satisfy the obligations of shipping OpenCV's FFmpeg DLL inside that
  installer. Nothing here redistributes that binary now — pip fetches the
  wheel from PyPI onto your own machine — so the obligation does not attach
  to this project. `NOTICE` and `THIRD_PARTY_NOTICES.md` still record which
  copyleft components an install pulls in, and what they mean for anyone who
  repackages Cls-Studio as a binary.

### Changed
- `NOTICE`, `THIRD_PARTY_NOTICES.md` and `licenses/third_party/MODEL_WEIGHTS.md`
  now describe a source-only distribution: no interpreter, no wheels, and no
  model weights are redistributed. The DINOv2 weight is downloaded by
  `torch.hub` on first use rather than bundled at build time.
- The edge package is written as `.clsedge.zip`. It was `.anomedge.zip`, the
  last extension still carrying the name this project had before it was
  called Cls-Studio — `.anombank` and `.anomproj` are renamed in this same
  release and this one was missed. Nothing in Cls-Studio reads the suffix
  back: an edge package is built for a device, not for re-import here, so the
  extension is only the name the download arrives under.
- A new project's id is 12 characters instead of a 36-character UUID. Every
  file in a project sits under that id, Windows still stops at 260 characters
  for most APIs, and a taught image already carries the user's own filename --
  so the id was spending 24 characters of a shared budget on something no one
  reads. Existing projects keep their UUID directories and are recognised as
  before; nothing moves on disk.
- Bank and project packages are written as `.clsbank.zip` and `.clsproj.zip`.
  They were `.anombank` / `.anomproj`, spellings carried over from the name
  this project had before it was called Cls-Studio. Import does not key on the
  extension — it looks inside the archive — so an older package still loads.
- API error codes read `CLS-1001` rather than `ANL-1001`. The numbering and
  meanings are unchanged; only the prefix moved.
- The bank panel heading "Used for training" / 「学習に使用中」 now reads
  "Taught into this bank" / 「このバンクに教示済み」 — Cls-Studio does not
  train, and the docs say so.
- The per-image patch cap now applies to the critical / negative tiers
  too. Reduced NG images record which patches survived, so defect marks
  keep landing on the right rows; pre-existing banks are untouched.
- The DINOv2 backbone actually runs in fp16 on CUDA (it was fp32 while
  reporting fp16): scoring ~1.6x faster, resident memory a few GB lower.
  Verdict statistics move by less than 0.06% against existing banks.
- Batch teach streams features per image instead of holding the whole
  group in memory — peak host RAM no longer scales with group size
  (measured: 8 large images now peak lower than 1 did before).

### Fixed
- The Windows installer shipped without most of its dependencies. The build
  pinned torch twice -- once in the build script, once in the lockfile -- and
  when they drifted apart pip failed, but the build ignored the exit code and
  packaged a 4 GB bundle with no fastapi, uvicorn, scikit-learn or OpenCV in
  it: a download that opens on a stack trace. The build now takes the pin from
  the lockfile, stops on a failed step, and imports every runtime dependency
  inside the staged interpreter before it will package anything.
- The installer could not be built or unpacked where the path ran long. The
  torch wheel carries a vendored licence tree 173 characters deep, which
  overflows Windows' 260-character limit once a normal install root is in
  front of it. That tree is now zipped in place -- every licence still ships --
  and the build refuses to package a tree with paths over 120 characters.
- A bank built by assembling one from the store showed a black viewer and a
  "heatmap failed: 404" toast. Taught images were served only out of the
  bank's own `_images/` copy, which nothing has written since the append path
  was removed — so every bank made the way the app now makes them advertised
  image URLs that 404. Images fall back to the store, matched on both name and
  tier so a name reused across tiers cannot serve the wrong picture.
- `scripts/build_installer.py` could not run on a console that is not UTF-8
  (a Japanese Windows install, for instance): it read `pyproject.toml` with the
  locale codec and died on the first non-ASCII character. The build machine was
  exactly such a console.
- Cancel buttons actually cancel. The in-flight request is aborted, the
  button acknowledges the click, cancelling the last queued image works
  (it was a guaranteed no-op), a cancelled score is no longer written to
  the inspection log, and the run-all chain stops instead of opening the
  next stage's dialog. Blocking dialogs also survive tab switches, so a
  running job's cancel stays reachable.
- Files dropped immediately after a cancel are no longer silently
  discarded.

### Removed
- The compatibility layer for the pre-rename name. `ANOMALENS_*` environment
  variables are no longer read, and a projects directory left at
  `Documents/Anomalens/projects` is no longer adopted on startup. An install
  still keeping its data there must name the directory explicitly with
  `CLS_PROJECTS_DIR`; nothing on disk needs to move.

## [0.1.0] - Unreleased

Complete rebuild of Cls-Studio on a FastAPI + React application shell
(`apps/api` + `apps/ui`), replacing the v1 single-file vanilla-JS
frontend.

### Added

- **React UI** with Operator / Develop tabs for HITL anomaly
  inspection, served by the FastAPI backend.
- **`clscore` package** hosting the core anomaly-detection
  primitives: DINOv2 patch features, memory bank, HITL append, and
  BG-aware distilled backbone support.
- **Inference device auto-selection** — score/append run on the best
  available GPU.
- **Single-port deployment** — the app binds one port (8791); the
  Windows firewall setup script opens only that port.
- **Bank compression** — optional int8 quantization and an IVF index
  for the normal-sample memory bank (both on by default), roughly
  halving bank VRAM and speeding up large-bank scoring (measured 2x
  at ~1M rows) with identical top-k scores; configurable per instance
  via Settings or `GET/PUT /system/compression`.
- **Upload size enforcement** on every upload endpoint (bank append,
  batch teach, score, captures, staging, bank import), plus a
  declared-size guard before bank-archive extraction.

### Changed

- The DINOv2 forward pass in append/score runs off the event loop so
  the API stays responsive during inference.
- Accent color moved to teal/cyan.
- User-facing error codes now use the `CLS-` prefix.
- Toasts and the About dialog changelog are fully localized (ja/en).
- Label Studio integration in the local start scripts is opt-in
  (`CLS_START_LABEL_STUDIO=1`) and requires user-supplied
  credentials.

### Removed

- Remaining dead code inherited from the application shell this
  codebase was derived from (~7,600 lines): unreachable UI trees and
  API clients, an unregistered websocket router, and unused
  dependencies (openseadragon, zustand, immer, matplotlib and
  friends), shrinking the UI bundle by ~12%.

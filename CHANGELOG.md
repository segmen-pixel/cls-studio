# Changelog

All notable changes to Cls-Studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Cls-Studio versioning starts at 0.1.0. Earlier 0.9.x entries that
previously appeared here belonged to the application shell this
codebase was derived from, not to Cls-Studio itself.

## [0.2.2-post3] - 2026-08-09

### Added

- A VRAM budget and an idle release, under Settings → VRAM. The server used to
  size its scoring buffers against however much VRAM happened to be free on the
  card — 60% of it — and, because the allocator holds what it takes for the life
  of the process, an otherwise idle 24 GB GPU ended up with most of it reserved
  by a server sitting there doing nothing. Anything else sharing the card, a
  training run in particular, then found nothing left. The budget caps what the
  buffers are sized against (8 GB by default, adjustable, or unlimited for the
  old behaviour) and the idle release hands the reserved-but-unused memory back
  to the driver a minute after the last piece of GPU work. The model and the
  bank stay loaded, so the next inspection is no slower; an option to unload
  those too is there for when the card has to be emptied, and says what it
  costs. There is also a "release now" button, in Settings and in the GPU
  monitor.
- The GPU monitor now separates VRAM (all) from VRAM (cls-studio). The figure it
  showed before came from nvidia-smi and was the whole machine's — under Windows
  it cannot be attributed to a process — so a training run's memory read as if
  cls-studio were holding it.

### Fixed

- A project's bank could be left half-loaded onto the wrong device. Several
  places re-read which device to use after having already been told, and a
  release landing in between made them fall back to main memory silently
  instead of failing, surfacing much later as an unrelated-looking error during
  scoring.
- Loading a bank quantised the whole of it a second time and threw the result
  away whenever the cluster index on disk was still valid — the common case.

- Opening a different project left the previous one's screen behind. The bank
  tab kept its image list, its selection and its preview photograph under the
  new project's name until something happened to replace them. The selection
  was the dangerous part: images are numbered per project, so the rows left
  selected pointed at real — different — images in the project just opened, and
  labelling or deleting them was accepted. Anything still being inspected also
  kept running and filed its results under the new project. Each workspace
  screen is now rebuilt from nothing when the project changes, requests made
  before the new project is ready are refused instead of being sent against the
  old one, and the bank tab says it is loading rather than showing an empty
  list.

## [0.2.2-post2] - 2026-08-08

### Fixed

- The inspection heatmap painted red on parts with nothing wrong with them, and
  what it painted had no readable gradient. Two separate causes with the same
  effect. The inspection screen had no absolute colour scale at all, so it
  stretched the colours across each image's own range — which paints the
  hottest 1% of every image bright red, a perfect part included. And on the
  teach screen the blue end of the scale was set from a whole-image number
  rather than a per-spot one, which put it near the hottest spot of a good
  image: on a real 667-image project 99.7% of the picture came out the same
  flat blue, with the defect a hard-edged block rather than a gradient. The
  inspection screen now uses the same absolute scale as the teach screen and
  says so under the image, and both take their blue end from the per-spot
  level.

- Creating a project and opening it showed "bank select failed: an internal
  error occurred", about a second after the click. Opening a project asks the
  server three things at once, and on a brand-new project all three tried to
  create its label set at the same moment, writing to one temp file: the first
  finished and the other two found their own file gone. Every file the program
  writes now gets its own temp name, so writes that overlap no longer collide.
  Anything a previous run was interrupted mid-write is also cleaned up now
  instead of being left behind — after a power cut, an unfinished bank could
  leave a file the size of the bank sitting in the project folder, which
  nothing removed and nothing reported.

## [0.2.2] - 2026-08-08

### Fixed

- `/api/v1/health` reported version `0.1.0`. The number was written down in a
  third place and stopped being updated, so the endpoint you reach for to
  confirm which build is running answered with a version that had not been true
  since 0.2.0 — and the same stale number was stamped onto trained artifacts.
  The API, the root project and the UI package now carry one version, and a
  test fails the build if they ever disagree again.

- A defect class whose filename contains a `#` kept its cached evaluation after
  the image was deleted. Clearing a tier, or deleting single taught images,
  matched the cache entry by cutting the key at the *first* `#` — but `#` is
  what separates the image's identity from its name, so it is the *last* one,
  and a filename that carries its own `#` (`lot#3 50%.png`) was cut in the
  wrong place and matched nothing. The orphan then outlived its image, and if
  the same filename was taught again without re-assembling, the retake was
  served the deleted photo's scores and defect positions.

- A re-shot part inherited the deleted one's operational history. Re-assembling
  carries each image's hit counts, severities and consolidation tier across so
  a rebuild does not reset them, and it matched them by filename and row
  count — and the row count always matches, since every frame comes off one
  camera at one resolution. Delete an image, import its retake under the same
  name, and the old part's history was donated onto a photograph it was never
  measured on. The carry-over follows the image now; banks assembled before
  images had an identity still match by name.
- Two images sharing a filename were scored once and shown twice. The
  evaluation route resolved an image's rows by name and took the first match,
  so the sweep, the histogram, the AUROC and the result column all treated a
  pair as a single image, and they shared one cached heatmap as well. Every
  per-image lookup — on the server and in the Teach tab — now addresses the
  store entry.
- In the Teach tab, two images sharing a filename could not be moved to
  over-detection suppression. The list matched its rows to store entries by
  name, a name that resolved to two entries was left alone rather than guessed
  at, and those rows were exactly the ones most likely to need suppressing.
  They also shared one checkbox: ticking either selected both. Rows now carry
  the store entry they came from and are keyed and assigned by it.
- Re-shooting an image and importing it under the same filename served the
  deleted photo's evaluation. The cached sweep was keyed by filename, and its
  only freshness check compared the stored patch count — which on a line where
  every frame comes off one camera at one resolution always matches. The stale
  entry's ranked patch positions were then used to pick the exemplar rows and
  drive the anomaly weight, selecting arbitrary patches of the new photo
  rather than its defective ones. The cache is keyed on the image's store
  entry now, not its name. Banks assembled before that identity existed are
  unaffected and keep their cached sweep.
- A defect class whose name differed from another only in capitalisation
  silently lost its patches. The class name doubles as a filename, and Windows
  (and macOS by default) treats `Scratch.npy` and `scratch.npy` as one file:
  the second write replaced the first, the stale-file sweep saw a name that
  was still in use and removed nothing, and the next load rebuilt the class
  list from the surviving filename — one class, while the manifest still named
  two. Nothing was raised or logged, and that reload happens on an ordinary
  tab switch. Class names now fold to lower case before the disambiguating
  suffix is added, so no two can share a file. Every already-lower-case name
  is unchanged; a project using a capitalised class name re-assembles once.

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

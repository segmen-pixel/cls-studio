# Changelog

All notable changes to Cls-Studio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Cls-Studio versioning starts at 0.1.0. Earlier 0.9.x entries that
previously appeared here belonged to the application shell this
codebase was derived from, not to Cls-Studio itself.

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

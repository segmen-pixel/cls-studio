# Import / Export

Two package formats, two scopes:

| | Bank package | Project package |
|---|---|---|
| File | `.clsbank.zip` | `.clsproj.zip` |
| Scope | ONE bank of one project | the whole project directory |
| Carries | features, marks, recipe, thumbnails | every bank + images, masks, inspection log, classes |
| Import lands as | a new bank in the **open** project | a **new project** |
| Typical use | hand a trained bank to another line | move / back up a whole project |

## Project package (`.clsproj.zip`)

Everything under the project directory in one zip: `project.json`,
`classes.json`, **every** memory bank (not just the active one), the taught
source images, masks and the inspection log.

- **Export** — the download button on the project card (Projects tab), or
  `GET /api/v1/projects/{id}/export`. The archive is written STORED
  (feature arrays barely compress), so expect roughly the on-disk size of
  the project — GB-scale for real projects.
- **Import** — `POST /api/v1/projects/import` (multipart `archive`,
  optional `name`). Always creates a **new project with a fresh id**;
  nothing existing is overwritten, and the id inside the package is
  rewritten to the new one. Verdict recipes ride along per bank, so the
  copy inspects identically.
- Archive uploads have their own size ceiling,
  `CLS_MAX_ARCHIVE_MB` (default 64 GB) — separate from the 200 MB
  per-image upload limit, which does not apply to packages.

## Bank package (`.clsbank.zip`)

A bank exports as one self-contained zip — "the trained model" of
Cls-Studio, except nothing was trained:

| Included | |
|---|---|
| Feature arrays | normal / critical / negative tiers |
| Exemplar metadata | defect marks, severity, freshness sidecars |
| Manifest | per-image row index, provenance names |
| Verdict recipe | metric, k, α, threshold (`runtime_config.json`) |
| Thumbnails | lossless copies of taught images |
| Eval cache | separation-check results (derived, optional) |

Both controls sit side by side in step ① of the **Bank** tab, under the
import button:

- **Export** — writes the active bank to a zip. Disabled until the bank has
  at least one taught OK image.
- **Import** — pick a `.clsbank.zip`; it is loaded as a **new bank inside
  the project you currently have open**, and the app switches to it. There
  is no project chooser, and no existing bank is overwritten. To land a
  package in its own project, create/open that project first, then import.

An imported bank inspects identically on the target machine; the verdict
recipe rides along, so no re-thresholding is needed unless the
camera/optics differ.

Notes:
- Machine-local absolute paths are deliberately never written into the
  package (provenance keeps file names only).
- The compression index (`ivf_index.npz`) is derived data; it rebuilds on
  first use after import, so packages stay portable across devices and
  compression settings.
- Thumbnails are lossless, so re-teaching from an imported package is
  bit-identical to the original teach.

## Images

**Input formats** — PNG, JPEG, WebP, BMP, GIF, TIFF. TIFF (line-scan
cameras) is scored natively; browsers can't display it, so a transcoded
preview is generated for the UI. Non-web formats are stored as lossless
PNG thumbnails.

**Inspection results** — persisted per project (log + preview + heatmap
files under `projects/<id>/inspections/`). Individual entries can be
fetched over the API (`/api/v1/inspections`); the log is capped
(`CLS_INSPECTION_LOG_CAP`, default 200), so export anything you need
to keep.

## Whole-instance migration

For a single project, the project package above is the supported path. To
move **everything at once**, copy the entire `CLS_PROJECTS_DIR` to
the new machine (see
[Deployment — data layout](deployment.md#data-layout-and-backups)). All
state is inside; derived caches rebuild themselves.

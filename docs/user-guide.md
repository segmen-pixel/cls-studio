# User Guide

Per-screen reference. For a guided first run, start with the
[Handbook](handbook.md) or the
[First Run Walkthrough](first-run-manual.md).

The four tabs are in working order: **Projects** picks what you are working
on, **Bank** builds the memory the verdict is made against, **Teach** shows
what that memory can and cannot tell apart, and **Inspect** uses it.

## Projects

- **Project cards** show the image count and a thumbnail from the first
  image in the bank.
- A project holds one memory bank, the images that built it, its label
  sets, and its inspection log. One project per product / camera setup is
  a good default.
- Each card has an **export** button that downloads the whole project as
  one `.clsproj.zip` — the bank, the images, the marks and the
  inspection log. Importing one (`POST /api/v1/projects/import`) always
  creates a new project. See [Import / Export](import_export.md).
- Deleting a project removes its bank, captures and inspection log.

## Bank

Three numbered steps down the right-hand side. Each one ticks when it is
finished, and step ③ unticks itself again as soon as a label changes — the
tick means "the bank matches these labels", not "you pressed the button".

### ① Bank — get the images in

- **+ Import images** accepts PNG / JPEG / WebP / BMP / GIF / TIFF.
  Non-web formats are transcoded losslessly for display; scoring always
  uses the original decoded pixels.
- Importing runs the encoder **once per image** and stores its patches.
  That is the only slow part of the workflow, and relabelling never
  repeats it.
- The panel reports image count, feature rows, on-disk size and which
  encoder produced them.
- **Remove selected images** deletes exactly those images' rows.
- **Export / Import** move a whole bank between machines — see
  [Import / Export](import_export.md).

Every image is subject to the per-image patch cap
(`CLS_MAX_PATCHES_PER_IMAGE`, default 2048): each contributes at most that
many representative patches. Reduced defect images remember which patches
survived, so defect marks — existing and future — still land on the right
rows.

### ② Labels — judge them one at a time

- **Label set** names the current set of judgements. **Duplicate** makes a
  second one from it, so you can keep a stricter and a looser labelling of
  the same images side by side without re-encoding anything.
- Select rows in the list and assign a tier with the buttons or the `1` /
  `2` / `3` keys; `0` clears. The filter chips above the list count each
  tier and let you show only one of them.
  - **Normal** — a good part.
  - **Defect** — a bad one. Give it a **defect kind** (`scratch`,
    `burnt`, …) so the bank grows named defects instead of one heap; the
    kinds already in use are offered as buttons, which is what keeps a
    bank from holding both `scratch` and `scrach`.
  - **Suppress FP** — a pattern that looks anomalous but is acceptable.
    Proximity to it damps the score.
- **Mark defect** opens rectangle drawing on the selected defect image.
  The marked patches become the exemplars the α boost lifts scores
  towards. Marks are editable and re-saveable; saving replaces that
  image's previous marks.

### ③ Assemble — fold the labels into the bank

**Changing labels alone changes nothing that inspection sees.** The bank
is built from the store when you press **Assemble the bank**, and the
panel says whether the bank currently matches the labels.

## Teach

The tab that answers "can this bank tell my OK and NG apart?". The toggle
at the top of the right-hand rail switches the figure between the
separation check and the feature map; both are driven by the same sweep.

### Separation check

- **▶ Run evaluation** scores every image in the bank against it, holding
  the image being scored out. Results are cached server-side and survive
  reloads; the cache invalidates itself when the bank or the compression
  settings change.
- **AUROC** and the OK / NG histograms with the threshold line. When the
  two do not separate cleanly, the images that overlap are listed as
  chips — click one to open it.
- **Metric / k / α** are auto-tuned after each sweep (best k, then best α
  at that k). Manual adjustments are respected and not overwritten.
- **Threshold** — accept the suggestion or set your own. Persisting
  happens on the **Inspect** tab: **Save verdict settings** stores metric,
  k, α and threshold into the bank (the verdict recipe that exports
  carry). A recipe saved against an older bank state shows a *stale —
  re-check* warning.
- **Validation** decides how much is held out per image:
  - *Leave-one-image-out* (default) excludes only the image being scored.
    Near-duplicate frames of the same lot left in the bank make this read
    optimistically.
  - *Date in the filename* / *Filename prefix* derive a group from the
    name and hold the whole group out. The panel reports how many groups
    the rule found and how many images it failed to place — a rule that
    places nothing quietly degrades back into leave-one-out.

  Changing this clears the displayed results, because a grouped run is a
  different measurement — and because a grouped run is deliberately kept
  out of the server's evaluation cache.

### Feature separation map

A 2-D projection of the bank's features (PCA, or contrastive when defect
images exist). Points are coloured by score, so outliers and mislabelled
images show up as points sitting with the wrong crowd. Clicking a point
opens that image.

## Inspect

- Drop one or many images; they queue and score sequentially with
  progress, cancel, and an OK / NG tally.
- **Viewer**: wheel zoom (1–8x), drag / Space-drag / middle-button pan,
  double-click to fit, ↑/↓ to walk results, `H` toggles the heatmap,
  `Delete` removes an entry, `Esc` clears selection — the same viewer the
  Teach and Bank tabs use.
- **Heatmap** uses absolute anchors (OK median → transparent, threshold →
  full colour), so colour is comparable across images. Without a saved
  recipe it falls back to per-image relative colouring.
- **Attribution** lists per-defect-kind proximity — "how much this looks
  like *scratch*".
- **Inspection log** persists per project server-side (default cap 200,
  oldest dropped). Restored after reloads; individual entries deletable;
  Clear wipes server-side too.
- Timing chips show server processing and round-trip time per image.

## Settings

- **Memory bank size** — small / medium / large budget with usage bar. The
  bar tracks the normal tier (that is what the budget bounds); a separate
  line underneath reports the defect / suppress-FP patches, which are
  resident too, and the total across all tiers. Banks that grew large
  before the per-image cap existed can be shrunk in place with
  `scripts/reduce_bank.py` (dry-run by default, keeps every defect mark).
- **Bank compression** — int8 quantisation and IVF routing toggles plus
  the probed-cluster count. Defaults on; verdict-neutral on the projects
  we tested. Changing settings invalidates cached evaluations (they
  recompute on next use) and flags saved recipes for re-check.
- **Network** — LAN access opt-in (requires restart), shows the LAN URLs
  and security warnings. See [Deployment](deployment.md).

The **language** (日本語 / English) and **theme** toggles are not in this
dialog — both live in the app header, on the right.

## Keyboard shortcuts

| Key | Where | Action |
|---|---|---|
| `1` / `2` / `3` | Bank list | assign normal / defect / suppress-FP |
| `0` | Bank list | clear the label |
| Wheel | viewers | zoom (1–8x) |
| Space + drag / middle drag | viewers | pan |
| Double-click | viewers | fit |
| ↑ / ↓ | lists | previous / next image |
| `H` | viewers | toggle heatmap |
| `Delete` | lists | delete entry |
| `Esc` | viewers | clear selection |

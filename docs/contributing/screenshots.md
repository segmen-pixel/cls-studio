# Docs screenshots — how they are made

The images under `docs/images/` are captured automatically against a
scratch server seeded with demo data. This file is the shot list and the
recapture procedure, so a UI change never leaves the docs showing a screen
that no longer exists.

> Japanese version: [screenshots.ja.md](screenshots.ja.md)

## Ground rules

- **Demo data only, on a scratch server.** The Projects shot captures the
  whole grid, so every project name on that server ends up in a public
  repository. The capture tool cannot tell a scratch instance from a
  working one — giving it its own `CLS_PROJECTS_DIR` is on you.
- Both languages are captured at 1920×1080. English lands in
  `docs/images/`, Japanese in `docs/images/ja/`, and each doc embeds the
  set that matches the language it is written in.
- Photograph-heavy shots are stored as JPEG (the noise defeats PNG); shots
  that are mostly flat UI panels stay PNG.

## Recapture

```bash
# 1. demo images. Either synthesise a set...
python scripts/make_demo_images.py --out C:\scratch\demo_imgs
#    ...or point the tool at any folder named by prefix:
#      ok_*      taught as normal
#      ng_<kind>_<n>   taught as a defect of kind <kind>
#      probe_*   dropped on Inspect, never taught

# 2. a scratch server on its own empty data dir
set CLS_PROJECTS_DIR=C:\scratch\demo_projects
.venv-windows\Scripts\python.exe -m uvicorn apps.api.app.main:app --port 8792

# 3. capture. The tool seeds over the API the same way the Bank tab does --
#    ingest, label, assemble -- then drives the UI.
cd apps/ui
node e2e/_tools/docs-screenshots.mjs --api http://localhost:8792 \
    --images C:\scratch\demo_imgs --lang en
node e2e/_tools/docs-screenshots.mjs --api http://localhost:8792 \
    --images C:\scratch\demo_imgs --lang ja

# 4. the README hero GIF (5 frames -> stepped slideshow)
node e2e/_tools/docs-demo-gif.mjs --api http://localhost:8792 \
    --images C:\scratch\demo_imgs
python e2e/_tools/assemble-gif.py
```

Delete the demo projects between runs (`DELETE /api/v1/projects/{id}`) —
the tool creates its own and a second run against a dirty server puts four
cards in the Projects shot.

Output lands in `apps/ui/e2e/screenshots/docs/<lang>/`; copy into
`docs/images/` (English) and `docs/images/ja/` (Japanese), converting the
photograph-heavy ones to JPEG, and commit.

## Shot list (all captured)

| File | Shows | Embedded in |
|---|---|---|
| `hero.gif` | bank → separation check → OK probe → NG probe with heatmap | README, README.ja |
| `hero.jpg` | Inspect tab, burnt part scored NG, heatmap on the defect | first-run |
| `projects.png` | Projects grid with demo cards, one selected | first-run |
| `bank_label.jpg` | Bank tab: labelled list, selected image, the three steps ticked | first-run |
| `bank_marks.jpg` | Defect image open on the Bank tab with two marked rectangles | first-run |
| `check_histogram.png` | Separation check after a sweep, distributions + threshold | first-run |
| `check_map.png` | Feature separation map, points coloured by score | first-run |
| `inspect_queue.jpg` | Inspect list with OK / NG tally and heatmap viewer | (shot list only) |

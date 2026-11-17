# Cls-Studio Handbook — from zero to your first inspection

A linear walkthrough. Follow it top to bottom once; afterwards the
[User Guide](user-guide.md) is the faster per-screen reference.

## 1. Install and start

Download the source zip (Releases page, or **Code → Download ZIP**) and
extract it — no git required. Then, right in the extracted folder:

**Windows** — double-click `install-windows.bat`, then `start-windows.bat`.
The installer auto-detects your GPU (pass `cpu` or `cuda124` to override)
and the start script opens the UI in your browser when ready.

**macOS:**
```bash
bash install-macos.sh
bash start-macos.sh
```

Open **http://localhost:8791/ui/** in your browser. The header has a
language toggle (日本語 / English). Stop later with `stop-windows.bat` /
`bash stop-macos.sh`.

No GPU? Everything works on CPU; teaching and scoring are slower on large
banks.

## 2. Create a project and a bank

1. On the **Projects** screen, create a project — one per product / camera
   setup is a good default.
2. Every project holds one memory bank. If one line produces several
   distinct products, give each its own project — verdicts, thresholds and
   exports all belong to the bank.

## 3. Fill the bank

Open the project's **Bank** tab. It is three numbered steps, and each one
ticks when it is done.

1. **Import** a few dozen OK (good part) images. The encoder runs once per
   image and stores its patches — a few seconds each, and you never pay it
   again.
2. **Label** them. Walk the list and mark each one Normal, Defect (with a
   kind: "scratch", "burnt", ...) or Suppress FP. Labels are free to
   change; the features do not move.
3. **Assemble the bank.** Labels on their own change nothing that
   inspection sees — assembly is what folds them in.

Aim for coverage, not volume: include the normal variation you expect (lot
differences, lighting drift, position tolerance). A few dozen diverse OK
images beat hundreds of identical ones.

TIFF from line-scan cameras works directly; previews are transcoded
automatically.

**If you have NG (defect) images**, label them Defect with a kind, then
open one and press **Mark defect** to draw rectangles over the actual
defect areas — those marked patches become *defect exemplars* that sharpen
detection later. NG images are optional; you can start without any.

## 4. Run the separation check

Open the **Teach** tab and run the evaluation sweep. Every image in the
bank is scored against it (its own patches excluded, so OK images don't
trivially match themselves) and the OK / NG score distributions are drawn
side by side. If your images come in lots, set **Validation** to hold the
whole lot out rather than the single image — near-duplicate frames left in
the bank make the score read high.

- **Distributions well separated** — pick the suggested threshold and
  save it (**Save verdict settings**, on the Inspect tab). Done.
- **Overlapping** — usually a coverage problem. Teach more OK variation
  (the overlapping OK images tell you which kind), or mark defect
  rectangles on NG images so the α term can help. Re-run the sweep.

k (neighbours) and α (exemplar weight) are auto-tuned after each sweep; you
can adjust them manually and watch the distributions move.

The saved verdict recipe is stamped against the bank's content. If the bank
changes later, the recipe is flagged as needing a re-check.

## 5. Inspect

Switch to the **Inspect** tab and drop one or many images:

- Each image gets a score, an OK / NG verdict against your threshold, and a
  heatmap. Heatmap colors are on an absolute scale — the same color means
  the same score on every image, anchored at your threshold.
- The attribution panel shows how strongly the image resembles each known
  defect label.
- Results are persisted server-side (default: last 200 per project) and
  restored after a reload.

## 6. Keep teaching — the correction loop

This is the core habit that makes Cls-Studio improve in production:

- **False alarm** (OK flagged as NG): teach that image as OK — or, if a
  specific harmless pattern keeps triggering, add it to the **negative**
  tier so reactions near it are suppressed.
- **Miss** (NG passed as OK): teach it as critical with a label and mark
  the defect rectangles.

Both are append-only and take effect on the next image. Nothing existing is
retrained or degraded. After heavier corrections, re-run the separation
check to confirm (and re-save) the threshold.

## 7. Move a bank — or a whole project — between machines

A bank exports as a single `.clsbank.zip` — features, exemplar marks,
thumbnails, and the verdict recipe together. Import it on another Cls-Studio
instance and it inspects identically.

To move **everything about a project** (every bank, the taught images, the
inspection log), use the project package instead: the download button on
the project card writes one `.clsproj.zip`, and importing it creates a
new, identical project. See [Import / Export](import_export.md).

## Where to go next

- [User Guide](user-guide.md) — every screen and control
- [Deployment](deployment.md) — LAN access, auth token, Docker, backups
- [Troubleshooting](troubleshooting.md) — when something looks wrong

# Troubleshooting

Symptoms first, most common first. When in doubt, the API log and
`http://localhost:8791/api/v1/health` are the fastest diagnosis tools.

## Scores and verdicts

**Every image scores high (all NG) right after setup**
The bank is too small or doesn't cover normal variation yet. Teach more OK
images covering the lot/lighting/position spread, then re-run the
separation check.

**A specific harmless pattern keeps triggering**
Teach one example of it into the **negative** tier. Reactions near it are
suppressed from the next image on. Alternatively teach more OK images
containing that pattern.

**OK and NG distributions overlap in the separation check**
- Mark defect rectangles on your NG images (exemplars sharpen the α term).
- Check the projection map for mislabeled teachings (an "OK" image sitting
  inside the NG cluster is usually a labeling mistake).
- Very subtle defects may sit below what the feature resolution can
  separate — try a tighter camera crop of the inspected area.

**Scores shifted slightly after an update or settings change**
Compression settings are part of the score definition. After toggling
int8 / IVF, cached evaluations recompute and saved verdict recipes show a
re-check warning — re-run the separation check and re-save. Distribution
shifts from compression itself were verdict-neutral on the projects we
tested.

**First score after teaching is slow**
Expected: the compressed bank tensor (and occasionally the routing index)
rebuilds once after bank changes. Subsequent scores are fast. On very
large banks this one-off can take several seconds.

## Teaching

**I relabelled images but inspection did not change**
Labels live in the label set; the bank is built from them. Press
**Assemble the bank** on step ③ of the Bank tab — the step unticks itself
whenever a label changes, which is the reminder that the bank and the
labels have drifted apart.

**Teach seems stuck on huge images**
Very large images (smartphone photos, stitched line scans) extract many
thousands of patches. The per-image cap (default 2048) bounds storage but
extraction still processes the full image; prefer camera crops of the
inspected region.

**"bank is at capacity"**
The selected size budget is full. Either raise the budget in Settings
(small → medium → large), or delete taught images you no longer need
(per-image delete removes exactly that image's rows).

**Server RAM in the tens of GB / machine paging after opening a project**
Almost always a bank whose NG (critical / negative) images were taught
before those tiers had a per-image cap — a single full-resolution NG image
could contribute 100k+ patches, and the Settings gauge only reported the
normal tier. Check Settings: the line under the capacity bar shows the
labelled patch count. Shrink such banks in place with

```
python scripts/reduce_bank.py --projects-dir <your projects dir>          # dry run
python scripts/reduce_bank.py --projects-dir <your projects dir> --apply  # writes, backs up first
```

Every defect mark survives the reduction. New teaches are capped on all
tiers, so this is a one-time cleanup, not maintenance.

## Server & UI

**Cancel / 中止 seems to do nothing**
On current builds a cancel aborts the in-flight request immediately, the
button acknowledges the click ("cancelling…"), and a cancelled inspection
is neither shown nor logged. If a cancel appears ignored, you are running
a pre-0.1.0 build where the flag was only read between requests — update.

**Tab shows "no active project" / repeating 409s**
Another client (or a server restart) unbound the active bank. Open tabs
re-select automatically within seconds; if not, pick the project again.
Writes that raced a bank switch fail with 409 by design — retry after the
tab re-binds.

**"corrupt bank" (422) when opening a project**
A metadata sidecar disagrees with its feature file (typically after a hard
crash). The server repairs recoverable cases automatically on load
(severity marks for the affected label reset to defaults). If the normal
`bank.npy` itself is unreadable, restore from backup — writes are atomic,
so this indicates storage-level damage.

**UI loads but everything 404s / the page is stale**
Hard-reload (Ctrl+F5). If self-built, confirm the UI was rebuilt
(`npm run build` in `apps/ui`) and the server restarted.

## GPU

**GPU not used (device shows cpu)**
Settings → hardware: pick the CUDA device. If none is listed, the
installed PyTorch is CPU-only — rerun the installer with `cuda`.

**CUDA out-of-memory on score**
Large banks + small GPUs: pick a smaller bank size budget, or keep
compression (int8 halves the resident bank). Scoring chunks itself to fit
free VRAM, so OOM usually indicates another process holding memory —
check the in-app GPU panel.

**`nvidia-smi` shows huge memory use on Windows**
WDDM reports reserved, not allocated, memory. Trust the in-app health
panel (`torch.cuda.memory_allocated`).

## Where logs live

- API log: console / redirected file of the `uvicorn` process
  (`_api_task.log` when using the shipped Windows task scripts).
- Each API response carries a request id; server log lines include it.
- `GET /api/v1/health` — version, disk, RAM, GPU allocation.

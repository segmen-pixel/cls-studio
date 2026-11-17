# Manual E2E checklist (release pass)

The automated Playwright suite covers the core flows; this list is the
human pass before a release. Use a scratch `CLS_PROJECTS_DIR`.
Check both languages where text is asserted.

## Boot & projects
- [ ] Fresh start: server up, UI loads, no console errors
- [ ] Create project → card appears with 0 images
- [ ] Delete project → gone from list; re-creating same name works

## Bank
- [ ] Import 3 OK images → they appear unlabelled; reload → still there
- [ ] Label them Normal → step ② ticks; **Assemble** → step ③ ticks, bank
      count rises, project card thumbnail appears
- [ ] Import 1 NG, label it Defect with a new kind → the kind appears with
      a count
- [ ] Open the NG image, **Mark defect**, draw 2 rectangles, save → reopen
      shows both
- [ ] Change one label → step ③ unticks itself; assemble again → it ticks
- [ ] Remove one imported OK image → counts drop; separation cache invalidated
- [ ] TIFF import → preview renders, import succeeds

## Teach
- [ ] Run sweep → distributions render; k/α auto-tune fires once
- [ ] Set **Validation** to a filename rule → displayed results clear, the
      rule reports its group count, and the next sweep sends `group_mode`
      on `/bank/images/evaluate` (watch the network tab)
- [ ] Set it back to leave-one-image-out → no `group_` parameters are sent
      and the cached results restore
- [ ] Switch tabs (Ctrl+→) mid-sweep → the progress dialog and its 中止
      stay on screen and the cancel still works
- [ ] Reload → cached results restore without re-scoring
- [ ] Save verdict settings (Inspect tab) → persists after reload
- [ ] Teach one more image → recipe shows stale warning
- [ ] Projection map renders; point click shows image name

## Inspect
- [ ] Drop 3 images → queue processes with tally; cancel works mid-queue
- [ ] Drop ONE image, cancel while it scores → row disappears, button shows
      "cancelling…", and a reload does NOT resurrect it
- [ ] Heatmap `H` toggle, zoom/pan/fit, ↑/↓ navigation
- [ ] Reload → results restore from server log; Delete removes one
- [ ] Verdict matches threshold (one clearly-OK, one clearly-NG image)

## Settings
- [ ] Compression toggles round-trip; score timings reflect state
- [ ] Capacity budget change persists; usage bar correct
- [ ] With NG taught: the labelled-patch line appears under the bar with
      the all-tier total
- [ ] LAN opt-in warns about restart + token; opt-out restores
- [ ] Language switch (header toggle): all four tabs render in both languages

## Multi-client / failure modes
- [ ] Second browser switches bank → first tab re-binds within seconds,
      in-flight writes 409 cleanly
- [ ] Kill API mid-import → restart → the store is intact, the bank loads,
      no orphan corruption
- [ ] `CLS_API_TOKEN` set: unauthenticated calls 401, UI with token works

## Packaging
- [ ] Export bank → open a fresh project → import → package lands as a new
      bank there; identical scores on a probe image; verdict recipe carried over
- [ ] Export project (card button) → import → separate NEW project with
      identical bank row counts; original untouched
- [ ] `docker compose up --build` → both containers healthy, UI usable

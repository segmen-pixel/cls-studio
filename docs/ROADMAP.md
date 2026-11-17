# Roadmap

Direction, not commitment — items ship when they are ready. Issues and
discussions are the right place to influence priorities.

## Near term

- **Tutorial media** — a narrated video walkthrough (the README hero GIF
  and docs screenshots landed; see `docs/contributing/screenshots.md`)
- **Camera capture integration** — score directly from connected cameras
  (the captures API groundwork exists)
- **Verdict webhooks / lightweight PLC-friendly result output**

## Exploring

- **Edge export** — packaging a bank + compressed search for on-device
  inspection (mobile / embedded targets)
- **Novelty gating** — teach-time filtering that only stores genuinely new
  patches, slowing bank growth further
- **Multi-bank routing** — one drop, automatic bank selection by product

## Non-goals

- Cloud training / hosted SaaS — Cls-Studio stays local-first
- Built-in neural network training — that is the sibling project
  seg-studio's territory; Cls-Studio deliberately does not train
- Enterprise auth (SSO/RBAC) — put a reverse proxy in front instead

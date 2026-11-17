# Deployment

Cls-Studio is local-first: the default configuration binds to `127.0.0.1`
only, sends no images anywhere, and needs no account. This page covers
everything beyond that default.

## Data layout and backups

All user data lives under `CLS_PROJECTS_DIR`
(default `~/Documents/ClsStudio/projects`):

```
projects/<project-id>/
  banks/<bank-id>/
    bank.npy               # normal-tier features
    critical/<label>.npy   # defect exemplars (+ .meta.npz sidecars)
    negative/<label>.npy
    bank_meta.json         # manifest incl. per-image row index
    runtime_config.json    # saved verdict recipe
    eval_cache.json        # separation-check cache (derived)
    ivf_index.npz          # compression index (derived, auto-rebuilt)
    _images/               # lossless thumbnails of taught images
    _staging/              # staged-but-untaught drops
  captures/
  inspections/             # persisted inspection log
```

**Backup = copy the projects directory** while the server is idle. All
writes are atomic (tmp-then-replace), so even a mid-write copy yields a
loadable state; derived files (`eval_cache.json`, `ivf_index.npz`) can
always be regenerated. A single bank can be moved by itself via
[export / import](import_export.md).

Keep `CLS_PROJECTS_DIR` **outside** the repository tree — the server
refuses to start otherwise (protects user data from repo operations).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CLS_PROJECTS_DIR` | `~/Documents/ClsStudio/projects` | Data root |
| `CLS_DB_PATH` | alongside data root | SQLite metadata |
| `CLS_MODELS_DIR` | `<repo>/models` | Reserved model directory; the server only creates `<dir>/registry` at startup. DINOv2 weights do **not** land here — torch.hub caches those under `TORCH_HOME` |
| `CLS_TORCH_DEVICE` | `auto` | `auto` / `cpu` / `cuda:N`; also switchable at runtime via `PUT /api/v1/hardware/torch/device` |
| `CLS_API_TOKEN` | unset | Shared secret required in the `X-API-Token` header on every API call when set |
| `CLS_HOST` | `127.0.0.1` | Bind address |
| `CLS_MAX_PATCHES_PER_IMAGE` | `2048` | Per-image patch cap, all tiers (0 = uncapped). NG defect marks survive the reduction — annotated rows are always kept |
| `CLS_CAPACITY_SMALL/_MEDIUM/_LARGE` | 350k / 1.4M / 4M | Row ceilings behind the Settings size budgets |
| `CLS_INSPECTION_LOG_CAP` | `200` | Inspection results kept per project |
| `CLS_MAX_UPLOAD_TOTAL_MB` | `2048` | Total size (MB) accepted per multi-file upload request; 200 MB per single file still applies |
| `CLS_MAX_UPLOAD_FILES` | `1024` | Maximum files accepted in one multi-file upload request |
| `CLS_MAX_ARCHIVE_MB` | `65536` | Size ceiling for package imports (bank / project zips) — these are streamed to disk, so the per-image limits above do not apply to them |
| `CLS_COMPILE_MODE` | `reduce-overhead` | `torch.compile` mode for the feature extractor: `default` / `reduce-overhead` / `max-autotune`, or `off` to skip compilation |
| `CLS_RUNTIME_DIR` | `~/.cls-studio/runtime` | Directory for the runtime process registry (`procs.json`) |
| `CLS_OPENVINO_IR_DIR` | `./ov_ir` | Where the OpenVINO backend looks for the `.xml` / `.bin` IR pair |
| `CLS_BG_BACKBONE_DIR` | unset | Directory holding distilled BG-aware encoder checkpoints. No default: BG backbones refuse to load unless it is set |
| `CLS_STATE_DIR` / `CLS_DATA_DIR` | unset | Legacy aliases kept for older launch scripts — when `CLS_PROJECTS_DIR` is unset the data root becomes `<value>/projects` |

## LAN access

For operator stations on the shop-floor network:

1. Settings → Network → enable **LAN access** (writes the opt-in, requires
   a restart; the server then binds `0.0.0.0`).
2. Set `CLS_API_TOKEN` — without it every LAN client has full,
   unauthenticated read-write access. The server logs a warning at startup
   when LAN access is on without a token.
3. Open TCP 8791 in the OS firewall (Windows installer ships a helper:
   `scripts/windows/_setup_firewall.ps1`).

Multiple clients can use the server simultaneously. The active bank is
bound per request (`X-Bank-Binding`): if another client switches banks
mid-operation, affected writes fail with a clear 409 instead of landing in
the wrong bank, and open tabs re-bind themselves automatically.

**Do not expose Cls-Studio directly to the internet.** For anything beyond
a trusted LAN put it behind a reverse proxy that terminates TLS and adds
real authentication — see [Reverse proxy](#reverse-proxy) below and
[SECURITY.md](../SECURITY.md).

## Reverse proxy

Cls-Studio has no user accounts and no TLS of its own. Any deployment
reachable from more than one trusted machine needs a proxy in front that
terminates TLS and performs the authentication. The proxy is the security
boundary; the app stays on loopback behind it.

Ground rules:

- Keep `CLS_HOST=127.0.0.1` when the proxy runs on the same host. An
  explicit `CLS_HOST` always wins over the Settings LAN opt-in, so
  setting it is the reliable way to keep the app off the network. Use
  `0.0.0.0` only when the proxy runs on another machine, and then firewall
  TCP 8791 down to the proxy address.
- Proxy the whole origin, not just `/api`. Which origin depends on how you
  run it:
  - Native install — `http://127.0.0.1:8791`: one port serves the UI at
    `/ui/` and the REST surface at `/api/v1/*`.
  - Docker compose — `http://127.0.0.1:5173`: the UI container serves the
    SPA at `/` and already forwards the API paths to the API container.
- No WebSocket plumbing is needed. This release exposes plain HTTP request /
  response only — there is no WebSocket endpoint and no streaming surface, so
  the proxy needs no `Upgrade` handling.
- Size the request body limit for a whole multi-file drop, not for one
  image. A batch import sends every selected file in a
  *single* multipart request: the 200 MB cap bounds one file inside that
  request, while the request as a whole is bounded by
  `CLS_MAX_UPLOAD_TOTAL_MB` (default 2048 MB) and
  `CLS_MAX_UPLOAD_FILES` (default 1024). Set the proxy limit to the
  total cap, and lower both together if 2 GB bodies are more than the proxy
  should accept.
- Set `CLS_API_TOKEN` and have the proxy inject the `X-API-Token`
  header, so the secret never reaches the browser — the shipped UI does not
  send the header itself. Requests to `/api/v1/*` without a valid token are
  rejected with `401`.

nginx, native install:

```nginx
server {
    listen 443 ssl;
    server_name cls-studio.example.internal;

    ssl_certificate     /etc/ssl/certs/cls-studio.crt;
    ssl_certificate_key /etc/ssl/private/cls-studio.key;

    # Authentication happens here — cls-studio has none of its own.
    auth_basic           "cls-studio";
    auth_basic_user_file /etc/nginx/cls-studio.htpasswd;

    # Whole multi-file upload request, not one file: keep in sync with
    # CLS_MAX_UPLOAD_TOTAL_MB (default 2048).
    client_max_body_size 2048M;

    location / {
        proxy_pass http://127.0.0.1:8791;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;

        # Shared secret injected server-side; never seen by the browser.
        proxy_set_header X-API-Token "<value of CLS_API_TOKEN>";
    }
}
```

Caddy equivalent:

```caddy
cls-studio.example.internal {
    basic_auth {
        operator <bcrypt-hash>
    }
    # Whole multi-file upload request; keep in sync with
    # CLS_MAX_UPLOAD_TOTAL_MB (default 2048).
    request_body {
        max_size 2048MB
    }
    reverse_proxy 127.0.0.1:8791 {
        header_up X-API-Token "<value of CLS_API_TOKEN>"
    }
}
```

## Docker

```bash
docker compose up --build
```

- UI on `http://localhost:5173/`, API proxied by the UI container.
- User data lives on the named volume `cls-studio-data` (survives rebuilds).
- All ports publish on `127.0.0.1` only; edit `docker-compose.yml`
  deliberately if you need more.

## GPU notes

- Device comes from `CLS_TORCH_DEVICE` (or `PUT
  /api/v1/hardware/torch/device` at runtime); switching takes effect
  immediately — the model and cached bank tensors reload on the new device.
- VRAM footprint is dominated by the normal bank: with default compression
  the bank sits resident as int8 (≈ 0.75 KB per patch row); the size
  budgets in Settings translate to roughly 0.26 / 1.05 / 3 GB resident at
  the small / medium / large ceilings.
- On Windows (WDDM) `nvidia-smi` memory numbers are misleading; the
  in-app health panel reports `torch.cuda.memory_allocated`, which is
  accurate.

## Operational behaviour worth knowing

- **First score after a teach or restart** rebuilds the compressed bank
  tensor and (rarely) the IVF index — expect a one-off multi-second delay
  on large banks; subsequent scores are fast.
- **Compression settings changes** apply on the next score, invalidate the
  separation-check cache, and mark saved verdict recipes for re-check. The
  on-disk bank always stays full precision.
- The **inspection log** is capped per project; export anything you need
  long-term.

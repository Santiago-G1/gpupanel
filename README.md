# GPU panel

One Tesla P100 (16 GB), four consumers, one owner at a time. This is the web
panel that arbitrates between them: it watches what the card is really doing —
`nvidia-smi` on the Proxmox node, not a flag of its own — and moves the other
consumers out of the way when llama.cpp needs the VRAM back.

```
browser ──► Caddy ──► https://gpu.<domain> ──► <ai-host>:8099
                                                  │
                                    SSH ──────────┼──► PVE node   nvidia-smi, power limit, pct exec
                                                  ├──► Immich     immich_machine_learning, redis, psql
                                                  └──► frontend   Jellyfin encoding.xml
                                    HTTP ─────────┼──► llama.cpp  /v1/models, /models/unload
                                                  └──► ComfyUI    /system_stats, /queue
```

The panel runs on the AI host, next to llama.cpp: both are containers on the
`ai-stack` network, and the panel talks to the router by its LAN address because
inside its own container `127.0.0.1` is itself. Nothing else is co-located — the
cards are driven over SSH, which is why the key is the mount that matters.

## The consumers

| Consumer | Lives on | Occupies | What the panel does to it |
| --- | --- | --- | --- |
| llama.cpp | AI host (CT, router mode) | up to ~15.8 GB, all of it with a large model | unloads every resident model over the router API |
| Immich ML | Immich host | ~6.8 GB, CUDA: it cannot share the card | `docker stop` / `start immich_machine_learning` |
| Jellyfin | frontend host | NVENC session, small | rewrites `<HardwareAccelerationType>` to `vaapi` (iGPU) or back to `nvenc` |
| ComfyUI | AI host (on-demand stack) | ~4–6 GB | `docker start` / `stop` through `pct exec` of the AI container |

The panel reads every card the node reports and attributes the processes by
UUID, so a second one is not mistaken for an idle first one. The power controls
still read GPU 0, which is the card the limit is set on.

Two rules decide everything:

1. **llama.cpp comes first.** While a model is `loading` or `loaded`, the other
   CUDA consumers step aside — Immich ML is stopped and Jellyfin drops to the
   iGPU, which keeps hardware transcoding for free. A model in `loading` already
   counts: the router asks for the VRAM before the status reads `loaded`, and a
   13.6 GB model does not fit next to ComfyUI.
2. **Nothing stays on the card idle.** Each consumer has a release timer, so an
   abandoned ComfyUI or a forgotten LLM hands the P100 back on its own.

## Modes

The four buttons are fixed sequences, not settings: each one leaves the card in
a consistent state and records the action in the state file.

| Mode | llama.cpp | Immich ML | Jellyfin | ComfyUI |
| --- | --- | --- | --- | --- |
| `llm` | untouched | stopped | iGPU (VAAPI) | stopped |
| `media` | unloaded | started | P100 (NVENC) | stopped |
| `balanced` | untouched | started | P100 (NVENC) | stopped |
| `image` | unloaded | stopped | iGPU (VAAPI) | started |

`balanced` is the only one that deliberately leaves the card oversubscribed: it
is for a small LLM next to Immich's face detection, and it is the one that can
fail with a `cudaMalloc` if a large model is asked for later.

## Auto-switch

A thread re-reads `nvidia-smi` every 20 seconds and keeps the highest utilisation
of all cards as the idle clock. Timers, in minutes, all `0 = off`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `llama_idle_min` | 60 | unload the resident models and give the card back |
| `ml_idle_min` | 0 | stop Immich ML, but only with an empty ML queue |
| `comfy_idle_min` | 10 | stop ComfyUI, but only with an empty prompt queue |

The queue checks are the point: a release never kills work in flight. The last
automatic action and the reason are shown on the dashboard and kept in the state
file, so a card that changed owner overnight can be explained afterwards.

## Layout

| Path | What it is |
| --- | --- |
| `app.py` | The whole service: config from the environment, auth, the GPU probes, the four modes and the auto-switch tick. |
| `static/index.html` | The dashboard. Polls `/api/status`, never renders a model the router was not allowed to name. |
| `static/howto.html` | The guide served at `/howto`, same auth. |
| `static/login.html` | The only page without a session, with its CSS inline so it loads unauthenticated. |
| `static/panel.css` | Shared by the dashboard and the guide. |
| `docker-compose.yml` | The stack: build, published port, SSH key, state volume, health check. |
| `.env.template` | Every variable the Compose file reads. Never a real secret. |

## Run it

```sh
cp .env.template .env      # fill in the credentials and the hosts
docker compose up -d --build
docker compose logs -f gpu-panel
```

The SSH key under `/root/.ssh` must reach `PVE_HOST`, `IMMICH_HOST` and
`FRONTEND_HOST` non-interactively (`BatchMode=yes` is used throughout), and the
user on each must be able to run `nvidia-smi`, `docker` and `pct exec`.

In the lab, Ansible (`homelab_WIP`, `roles/stacks`) checks this repository out
at `/opt/docker/gpu-panel` on the `main` branch and renders `.env` from the
encrypted vault with mode `0600`. The checkout is the deployed copy, so local
edits there are discarded on the next run — the panel is never edited on the
host.

```sh
# from the homelab repository's ansible/ directory
ansible-playbook playbooks/stacks.yml --limit ai -e '{"stacks_only":["gpu-panel"]}'
```

## API

Browsers get a signed cookie (HMAC-SHA256, 7 days, revocable) from `/api/login`.
Automation sends the token instead:

```sh
curl -H "Authorization: Bearer $PANEL_TOKEN" http://<ai-host>:8099/api/status
```

Cookie-authenticated writes must come from the panel's own origin; a cross-origin
`POST` is refused with `403`. The token bypasses that check, which is what makes
it usable from scripts.

| Endpoint | Method | What it does |
| --- | --- | --- |
| `/api/session` | GET | Who the caller is. The only probe that never leaves the container. |
| `/api/status` | GET | Every card the node reports, the processes on them attributed by UUID, `auto`, Immich, Jellyfin, the router's models and ComfyUI's stats. |
| `/api/mode/{llm,media,balanced,image}` | POST | The table above. |
| `/api/auto` | POST | `enabled` and the three idle timers, `0…1440` minutes. |
| `/api/power` | POST | `nvidia-smi -pl`, bounded to the range the card reports (`125–250 W` on a P100). |
| `/api/llama/unload` | POST | Unload every resident model. |
| `/api/llama/logs?lines=200` | GET | `journalctl` of the unit through `pct exec`, capped at 2000 lines. |
| `/api/comfy/{start,stop,restart}` | POST | The ComfyUI container on the AI host. |
| `/api/immich/ml/{start,stop,restart}` | POST | `immich_machine_learning` on the Immich host. |
| `/api/jellyfin/accel/{nvenc,vaapi,none}` | POST | Rewrite `encoding.xml` and restart Jellyfin. |

Every `502` carries the failing command's own stderr, because the useful part of
a broken SSH hop is what the far side said.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `PANEL_USER` | `admin` | |
| `PANEL_PASSWORD` | — | Required. Rotating it logs every browser out. |
| `PANEL_TOKEN` | — | Required. Rotating it does the same. |
| `LLAMA_URL` | — | Required, LAN address of the router. |
| `PVE_HOST`, `IMMICH_HOST`, `FRONTEND_HOST` | — | Required, `user@host`. |
| `JELLYFIN_XML` | — | Required, absolute path on the frontend host. |
| `AI_CTID` | — | Required, Proxmox VM ID of the AI container. |
| `LLAMA_LOG_UNIT` | `llama-server` | The systemd unit inside that container. |
| `JELLYFIN_FALLBACK` | `vaapi` | Where Jellyfin goes while llama.cpp owns the card; `none` restores CPU. |
| `COMFY_URL` | empty | LAN address; empty hides ComfyUI's stats but not its controls. |
| `COMFY_UI_URL` | empty | Link out to its web UI. |
| `COMFY_CONTAINER` | `comfyui` | Container name inside the AI container. |
| `LLAMA_PUBLIC_MODELS` | empty | Comma-separated substrings. Empty names every model the router serves. |
| `SESSION_TTL` | `604800` | Seconds. |
| `LOGIN_DELAY` | `0.4` | Seconds burned on a bad password. |
| `STATE_PATH` | `/state/auto.json` | Inside the named volume. |

`LLAMA_PUBLIC_MODELS` is the only display filter: unlisted models still load and
unload through the panel, they are just never printed — a count stands in for
them. That keeps a model the lab does not advertise out of a screenshot.

## State

`/state/auto.json`, in the `gpu_panel_state` volume, holds what has to survive a
recreate: the auto-switch settings, the last action and when it happened, the
revoked session nonces (sessions are stateless, so a logout has to remember one)
and the Jellyfin device to restore. Losing it costs a re-login and the timers
going back to their defaults; the card is where the truth is.

## Secrets

Three, and none of them is in this repository: the panel's password, its token
(`openssl rand -hex 32` each) and the SSH key that reaches the three hosts. In
the lab the first two live in the Ansible vault and reach the container through
the rendered `.env`; the key is `/root/.ssh` on the AI host, mounted read-only.

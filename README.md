# GPU panel

One card (or several), a set of things that want it, one owner at a time. This
is the web panel that arbitrates between them: it watches what the card is
really doing — `nvidia-smi` over SSH, not a flag of its own — and moves the
other consumers out of the way when llama.cpp needs the VRAM back.

```
browser ──► reverse proxy ──► https://gpu.<domain> ──► <host>:8099
                                                          │
                                    SSH ──────────────────┼──► GPU host    nvidia-smi, power limit
                                                          ├──► AI host     llama.cpp logs, ComfyUI
                                                          ├──► Immich host ML container, redis, psql
                                                          └──► Jellyfin    encoding.xml
                                    HTTP ─────────────────┼──► llama.cpp   /v1/models, /models/unload
                                                          └──► ComfyUI     /system_stats, /queue
```

The panel talks to the router by address, because inside its own container
`127.0.0.1` is itself, and drives everything else over SSH — which is why the
key is the mount that matters. Nothing in the list is mandatory except the card
host and the router: see [Adapting it to your setup](#adapting-it-to-your-setup).

## Requirements

| What | Why |
| --- | --- |
| Docker with Compose v2 | the only runtime dependency; nothing is installed on the hosts |
| `nvidia-smi` on `GPU_HOST`, over SSH with a key | the cards, their processes and the power limit |
| A llama.cpp router (`/v1/models`, `/models/unload`) | llama.cpp in router mode or llama-swap; `LLAMA_API_KEY` if it wants a key |
| SSH to the consumers' hosts | `docker` there for Immich, a file rewrite for Jellyfin, `pct exec` only with `AI_CTID` |
| journald on `AI_HOST` | only for the log tail endpoint |

No Python, no agent and no client library goes on the hosts: every consumer is
either an HTTP call or a command over SSH.

## The consumers

| Consumer | Reached by | Occupies | What the panel does to it |
| --- | --- | --- | --- |
| llama.cpp | `LLAMA_URL` (HTTP) | a large model takes the whole card | unloads every resident model over the router API |
| Immich ML | `IMMICH_HOST` (SSH) | CUDA: it cannot share the card | `docker stop` / `start` on `IMMICH_ML_CONTAINER` |
| Jellyfin | `JELLYFIN_HOST` (SSH) | an NVENC session, small | rewrites `<HardwareAccelerationType>` to `vaapi` (integrated GPU) or back to `nvenc` |
| ComfyUI | `AI_HOST` + `AI_CTID` | enough to matter, and it grows | `docker start` / `stop` on `COMFY_CONTAINER` |

The panel reads every card the host reports and attributes the processes by
UUID, so a second one is not mistaken for an idle first one. The power controls
still read GPU 0, which is the card the limit is set on.

Two rules decide everything:

1. **llama.cpp comes first.** While a model is `loading` or `loaded`, the other
   CUDA consumers step aside — Immich ML is stopped and Jellyfin drops to the
   integrated GPU, which keeps hardware transcoding for free. A model in
   `loading` already counts: the router asks for the VRAM before the status reads
   `loaded`, and a big model does not fit next to ComfyUI.
2. **Nothing stays on the card idle.** Each consumer has a release timer, so an
   abandoned ComfyUI or a forgotten LLM hands the card back on its own.

## Card inventory

A card that does not initialise is silent everywhere else: no driver error
reaches the panel, `nvidia-smi` just answers with one card fewer, and everything
still works on the ones that are there. This is how the second P100 of the
reference deployment fails after a kernel update, so the panel keeps the set of
UUIDs it has seen and compares it with every observation (the 20-second tick, plus each
dashboard load):

| Situation | What the dashboard shows |
| --- | --- |
| `count == GPU_EXPECTED` | a quiet `2 of 2` pill |
| `count < GPU_EXPECTED` | `1 of 2 · 1 card missing`, with the reason it cannot be fixed from here |
| a card appears | `new card`, its name, its UUID and when it came up |
| `GPU_EXPECTED=0` | only changes are reported; nothing is called missing |

A change is written to the state file and to the last-action line, so a card
that vanished overnight has a timestamp in the morning. The comparison is on the
UUID set, not the count: a card that is swapped for another is a change too, and
the order `nvidia-smi` reports them in is not.

## Modes

The four buttons are fixed sequences, not settings: each one leaves the card in
a consistent state and records the action in the state file.

| Mode | llama.cpp | Immich ML | Jellyfin | ComfyUI |
| --- | --- | --- | --- | --- |
| `llm` | untouched | stopped | integrated GPU (VAAPI) | stopped |
| `media` | unloaded | started | P100 (NVENC) | stopped |
| `balanced` | untouched | started | P100 (NVENC) | stopped |
| `image` | unloaded | stopped | integrated GPU (VAAPI) | started |

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
| `LICENSE` | MIT: use it, fork it, keep the notice. |
| `tests/` | `pytest` suite: the panel's promises, with the probes replaced by stubs. |
| `pyproject.toml` | Ruff and pytest configuration (`requirements-dev.txt` for the tools). |
| `.github/workflows/ci.yml` | Lint, tests on 3.10 and 3.12, and a boot of the built image. |

## Run it

```sh
cp .env.template .env      # fill in the credentials and the hosts
docker compose up -d --build
docker compose logs -f gpu-panel
```

The SSH key under `/root/.ssh` must reach every host you configured
non-interactively (`BatchMode=yes` is used throughout): `nvidia-smi` on
`GPU_HOST`, `docker` where the containers are, `pct exec` only if you set
`AI_CTID`.

In the lab, Ansible (`homelab_WIP`, `roles/stacks`) checks this repository out
at `/opt/docker/gpu-panel` on the `main` branch and renders `.env` from the
encrypted vault with mode `0600`. The checkout is the deployed copy, so local
edits there are discarded on the next run — the panel is never edited on the
host.

```sh
# from the homelab repository's ansible/ directory
ansible-playbook playbooks/stacks.yml --limit ai -e '{"stacks_only":["gpu-panel"]}'
```

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
ruff check .          # lint
pytest                # tests
```

The tests never touch a GPU, a host or an SSH key: `tests/conftest.py` imports
`app` with a fake environment and each test replaces the probe it cares about, so
"a card that answers `N/A`", "a deployment with no Immich" or "an unreachable
host" are three ordinary cases. `tests/test_ui.py` checks the other half — that
every element the dashboard looks up exists in the markup, and that the inline
JavaScript parses.

CI runs the same two commands on Python 3.10 and 3.12, and then builds the image
and boots it with no card to prove that it answers `/api/session` and fails
cleanly (`502`) everywhere else.

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
| `/api/status` | GET | Every card the host reports, the processes on them attributed by UUID, `inventory` (cards vs expected), `configured` (which consumers exist here), `auto`, Immich, Jellyfin, the router's models and ComfyUI's stats. |
| `/api/mode/{llm,media,balanced,image}` | POST | The table above. |
| `/api/auto` | POST | `enabled` and the three idle timers, `0…1440` minutes. |
| `/api/power` | POST | `nvidia-smi -i <index> -pl`, per card. Bounded by what the cards report, and the answer says which ones refused. |
| `/api/llama/unload` | POST | Unload every resident model. |
| `/api/llama/logs?lines=200` | GET | `journalctl` of the unit on `AI_HOST` (through `pct exec` when `AI_CTID` is set), capped at 2000 lines. |
| `/api/comfy/{start,stop,restart}` | POST | The ComfyUI container on `AI_HOST`; skipped when `COMFY_CONTAINER` is empty. |
| `/api/immich/ml/{start,stop,restart}` | POST | `IMMICH_ML_CONTAINER` on `IMMICH_HOST`; skipped when it is not configured. |
| `/api/jellyfin/accel/{nvenc,vaapi,none}` | POST | Rewrite the config and restart `JELLYFIN_CONTAINER`; skipped when not configured. |

Every `502` carries the failing command's own stderr, because the useful part of
a broken SSH hop is what the far side said. A timeout is one of those answers:
`timed out after 30s`, never a traceback.

`/api/status` is the document a widget or a script parses, so here it is,
abridged and with the numbers shortened:

```json
{
  "gpu": {
    "index": "0", "uuid": "GPU-00000000-…", "name": "Tesla P100-PCIE-16GB",
    "driver": "580.159.04", "mem_used": "14478", "mem_total": "16384",
    "util": "0", "util_max": 0, "temp": "39",
    "power_draw": "36.71", "power_limit": "200.00",
    "power_min": "125.00", "power_max": "250.00", "power_limits": ["200.00"],
    "clk_sm": "1328", "clk_mem": "715", "pcie_gen": "3", "pcie_width": "16",
    "processes": [{"pid": "3331196", "name": "…/llama-server", "mem_mib": 14478, "gpu": "0"}]
  },
  "gpus": [ "…one entry per card, the shape of `gpu` above…" ],
  "auto": {
    "enabled": true, "llama_idle_min": 60, "ml_idle_min": 0, "comfy_idle_min": 10,
    "idle_seconds": 42, "llm_blocked": true,
    "last_action": "LLM in use: Immich ML stopped (card free for llama.cpp)",
    "last_action_at": "2026-11-05 08:25:32"
  },
  "inventory": {
    "count": 2, "expected": 2, "missing": 0, "last_change": null,
    "present": [{"uuid": "GPU-00000000-…", "index": "0", "name": "Tesla P100-PCIE-16GB"}]
  },
  "configured": {"immich": true, "jellyfin": true, "comfyui": true, "comfyui_api": true, "llama_logs": true},
  "immich_ml": "running",
  "immich_ffmpeg": {"accel": "nvenc", "preferredHwDevice": ""},
  "jellyfin_accel": "vaapi",
  "llama": [{"id": "some-model", "status": "loaded"}],
  "llama_counts": {"total": 7, "loaded": 1},
  "comfyui": {"state": "created"}
}
```

The readings are strings exactly as `nvidia-smi` prints them, `N/A` included; the
two computed fields are `util_max` (an int) for the idle clock and `power_limits`
(one per card) for the dashboard. A consumer that is not configured answers
`"unconfigured"` instead of an object, so a client can tell it apart from a
consumer that is down.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `GPU_HOST` | — | **Required.** `user@host` where `nvidia-smi` answers. Proxmox not needed. |
| `LLAMA_URL` | — | **Required.** Base URL of the router. |
| `PANEL_PASSWORD` | — | **Required.** Rotating it logs every browser out. |
| `PANEL_TOKEN` | — | **Required.** Rotating it does the same. |
| `PANEL_USER` | `admin` | |
| `SESSION_TTL` | `604800` | Seconds. |
| `LOGIN_DELAY` | `0.4` | Seconds burned on a bad password. |
| `STATE_PATH` | `/state/auto.json` | Inside the named volume. |
| `AI_HOST` | `GPU_HOST` | Where llama.cpp and ComfyUI run. |
| `AI_CTID` | empty | Proxmox VM/CT id: commands run through `pct exec` there. Empty = on the host itself. |
| `LLAMA_LOG_UNIT` | `llama-server` | The systemd unit whose journal is shown. |
| `LLAMA_PUBLIC_MODELS` | empty | Comma-separated substrings. Empty names every model the router serves. |
| `GPU_EXPECTED` | `0` | Cards the host should have. `0` reports what `nvidia-smi` answers without calling anything missing. |
| `IMMICH_HOST` | empty | Empty = no Immich. |
| `IMMICH_ML_CONTAINER` | `immich_machine_learning` | |
| `IMMICH_PG_CONTAINER` | `immich_postgres` | Read for Immich's own ffmpeg settings. |
| `IMMICH_REDIS_CONTAINER` | `immich_redis` | |
| `IMMICH_QUEUE_PREFIX` | `immich_bull` | Queue prefix, without the `:` or the queue name. |
| `IMMICH_ML_QUEUES` | `faceDetection,smartSearch,ocr` | Queues that mean the ML container has work. |
| `JELLYFIN_HOST` | empty | Empty = no Jellyfin. Needs `JELLYFIN_XML` too. |
| `JELLYFIN_XML` | empty | Absolute path of `encoding.xml` on that host. |
| `JELLYFIN_CONTAINER` | `jellyfin` | Restarted around the rewrite. |
| `JELLYFIN_FALLBACK` | `vaapi` | Where Jellyfin goes while llama.cpp owns the card; `none` restores CPU. |
| `COMFY_CONTAINER` | empty | Empty = no start/stop control. |
| `COMFY_URL` | empty | Its API, for the queue depth and the VRAM it sees. |
| `COMFY_UI_URL` | empty | Link out to its web UI. |

`LLAMA_PUBLIC_MODELS` is the only display filter: unlisted models still load and
unload through the panel, they are just never printed — a count stands in for
them. That keeps a model the lab does not advertise out of a screenshot.

## Adapting it to your setup

The panel is opinionated about *which* consumers exist — llama.cpp, Immich ML,
Jellyfin and ComfyUI — and not at all about the machines they live on:

| You have | Set |
| --- | --- |
| A card + the router | `GPU_HOST`, `LLAMA_URL`. Nothing else. |
| llama.cpp on another host | `AI_HOST` |
| …inside a Proxmox container | `AI_HOST` (the node) + `AI_CTID` |
| Immich | `IMMICH_HOST` |
| Jellyfin | `JELLYFIN_HOST` + `JELLYFIN_XML` |
| ComfyUI in Docker | `COMFY_CONTAINER` (and `COMFY_URL` for its stats) |

Anything left empty is reported as `not configured` in `/api/status`
(`configured`), the dashboard disables that consumer's buttons, and the switch
logic never claims an action it did not take — the last-action line lists only
what actually moved. `/api/mode/...` still works with a subset: `mode llm` on a
deployment with no Immich simply frees the card for llama.cpp.

Names that changed when the panel became generic keep working as aliases:
`PVE_HOST` for `GPU_HOST`, `FRONTEND_HOST` for `JELLYFIN_HOST`.

What is *not* generic, by design: the four consumers above. Each one is a small
block of code that knows how to stop it, start it or read its state, and there is
no plugin interface — a fifth consumer is a patch to `app.py`, not a config
file. The alternative (a generic "run this command" consumer) would have to
guess whether a process is busy before killing it, which is the one thing these
four probes do properly.

## State

`/state/auto.json`, in the `gpu_panel_state` volume, holds what has to survive a
recreate: the auto-switch settings, the last action and when it happened, the
revoked session nonces (sessions are stateless, so a logout has to remember one),
the Jellyfin device to restore and the card inventory (`gpus_known`, the names of
cards that have since disappeared, and the last change). Losing it costs a re-login and the timers
going back to their defaults; the card is where the truth is.

## Licence

MIT — see [`LICENSE`](LICENSE). A hosted derivative that must stay open is the
one thing this licence does not require; AGPL-3.0 would, and it is a single file
to replace.

## Troubleshooting

| Symptom | What it means |
| --- | --- |
| `502 nvidia-smi failed: … Permission denied (publickey)` | the key in `/root/.ssh` does not reach `GPU_HOST`; nothing else will work either |
| `502 … timed out after 30s` | a host is unreachable — the panel turns a timeout into a `502`, never a `500` |
| everything reads `not configured` | those variables are empty on purpose; see [Adapting it to your setup](#adapting-it-to-your-setup) |
| `1 of 2 · 1 card missing` | `GPU_EXPECTED` is above what `nvidia-smi` answers: the card is not initialising |
| `Power limit: not reported by the card` | the card exposes no settable range (typical on vGPU); the slider is disabled |
| `N card(s) refused … W` | the limit was set on the cards that accept it; the others are named with their own `nvidia-smi` error |
| signed out after changing a password or the token | the session key is derived from both, so rotating either revokes every cookie |
| the browser keeps asking to log in behind a proxy | the cookie is `Secure` only when the request says HTTPS; forward `X-Forwarded-Proto` |
| `403 cross-origin request rejected` | a write came from another origin with a session cookie; use the bearer token for scripts |

## Secrets

Three, and none of them is in this repository: the panel's password, its token
(`openssl rand -hex 32` each) and the SSH key that reaches the three hosts. In
the lab the first two live in the Ansible vault and reach the container through
the rendered `.env`; the key is `/root/.ssh` on the AI host, mounted read-only.

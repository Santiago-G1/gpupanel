"""GPU control panel: one card, several consumers, one owner at a time.

The cards are read and driven over SSH (nvidia-smi, the power limit), and every
consumer is a setting: llama.cpp over its HTTP router, Immich and Jellyfin and
ComfyUI through the host that runs them. Nothing here is tied to a particular
machine — only GPU_HOST and LLAMA_URL are required.

Auth is a signed session cookie plus a bearer token for automation.
"""

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import logging
import os
import secrets
import subprocess
import threading
import time
from urllib.parse import urlparse

import httpx
from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def required(*names):
    """The first of these variables that is set, or a message naming them all.

    Compose only reports that the container exited, so the traceback has to say
    which variable was empty. More than one name covers a variable that was
    renamed here: the old spelling keeps working.
    """
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(f"{' or '.join(names)} is required and must not be empty")


def optional(*names):
    """The first of these variables that is set, or "" when none of them is.

    An empty consumer is not an error: the panel reports it as not configured
    and leaves that part of the card alone.
    """
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def number(name, default, cast=int):
    raw = optional(name)
    if not raw:
        return cast(default)
    try:
        return cast(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a number, not {raw!r}")


def as_number(value, default=0.0):
    """A reading from nvidia-smi, or a default.

    Every field the tool queries can answer `N/A`: a card that does not report a
    power limit, a `utilization.gpu` with no driver support, a virtualised GPU.
    Those are not zero and must not become a traceback either.
    """
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


# --- the cards, and the model server on them ---
# Any SSH target where nvidia-smi answers: no Proxmox, no specific distro. The
# old name of this variable still works.
GPU_HOST = required("GPU_HOST", "PVE_HOST")
# The llama.cpp router. The panel only speaks HTTP to it, so it can be anywhere
# it can reach.
LLAMA_URL = required("LLAMA_URL")

# llama.cpp and ComfyUI: the machine they run on, and optionally the Proxmox CT
# to enter there with `pct exec`. No CT id means they run on that host itself.
AI_HOST = optional("AI_HOST") or GPU_HOST
AI_CTID = optional("AI_CTID")
LLAMA_LOG_UNIT = optional("LLAMA_LOG_UNIT") or "llama-server"
LLAMA_LOG_MAX = 2000

# The panel's own credentials. Both are required: an empty password would leave
# the panel unloggable, and an empty token would sign every session with a key
# derived from it.
PANEL_USER = optional("PANEL_USER") or "admin"
PANEL_PASSWORD = required("PANEL_PASSWORD")
PANEL_TOKEN = required("PANEL_TOKEN")
SESSION_COOKIE = optional("SESSION_COOKIE") or "panel_session"
SESSION_TTL = number("SESSION_TTL", 7 * 24 * 3600)
LOGIN_DELAY = number("LOGIN_DELAY", 0.4, float)
STATE_PATH = optional("STATE_PATH") or "/state/auto.json"

# --- the consumers, each one optional ---
# Immich. Its container names come from Immich itself and they have changed
# between releases, so they are settings rather than constants.
IMMICH_HOST = optional("IMMICH_HOST")
IMMICH_ML_CONTAINER = optional("IMMICH_ML_CONTAINER") or "immich_machine_learning"
IMMICH_PG_CONTAINER = optional("IMMICH_PG_CONTAINER") or "immich_postgres"
IMMICH_REDIS_CONTAINER = optional("IMMICH_REDIS_CONTAINER") or "immich_redis"
IMMICH_QUEUE_PREFIX = optional("IMMICH_QUEUE_PREFIX") or "immich_bull"
# Queues that mean somebody is waiting for the ML container.
IMMICH_ML_QUEUES = [q.strip() for q in (optional("IMMICH_ML_QUEUES") or "faceDetection,smartSearch,ocr").split(",") if q.strip()]

# Jellyfin needs the host and the file it rewrites, or it is not there.
JELLYFIN_HOST = optional("JELLYFIN_HOST", "FRONTEND_HOST")
JELLYFIN_XML = optional("JELLYFIN_XML")
JELLYFIN_CONTAINER = optional("JELLYFIN_CONTAINER") or "jellyfin"

# ComfyUI: a container to start and stop, its API for the queue depth, or both.
COMFY_CONTAINER = optional("COMFY_CONTAINER")
COMFY_URL = optional("COMFY_URL")
COMFY_UI_URL = optional("COMFY_UI_URL")

# What this deployment actually has. The dashboard hides what is missing and the
# switch logic never claims an action it did not take.
IMMICH = bool(IMMICH_HOST)
JELLYFIN = bool(JELLYFIN_HOST and JELLYFIN_XML)
COMFY = bool(COMFY_CONTAINER)
COMFY_API = bool(COMFY_URL)

# Where Jellyfin transcodes while llama.cpp owns the card: the other GPU
# (nvenc), the integrated one (vaapi) or the CPU (none).
JELLYFIN_FALLBACK = optional("JELLYFIN_FALLBACK") or "vaapi"
JELLYFIN_DEVICES = {"nvenc": "GPU (NVENC)", "vaapi": "integrated GPU (VAAPI)", "none": "CPU"}
if JELLYFIN_FALLBACK not in ("vaapi", "none"):
    JELLYFIN_FALLBACK = "vaapi"

# Models the dashboard may name (case-insensitive substrings). Anything else the
# router serves still loads and unloads, but is never printed. Empty lists all.
LLAMA_PUBLIC_MODELS = [m.strip().lower() for m in optional("LLAMA_PUBLIC_MODELS").split(",") if m.strip()]

# Cards the node is supposed to have. A card that does not initialise is not an
# error anywhere else: nvidia-smi simply lists one card less. 0 = the panel
# reports what it sees and calls nothing missing.
GPU_EXPECTED = number("GPU_EXPECTED", 0)

# Delivered base64: the psql quoting does not survive nested shells.
IMMICH_CFG_SCRIPT = r"""docker exec %s sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "select value from system_metadata where key = '"'"'system-config'"'"';"'"""

GPU_FIELDS = (
    "index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,"
    "power.draw,power.limit,power.default_limit,power.min_limit,power.max_limit,"
    "clocks.current.sm,clocks.current.memory,pcie.link.gen.current,pcie.link.width.current"
)
GPU_KEYS = [
    "index", "uuid", "name", "driver", "mem_used", "mem_total", "util", "temp",
    "power_draw", "power_limit", "power_default", "power_min", "power_max",
    "clk_sm", "clk_mem", "pcie_gen", "pcie_width",
]

# comfy_idle_min: with an empty queue and an idle card, the GPU goes back to
# llama.cpp by itself. 0 disables the timer.
#
# gpus_known is the baseline the inventory is compared against, gpus_names keeps
# the names of cards that have since disappeared.
DEFAULTS = {"enabled": True, "llama_idle_min": 60, "ml_idle_min": 0, "comfy_idle_min": 10, "last_action": None, "last_action_at": None, "revoked": {}, "jellyfin_before_llm": None, "gpus_known": None, "gpus_names": {}, "gpus_last_change": None}


log = logging.getLogger("gpu-panel")

# Derived from the credentials: rotating either one logs every browser out.
SESSION_KEY = hashlib.sha256(f"session|{PANEL_PASSWORD}|{PANEL_TOKEN}".encode()).digest()

state = dict(DEFAULTS)
STATE_LOCK = threading.Lock()
last_busy = time.time()
llm_blocked = False

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the saved settings and start the auto-switch tick."""
    load_state()
    task = asyncio.create_task(auto_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="GPU control panel", lifespan=lifespan)
# Shared by the dashboard and the guide.
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")


# --- state ---
def load_state():
    """Settings from the volume. A missing file is normal; a broken one is not."""
    try:
        with open(STATE_PATH) as fh:
            stored = json.load(fh)
    except FileNotFoundError:
        return
    except Exception as exc:  # noqa: BLE001 - a bad file must not stop the panel
        log.warning("ignoring %s: %s", STATE_PATH, exc)
        return
    with STATE_LOCK:
        state.update({k: v for k, v in stored.items() if k in DEFAULTS})


def save_state():
    """Written to a temporary file and renamed: a crash must not truncate it.

    The endpoints run in a thread pool, so two saves can overlap; the rename is
    atomic and the lock keeps them from writing the same temporary file.
    """
    parent = os.path.dirname(STATE_PATH) or "."
    with STATE_LOCK:
        try:
            blob = json.dumps(dict(state), indent=2)
        except TypeError as exc:  # a non-serialisable value would otherwise vanish
            log.warning("state is not serialisable: %s", exc)
            return
        tmp = f"{STATE_PATH}.tmp"
        try:
            os.makedirs(parent, exist_ok=True)
            with open(tmp, "w") as fh:
                fh.write(blob)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, STATE_PATH)
        except OSError as exc:
            log.warning("could not save %s: %s", STATE_PATH, exc)


# --- auth: login page -> signed session cookie; bearer token for automation ---
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def is_https(request: Request) -> bool:
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


def same_text(a: str, b: str) -> bool:
    """compare_digest for text. A non-ASCII password must be a 401, not a 500."""
    return secrets.compare_digest(a.encode("utf-8", "replace"), b.encode("utf-8", "replace"))


def make_session(user: str) -> str:
    payload = {"u": user, "exp": int(time.time()) + SESSION_TTL, "n": secrets.token_urlsafe(12)}
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64e(hmac.new(SESSION_KEY, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def session_payload(request: Request) -> dict:
    """Verified session payload, or {} when the cookie is absent/invalid/revoked."""
    raw = request.cookies.get(SESSION_COOKIE, "")
    body, _, sig = raw.partition(".")
    if not body or not sig:
        return {}
    expected = _b64e(hmac.new(SESSION_KEY, body.encode(), hashlib.sha256).digest())
    if not secrets.compare_digest(sig, expected):
        return {}
    try:
        payload = json.loads(_b64d(body))
    except Exception:
        return {}
    if int(payload.get("exp", 0)) < time.time():
        return {}
    if payload.get("n") in state.get("revoked", {}):
        return {}
    return payload


def read_session(request: Request) -> str:
    return str(session_payload(request).get("u") or "")


def has_bearer(request: Request) -> bool:
    header = request.headers.get("authorization", "")
    return bool(PANEL_TOKEN) and (
        same_text(header, f"Bearer {PANEL_TOKEN}")
        or same_text(request.headers.get("x-api-token", ""), PANEL_TOKEN)
    )


def require_auth(request: Request) -> str:
    if has_bearer(request):
        return "token"
    user = read_session(request)
    if user:
        # Cookie-authenticated writes must come from this same origin.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host", ""):
                raise HTTPException(403, "cross-origin request rejected")
        return user
    raise HTTPException(401, "unauthorized")


# --- helpers ---
def run(cmd, timeout=30, **kwargs):
    """A command, with its timeout and its missing binary turned into a result.

    Every caller checks `returncode` and reports its stderr, which is a far more
    useful answer than a 500 from an unreachable host.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"timed out after {timeout}s")
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def ssh(host, command, timeout=30):
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", host, command], timeout)


def ai_cmd(command):
    """A command where llama.cpp and ComfyUI live: on AI_HOST, or in its CT."""
    return f"pct exec {AI_CTID} -- {command}" if AI_CTID else command


def ssh_script(host, script, timeout=60):
    payload = base64.b64encode(script.encode()).decode()
    return run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", host, "base64 -d | sh"],
        timeout,
        input=payload,
    )


def gpu_status():
    """Every card, plus the single-card shape the power controls still read.

    `gpu` stays GPU 0 for backwards compatibility and `gpus` carries the list.
    Processes are attributed by UUID: without that the second card looks idle.
    """
    p = ssh(GPU_HOST, f"nvidia-smi --query-gpu={GPU_FIELDS} --format=csv,noheader,nounits")
    if p.returncode != 0:
        raise HTTPException(502, f"nvidia-smi failed: {p.stderr.strip() or p.stdout.strip()}")
    gpus = []
    for line in p.stdout.splitlines():
        parts = [v.strip() for v in line.split(",")]
        if len(parts) != len(GPU_KEYS):
            continue
        gpu = dict(zip(GPU_KEYS, parts))
        gpu["processes"] = []
        gpus.append(gpu)
    if not gpus:
        raise HTTPException(502, "nvidia-smi returned no GPUs")

    pr = ssh(GPU_HOST,
             "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
             "--format=csv,noheader,nounits")
    by_uuid = {g["uuid"]: g for g in gpus}
    for line in pr.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4 or not parts[1].isdigit():
            continue
        proc = {"pid": parts[1], "name": parts[2], "mem_mib": int(parts[3])}
        target = by_uuid.get(parts[0])
        if target is not None:
            proc["gpu"] = target["index"]
            target["processes"].append(proc)

    gpu = gpus[0]
    # A card is busy if any card is busy. `N/A` counts as idle here: nothing is
    # being computed, and it must not be read as a number.
    gpu["util_max"] = max(int(as_number(g["util"])) for g in gpus)
    gpu["power_limits"] = [g["power_limit"] for g in gpus]
    gpu["gpus"] = gpus
    return gpu


def llama_models():
    r = httpx.get(f"{LLAMA_URL}/v1/models", timeout=8)
    r.raise_for_status()
    return [
        {"id": m.get("id"), "status": (m.get("status") or {}).get("value", "unknown")}
        for m in r.json().get("data", [])
    ]


def llama_loaded():
    return [m for m in llama_models() if m["status"] == "loaded"]


def llama_unload_all():
    out = []
    for m in llama_loaded():
        r = httpx.post(f"{LLAMA_URL}/models/unload", json={"model": m["id"]}, timeout=60)
        out.append({"model": m["id"], "http": r.status_code})
    return out


def llama_logs(lines=200):
    """Last N journald lines of the llama.cpp unit, capped here and not in the browser."""
    try:
        lines = max(20, min(int(lines), LLAMA_LOG_MAX))
    except (TypeError, ValueError):
        raise HTTPException(400, "lines must be an integer")
    p = ssh(
        AI_HOST,
        ai_cmd(f"journalctl -u {LLAMA_LOG_UNIT} -n {lines} --no-pager -o short-iso"),
        timeout=25,
    )
    if p.returncode != 0:
        raise HTTPException(502, (p.stderr or p.stdout).strip() or "journalctl failed")
    return [line for line in p.stdout.splitlines() if line.strip()]


def listed_models(models):
    """Keep only the models the panel is allowed to name (see LLAMA_PUBLIC_MODELS)."""
    if not LLAMA_PUBLIC_MODELS:
        return list(models)
    return [m for m in models if any(p in str(m.get("id", "")).lower() for p in LLAMA_PUBLIC_MODELS)]


def model_names(models):
    """Names for messages: the listed ids, plus how many unlisted ones were touched."""
    listed = listed_models(models)
    names = [m.get("id") for m in listed]
    unlisted = len(models) - len(listed)
    if unlisted:
        names.append("%d unlisted" % unlisted)
    return names


def immich_ml_state():
    if not IMMICH:
        return "unconfigured"
    p = ssh(IMMICH_HOST, f"docker inspect -f '{{{{.State.Status}}}}' {IMMICH_ML_CONTAINER}")
    return p.stdout.strip() or "unknown"


def comfy_state():
    if not COMFY:
        return "unconfigured"
    p = ssh(AI_HOST, ai_cmd(f"docker inspect -f '{{{{.State.Status}}}}' {COMFY_CONTAINER}"))
    return p.stdout.strip() or "unknown"


def comfy(action):
    if action not in ("start", "stop", "restart"):
        raise HTTPException(400, "action must be start, stop or restart")
    if not COMFY:
        return {"action": action, "skipped": "ComfyUI is not configured"}
    p = ssh(AI_HOST, ai_cmd(f"docker {action} {COMFY_CONTAINER}"), timeout=90)
    if p.returncode != 0:
        raise HTTPException(502, p.stderr.strip() or p.stdout.strip())
    return {"action": action, "ok": True}


def comfy_queue_depth():
    """Prompts running or waiting: stopping the container would kill them."""
    if not COMFY_API:
        return 0
    try:
        queue = httpx.get(f"{COMFY_URL}/queue", timeout=6).json()
        return len(queue.get("queue_running") or []) + len(queue.get("queue_pending") or [])
    except Exception:  # noqa: BLE001 - starting container has no API yet
        return 0


def comfy_release(why=None):
    """Stop ComfyUI (only when idle) and hand the card back."""
    if comfy_queue_depth():
        return False
    with contextlib.suppress(HTTPException):
        comfy("stop")
        if why:
            record_action(why)
        return True
    return False


def comfy_stats():
    """State from ComfyUI itself: running, how much VRAM it sees free and how
    many prompts are queued. Silently empty when the container is down."""
    out = {"state": comfy_state()}
    if not COMFY_API or out["state"] != "running":
        return out
    try:
        stats = httpx.get(f"{COMFY_URL}/system_stats", timeout=8).json()
        device = (stats.get("devices") or [{}])[0]
        out["device"] = device.get("name")
        out["type"] = device.get("type")
        out["vram_free"] = round((device.get("vram_free") or 0) / 2**20)
        out["vram_total"] = round((device.get("vram_total") or 0) / 2**20)
        out["torch"] = (stats.get("system") or {}).get("pytorch_version")
        queue = httpx.get(f"{COMFY_URL}/queue", timeout=8).json()
        out["queue"] = len(queue.get("queue_running") or []) + len(queue.get("queue_pending") or [])
    except Exception as exc:  # noqa: BLE001 - a starting container has no API yet
        out["error"] = str(exc)
    return out


def immich_ml(action):
    if not IMMICH:
        return {"action": action, "skipped": "Immich is not configured"}
    p = ssh(IMMICH_HOST, f"docker {action} {IMMICH_ML_CONTAINER}")
    if p.returncode != 0:
        raise HTTPException(502, p.stderr.strip() or p.stdout.strip())
    return {"action": action, "ok": True}


def immich_ml_queue_active():
    """Jobs running or waiting on the queues that want the ML container."""
    if not IMMICH:
        return 0
    script = "\n".join(
        f'echo "$(docker exec {IMMICH_REDIS_CONTAINER} redis-cli llen {IMMICH_QUEUE_PREFIX}:{q}:active) '
        f'$(docker exec {IMMICH_REDIS_CONTAINER} redis-cli llen {IMMICH_QUEUE_PREFIX}:{q}:wait)"'
        for q in IMMICH_ML_QUEUES
    )
    p = ssh_script(IMMICH_HOST, script)
    total = 0
    for line in p.stdout.split():
        if line.isdigit():
            total += int(line)
    return total


def immich_transcode():
    """Immich's own ffmpeg settings, read from its Postgres system config."""
    if not IMMICH:
        return {}
    p = ssh_script(IMMICH_HOST, IMMICH_CFG_SCRIPT % IMMICH_PG_CONTAINER)
    try:
        return json.loads(p.stdout).get("ffmpeg", {})
    except Exception:
        return {"error": (p.stdout or p.stderr).strip()[:200]}


def jellyfin_accel():
    if not JELLYFIN:
        return "unconfigured"
    p = ssh(JELLYFIN_HOST, f"sed -n 's:.*<HardwareAccelerationType>\\([^<]*\\).*:\\1:p' {JELLYFIN_XML}")
    return p.stdout.strip() or "unknown"


def jellyfin_set_accel(accel):
    if accel not in ("nvenc", "vaapi", "none"):
        raise HTTPException(400, "accel must be nvenc, vaapi or none")
    if not JELLYFIN:
        return {"accel": accel, "skipped": "Jellyfin is not configured"}
    script = (
        "set -e\n"
        f"F={JELLYFIN_XML}\n"
        f"docker stop {JELLYFIN_CONTAINER} >/dev/null\n"
        "sed -i 's#<HardwareAccelerationType>[^<]*</HardwareAccelerationType>"
        f"#<HardwareAccelerationType>{accel}</HardwareAccelerationType>#' \"$F\"\n"
        f"docker start {JELLYFIN_CONTAINER} >/dev/null\n"
    )
    p = ssh_script(JELLYFIN_HOST, script)
    if p.returncode != 0:
        raise HTTPException(502, p.stderr.strip() or p.stdout.strip())
    return {"accel": accel}


def jellyfin_ensure_accel(accel):
    if not JELLYFIN:
        return {"accel": accel, "skipped": "Jellyfin is not configured"}
    if jellyfin_accel() == accel:
        return {"accel": accel, "already": True}
    return jellyfin_set_accel(accel)


def jellyfin_release_gpu():
    """Move Jellyfin off the card, remembering the device it had before."""
    current = jellyfin_accel()
    if current in ("nvenc", "vaapi"):
        state["jellyfin_before_llm"] = current
    return jellyfin_ensure_accel(JELLYFIN_FALLBACK)


def jellyfin_reclaim_gpu():
    """Undo jellyfin_release_gpu() once llama.cpp gives the card back."""
    accel = state.get("jellyfin_before_llm") or "nvenc"
    state["jellyfin_before_llm"] = None
    return jellyfin_ensure_accel(accel)


def auto_snapshot():
    return {
        "enabled": state["enabled"],
        "llama_idle_min": state["llama_idle_min"],
        "ml_idle_min": state["ml_idle_min"],
        "comfy_idle_min": state["comfy_idle_min"],
        "last_action": state.get("last_action"),
        "last_action_at": state.get("last_action_at"),
        "idle_seconds": int(time.time() - last_busy),
        "llm_blocked": llm_blocked,
    }


def record_action(text):
    state["last_action"] = text
    state["last_action_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state()


def _short(uuid):
    """The part of a PCI UUID that tells two identical cards apart.

    A PCI UUID is `GPU-<serial>-<more>`, but a MIG device is `MIG-<parent>-...`
    and the serial is not always the same length: cut at the dash instead of at
    a fixed offset, which used to leave `bbbb-222` in a message.
    """
    text = str(uuid)
    parts = text.split("-")
    serial = parts[1] if len(parts) > 1 else text
    return serial[:12] or text


def track_gpus(gpus):
    """Remember the card set, and record the moment it changes.

    A card that fails to initialise is silent: no driver error reaches the
    panel, `nvidia-smi` just answers with one card fewer, and every consumer on
    the remaining card still works. Comparing UUIDs with the previous look is
    what turns that silence into an event with a timestamp.

    The first observation after a fresh state file is the baseline, not news.
    """
    uuids = [g["uuid"] for g in gpus]
    names = {g["uuid"]: g.get("name", "GPU") for g in gpus}
    known = state.get("gpus_known")
    if known is None:
        state["gpus_known"] = uuids
        state["gpus_names"] = names
        save_state()
        return
    if sorted(uuids) == sorted(known):
        # Same cards, maybe reported in another order: not a change.
        return
    seen = dict(state.get("gpus_names") or {})
    added = [{"uuid": u, "name": names[u]} for u in uuids if u not in known]
    removed = [{"uuid": u, "name": seen.get(u, "GPU")} for u in known if u not in uuids]
    state["gpus_known"] = uuids
    state["gpus_names"] = {**{u: seen[u] for u in known if u in seen}, **names}
    state["gpus_last_change"] = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "added": added,
        "removed": removed,
        "count": len(uuids),
    }
    parts = []
    if added:
        parts.append("%d new GPU: %s" % (len(added), ", ".join("%s (%s)" % (g["name"], _short(g["uuid"])) for g in added)))
    if removed:
        parts.append("%d GPU lost: %s" % (len(removed), ", ".join("%s (%s)" % (g["name"], _short(g["uuid"])) for g in removed)))
    record_action("cards changed (now %d): %s" % (len(uuids), "; ".join(parts)))


def inventory_view(gpus):
    """What the dashboard needs to say whether the card set is complete."""
    return {
        "count": len(gpus),
        "expected": GPU_EXPECTED,
        "missing": max(0, GPU_EXPECTED - len(gpus)) if GPU_EXPECTED else 0,
        "present": [
            {"uuid": g["uuid"], "index": g.get("index"), "name": g.get("name")}
            for g in gpus
        ],
        "last_change": state.get("gpus_last_change"),
    }


# --- auto switch ---
def auto_tick():
    global last_busy, llm_blocked
    gpu = gpu_status()
    # Before the switch logic and whatever it decides: the inventory has to be
    # kept even when the auto mode is off.
    track_gpus(gpu["gpus"])
    util = gpu["util_max"]
    if util > 0:
        last_busy = time.time()
    if not state["enabled"]:
        return
    idle = time.time() - last_busy

    try:
        models = llama_models()
    except Exception:
        models = []
    # "loading" counts as in use: the router asks for the VRAM before the status
    # reads "loaded", and a 13.6 GB model does not fit next to ComfyUI.
    in_use = [m for m in models if str(m.get("status", "")).lower().startswith("load")]

    comfy_up = False
    with contextlib.suppress(Exception):
        comfy_up = comfy_state() == "running"

    if in_use:
        # llama.cpp has priority over every other consumer of the card.
        done = []
        if comfy_up and comfy_release():
            comfy_up = False
            done.append("ComfyUI stopped")
        if not llm_blocked:
            # Immich ML is CUDA and cannot share; Jellyfin keeps transcoding on
            # whatever device the fallback names.
            with contextlib.suppress(HTTPException):
                if IMMICH:
                    immich_ml("stop")
                    done.append("Immich ML stopped")
                if JELLYFIN:
                    jellyfin_release_gpu()
                    done.append("Jellyfin on %s" % JELLYFIN_DEVICES[JELLYFIN_FALLBACK])
            llm_blocked = True
            if done:
                record_action("LLM in use: %s (card free for llama.cpp)" % ", ".join(done))
        if state["llama_idle_min"] and idle > state["llama_idle_min"] * 60:
            unloaded = llama_unload_all()
            if unloaded and not comfy_up:
                back = []
                with contextlib.suppress(HTTPException):
                    if IMMICH:
                        immich_ml("start")
                        back.append("Immich ML on")
                    if JELLYFIN:
                        accel = jellyfin_reclaim_gpu().get("accel")
                        back.append("Jellyfin back to %s" % JELLYFIN_DEVICES.get(accel, accel))
                llm_blocked = False
                record_action("GPU idle %d min: unloaded %s%s"
                              % (idle // 60,
                                 ", ".join(model_names([{"id": u["model"]} for u in unloaded])),
                                 ("; " + ", ".join(back)) if back else ""))
    else:
        llm_blocked = False
        # An idle ComfyUI hands the card back on its own.
        if comfy_up and state["comfy_idle_min"] and idle > state["comfy_idle_min"] * 60:
            if comfy_release("ComfyUI idle %d min: stopped, card free again" % (idle // 60)):
                comfy_up = False

    if IMMICH and state["ml_idle_min"] and idle > state["ml_idle_min"] * 60:
        if immich_ml_state() == "running" and immich_ml_queue_active() == 0:
            with contextlib.suppress(HTTPException):
                immich_ml("stop")
            record_action("Immich ML idle %d min: stopped to free VRAM" % (idle // 60))


async def auto_loop():
    while True:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(auto_tick)
        await asyncio.sleep(20)


# --- pages ---
NO_STORE = {"Cache-Control": "no-store"}


@app.get("/login")
def login_page():
    return FileResponse(os.path.join(APP_DIR, "static", "login.html"), headers=NO_STORE)


@app.get("/")
def index(request: Request):
    if not (read_session(request) or has_bearer(request)):
        return RedirectResponse("/login", status_code=303)
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"), headers=NO_STORE)


@app.get("/howto")
def howto(request: Request):
    """The guide: what each mode does and how to use the four consumers."""
    if not (read_session(request) or has_bearer(request)):
        return RedirectResponse("/login?next=/howto", status_code=303)
    return FileResponse(os.path.join(APP_DIR, "static", "howto.html"), headers=NO_STORE)


@app.post("/api/login")
def login(request: Request, body: dict = Body(...)):
    user = str(body.get("user") or "").strip()
    password = str(body.get("password") or "")
    ok = bool(PANEL_PASSWORD) and same_text(user, PANEL_USER) and same_text(password, PANEL_PASSWORD)
    if not ok:
        time.sleep(LOGIN_DELAY)
        raise HTTPException(401, "wrong user or password")
    response = JSONResponse({"ok": True, "user": user})
    response.set_cookie(
        SESSION_COOKIE,
        make_session(user),
        max_age=SESSION_TTL,
        httponly=True,
        samesite="lax",
        secure=is_https(request),
        path="/",
    )
    return response


@app.post("/api/logout")
def logout(request: Request):
    payload = session_payload(request)
    if payload.get("n"):
        # Sessions are stateless, so logout has to remember the nonce.
        revoked = {n: exp for n, exp in state.get("revoked", {}).items() if int(exp) > time.time()}
        revoked[str(payload["n"])] = int(payload.get("exp") or time.time())
        state["revoked"] = revoked
        save_state()
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get("/api/session")
def session_info(request: Request):
    if has_bearer(request):
        return {"authenticated": True, "user": "token", "mode": "token"}
    user = read_session(request)
    return {"authenticated": bool(user), "user": user or None, "mode": "session" if user else None}


# --- api ---
@app.get("/api/status")
def status(_: str = Depends(require_auth)):
    gpu = gpu_status()
    out = {"gpu": gpu, "gpus": gpu.pop("gpus"), "auto": auto_snapshot()}
    # A page load is also an observation: the set is compared here, not only on
    # the 20-second tick, so a card that just came back shows up at once.
    track_gpus(out["gpus"])
    out["inventory"] = inventory_view(out["gpus"])
    # What this deployment has: the dashboard says "not configured" instead of
    # showing a consumer's controls as broken.
    out["configured"] = {
        "immich": IMMICH,
        "jellyfin": JELLYFIN,
        "comfyui": COMFY,
        "comfyui_api": COMFY_API,
        "llama_logs": bool(AI_HOST),
    }
    for key, fn in (
        ("immich_ml", immich_ml_state),
        ("immich_ffmpeg", immich_transcode),
        ("jellyfin_accel", jellyfin_accel),
    ):
        try:
            out[key] = fn()
        except Exception as exc:  # noqa: BLE001 - surface each probe failure
            out[key] = {"error": str(exc)}
    try:
        models = llama_models()
        out["llama"] = listed_models(models)
        # Counts cover every model, not only the listed ones.
        out["llama_counts"] = {
            "total": len(models),
            "loaded": sum(1 for m in models if m["status"] == "loaded"),
        }
    except Exception as exc:  # noqa: BLE001 - surface the probe failure
        out["llama"] = {"error": str(exc)}
        out["llama_counts"] = {"error": str(exc)}
    try:
        out["comfyui"] = comfy_stats()
        out["comfyui"]["url"] = COMFY_UI_URL
    except Exception as exc:  # noqa: BLE001 - surface the probe failure
        out["comfyui"] = {"state": "unknown", "error": str(exc)}
    return out


@app.post("/api/mode/{mode}")
def set_mode(mode: str, _: str = Depends(require_auth)):
    global llm_blocked
    result = {}
    done = []
    # ComfyUI would keep holding the card: stop it before leaving Image mode.
    if mode in ("llm", "media", "balanced") and COMFY:
        with contextlib.suppress(Exception):
            if comfy_state() == "running":
                result["comfyui"] = comfy("stop")
                done.append("ComfyUI stopped")

    if mode == "llm":
        if IMMICH:
            result["immich_ml"] = immich_ml("stop")
            done.append("Immich ML stopped")
        if JELLYFIN:
            result["jellyfin"] = jellyfin_release_gpu()
            done.append("Jellyfin on %s" % JELLYFIN_DEVICES[JELLYFIN_FALLBACK])
        llm_blocked = True
        record_action("mode LLM: %s (card free for llama.cpp)" % (", ".join(done) or "nothing to move"))
    elif mode == "media":
        result["llama"] = llama_unload_all()
        done.append("llama.cpp unloaded")
        if IMMICH:
            result["immich_ml"] = immich_ml("start")
            done.append("Immich ML on")
        if JELLYFIN:
            result["jellyfin"] = jellyfin_ensure_accel("nvenc")
            done.append("Jellyfin on %s" % JELLYFIN_DEVICES["nvenc"])
        state["jellyfin_before_llm"] = None
        llm_blocked = False
        record_action("mode Media: %s" % ", ".join(done))
    elif mode == "balanced":
        if IMMICH:
            result["immich_ml"] = immich_ml("start")
            done.append("Immich ML on")
        if JELLYFIN:
            result["jellyfin"] = jellyfin_ensure_accel("nvenc")
            done.append("Jellyfin on %s" % JELLYFIN_DEVICES["nvenc"])
        state["jellyfin_before_llm"] = None
        llm_blocked = False
        record_action("mode Shared: %s (llama.cpp not unloaded)" % (", ".join(done) or "nothing to move"))
    elif mode == "image":
        # Image generation needs the whole card: unload llama.cpp, stop the CUDA
        # Immich workload and move Jellyfin off (it keeps transcoding).
        result["llama"] = llama_unload_all()
        done.append("llama.cpp unloaded")
        if IMMICH:
            result["immich_ml"] = immich_ml("stop")
            done.append("Immich ML stopped")
        if JELLYFIN:
            result["jellyfin"] = jellyfin_release_gpu()
            done.append("Jellyfin on %s" % JELLYFIN_DEVICES[JELLYFIN_FALLBACK])
        state["jellyfin_before_llm"] = None
        if COMFY:
            result["comfyui"] = comfy("start")
            done.append("ComfyUI started with the card to itself")
        llm_blocked = False
        record_action("mode Image: %s" % ", ".join(done))
    else:
        raise HTTPException(400, "mode must be llm, media, balanced or image")
    return result


@app.post("/api/auto")
def set_auto(body: dict = Body(...), _: str = Depends(require_auth)):
    if "enabled" in body:
        state["enabled"] = bool(body["enabled"])
    for key in ("llama_idle_min", "ml_idle_min", "comfy_idle_min"):
        if key in body:
            value = int(body[key])
            if value < 0 or value > 1440:
                raise HTTPException(400, f"{key} must be 0..1440 minutes")
            state[key] = value
    save_state()
    return auto_snapshot()


@app.post("/api/power")
def set_power(body: dict = Body(...), _: str = Depends(require_auth)):
    try:
        watts = int(body.get("watts"))
    except (TypeError, ValueError):
        raise HTTPException(400, "watts must be an integer")
    # The range is the card's, not a constant: a P100 allows 125-250 W and
    # anything else reports its own limits. A card that reports neither (or
    # answers `N/A`) is left to nvidia-smi, whose refusal is the honest answer.
    card = gpu_status()
    low, high = as_number(card["power_min"], None), as_number(card["power_max"], None)
    if low is not None and high is not None and not low <= watts <= high:
        raise HTTPException(400, f"watts must be between {int(low)} and {int(high)}")
    p = ssh(GPU_HOST, f"nvidia-smi -pl {watts}")
    if p.returncode != 0:
        raise HTTPException(502, p.stderr.strip() or p.stdout.strip())
    record_action(f"power limit = {watts} W")
    return {"ok": True, "watts": watts}


@app.post("/api/llama/unload")
def llama_unload(_: str = Depends(require_auth)):
    unloaded = llama_unload_all()
    return {"unloaded": model_names([{"id": u["model"]} for u in unloaded])}


@app.get("/api/llama/logs")
def llama_logs_api(lines: int = 200, _: str = Depends(require_auth)):
    """Tail of the llama.cpp service log, for debugging a load or a crash."""
    out = llama_logs(lines)
    return {
        "unit": LLAMA_LOG_UNIT,
        "host": AI_HOST,
        "ct": AI_CTID or None,
        "count": len(out),
        "lines": out,
    }


@app.post("/api/comfy/{action}")
def comfy_action(action: str, _: str = Depends(require_auth)):
    return comfy(action)


@app.post("/api/immich/ml/{action}")
def immich_ml_action(action: str, _: str = Depends(require_auth)):
    if action not in ("start", "stop", "restart"):
        raise HTTPException(400, "action must be start, stop or restart")
    return immich_ml(action)


@app.post("/api/jellyfin/accel/{accel}")
def jellyfin_accel_action(accel: str, _: str = Depends(require_auth)):
    return jellyfin_set_accel(accel)

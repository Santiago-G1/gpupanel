"""GPU control panel for the lab.

Drives the Tesla P100 through the PVE node: that is where nvidia-smi and the
power limit live, and where the SSH reach to the card's other consumers starts.
Auth is a signed session cookie plus a bearer token for automation.
"""

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import time
from collections import deque
from urllib.parse import urlparse

import httpx
from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def required(name):
    """A variable the panel cannot work without, named when it is missing.

    Compose only reports that the container exited, so the traceback has to say
    which variable was empty.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required and must not be empty")
    return value


# Required from the environment: this file never carries the lab's addresses.
PVE_HOST = required("PVE_HOST")
IMMICH_HOST = required("IMMICH_HOST")
FRONTEND_HOST = required("FRONTEND_HOST")
LLAMA_URL = required("LLAMA_URL")
# llama-server is a systemd unit, so its logs are read over SSH to the PVE node.
AI_CTID = required("AI_CTID")
# Jellyfin's encoding.xml on the frontend host, read and rewritten over SSH.
JELLYFIN_XML = required("JELLYFIN_XML")
LLAMA_LOG_UNIT = os.environ.get("LLAMA_LOG_UNIT", "llama-server")
LLAMA_LOG_MAX = 2000
# On-demand stack: started here, stopped when the card goes back to llama.cpp.
COMFY_URL = os.environ.get("COMFY_URL", "")
COMFY_UI_URL = os.environ.get("COMFY_UI_URL", "")
COMFY_CONTAINER = os.environ.get("COMFY_CONTAINER", "comfyui")
# Both are required: an empty password would leave the panel unloggable, and an
# empty token would sign every session with the same predictable key.
PANEL_USER = os.environ.get("PANEL_USER", "admin")
PANEL_PASSWORD = required("PANEL_PASSWORD")
PANEL_TOKEN = required("PANEL_TOKEN")
SESSION_COOKIE = os.environ.get("SESSION_COOKIE", "panel_session")
SESSION_TTL = int(os.environ.get("SESSION_TTL", 7 * 24 * 3600))
LOGIN_DELAY = float(os.environ.get("LOGIN_DELAY", 0.4))
STATE_PATH = os.environ.get("STATE_PATH", "/state/auto.json")

# Jellyfin moves to the iGPU (VAAPI) when llama.cpp owns the P100: dropping to
# CPU would lose hardware transcoding for nothing. `none` restores CPU.
JELLYFIN_FALLBACK = os.environ.get("JELLYFIN_FALLBACK", "vaapi")
JELLYFIN_DEVICES = {"nvenc": "P100 (NVENC)", "vaapi": "iGPU (VAAPI)", "none": "CPU"}
if JELLYFIN_FALLBACK not in ("vaapi", "none"):
    JELLYFIN_FALLBACK = "vaapi"

# Models the dashboard may name (case-insensitive substrings). Anything else the
# router serves still loads and unloads, but is never printed. Empty lists all.
LLAMA_PUBLIC_MODELS = [m.strip().lower() for m in os.environ.get("LLAMA_PUBLIC_MODELS", "").split(",") if m.strip()]

IMMICH_ML = "immich_machine_learning"
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
ML_QUEUES = ("faceDetection", "smartSearch", "ocr")

# Delivered base64: the psql quoting does not survive nested shells.
IMMICH_CFG_SCRIPT = r"""docker exec immich_postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "select value from system_metadata where key = '"'"'system-config'"'"';"'"""

# comfy_idle_min: with an empty queue and an idle card, the P100 goes back to
# llama.cpp by itself. 0 disables the timer.
DEFAULTS = {"enabled": True, "llama_idle_min": 60, "ml_idle_min": 0, "comfy_idle_min": 10, "last_action": None, "last_action_at": None, "revoked": {}, "jellyfin_before_llm": None}

# Derived from the credentials: rotating either one logs every browser out.
SESSION_KEY = hashlib.sha256(f"session|{PANEL_PASSWORD}|{PANEL_TOKEN}".encode()).digest()

state = dict(DEFAULTS)
busy_samples = deque(maxlen=240)
last_busy = time.time()
llm_blocked = False

app = FastAPI(title="GPU control panel")
# Shared by the dashboard and the guide.
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")


# --- state ---
def load_state():
    try:
        with open(STATE_PATH) as fh:
            state.update({k: v for k, v in json.load(fh).items() if k in DEFAULTS})
    except Exception:
        pass


def save_state():
    with contextlib.suppress(Exception):
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as fh:
            json.dump(state, fh, indent=2)


# --- auth: login page -> signed session cookie; bearer token for automation ---
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def is_https(request: Request) -> bool:
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


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
        secrets.compare_digest(header, f"Bearer {PANEL_TOKEN}")
        or secrets.compare_digest(request.headers.get("x-api-token", ""), PANEL_TOKEN)
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
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)


def ssh(host, command, timeout=30):
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", host, command], timeout)


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
    p = ssh(PVE_HOST, f"nvidia-smi --query-gpu={GPU_FIELDS} --format=csv,noheader,nounits")
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

    pr = ssh(PVE_HOST,
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
    # A card is busy if any card is busy.
    gpu["util_max"] = max(int(g["util"]) for g in gpus)
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
        PVE_HOST,
        f"pct exec {AI_CTID} -- journalctl -u {LLAMA_LOG_UNIT} -n {lines} --no-pager -o short-iso",
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
    p = ssh(IMMICH_HOST, f"docker inspect -f '{{{{.State.Status}}}}' {IMMICH_ML}")
    return p.stdout.strip() or "unknown"


def comfy_state():
    """ComfyUI runs in this same CT: SSH to the PVE node, then enter it."""
    p = ssh(PVE_HOST, f"pct exec {AI_CTID} -- docker inspect -f '{{{{.State.Status}}}}' {COMFY_CONTAINER}")
    return p.stdout.strip() or "unknown"


def comfy(action):
    if action not in ("start", "stop", "restart"):
        raise HTTPException(400, "action must be start, stop or restart")
    p = ssh(PVE_HOST, f"pct exec {AI_CTID} -- docker {action} {COMFY_CONTAINER}", timeout=90)
    if p.returncode != 0:
        raise HTTPException(502, p.stderr.strip() or p.stdout.strip())
    return {"action": action, "ok": True}


def comfy_queue_depth():
    """Prompts running or waiting: stopping the container would kill them."""
    if not COMFY_URL:
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
    if not COMFY_URL or out["state"] != "running":
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
    p = ssh(IMMICH_HOST, f"docker {action} {IMMICH_ML}")
    if p.returncode != 0:
        raise HTTPException(502, p.stderr.strip() or p.stdout.strip())
    return {"action": action, "ok": True}


def immich_ml_queue_active():
    script = "\n".join(
        f'echo "$(docker exec immich_redis redis-cli llen immich_bull:{q}:active) $(docker exec immich_redis redis-cli llen immich_bull:{q}:wait)"'
        for q in ML_QUEUES
    )
    p = ssh_script(IMMICH_HOST, script)
    total = 0
    for line in p.stdout.split():
        if line.isdigit():
            total += int(line)
    return total


def immich_transcode():
    p = ssh_script(IMMICH_HOST, IMMICH_CFG_SCRIPT)
    try:
        return json.loads(p.stdout).get("ffmpeg", {})
    except Exception:
        return {"error": (p.stdout or p.stderr).strip()[:200]}


def jellyfin_accel():
    p = ssh(FRONTEND_HOST, f"sed -n 's:.*<HardwareAccelerationType>\\([^<]*\\).*:\\1:p' {JELLYFIN_XML}")
    return p.stdout.strip() or "unknown"


def jellyfin_set_accel(accel):
    if accel not in ("nvenc", "vaapi", "none"):
        raise HTTPException(400, "accel must be nvenc, vaapi or none")
    script = (
        "set -e\n"
        f"F={JELLYFIN_XML}\n"
        "docker stop jellyfin >/dev/null\n"
        "sed -i 's#<HardwareAccelerationType>[^<]*</HardwareAccelerationType>"
        f"#<HardwareAccelerationType>{accel}</HardwareAccelerationType>#' \"$F\"\n"
        "docker start jellyfin >/dev/null\n"
    )
    p = ssh_script(FRONTEND_HOST, script)
    if p.returncode != 0:
        raise HTTPException(502, p.stderr.strip() or p.stdout.strip())
    return {"accel": accel}


def jellyfin_ensure_accel(accel):
    if jellyfin_accel() == accel:
        return {"accel": accel, "already": True}
    return jellyfin_set_accel(accel)


def jellyfin_release_p100():
    """Move Jellyfin to the iGPU, remembering the device it had before."""
    current = jellyfin_accel()
    if current in ("nvenc", "vaapi"):
        state["jellyfin_before_llm"] = current
    return jellyfin_ensure_accel(JELLYFIN_FALLBACK)


def jellyfin_reclaim_p100():
    """Undo jellyfin_release_p100() once llama.cpp gives the card back."""
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


# --- auto switch ---
def auto_tick():
    global last_busy, llm_blocked
    util = int(gpu_status()["util_max"])
    busy_samples.append(util)
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
        # llama.cpp has priority over every other consumer of the P100.
        comfy_stopped = False
        if comfy_up:
            comfy_stopped = comfy_release(
                None if not llm_blocked
                else "LLM in use: ComfyUI stopped (P100 free for llama.cpp)")
            comfy_up = comfy_up and not comfy_stopped
        if not llm_blocked:
            with contextlib.suppress(HTTPException):
                immich_ml("stop")        # Immich ML is CUDA: it needs the P100
                jellyfin_release_p100()  # Jellyfin keeps transcoding on the iGPU
            llm_blocked = True
            record_action("LLM in use: %sImmich ML stopped, Jellyfin on %s (P100 free for llama.cpp)"
                          % ("ComfyUI stopped, " if comfy_stopped else "",
                             JELLYFIN_DEVICES[JELLYFIN_FALLBACK]))
        if state["llama_idle_min"] and idle > state["llama_idle_min"] * 60:
            unloaded = llama_unload_all()
            if unloaded and not comfy_up:
                back = {"accel": state.get("jellyfin_before_llm") or "nvenc"}
                with contextlib.suppress(HTTPException):
                    immich_ml("start")
                    back = jellyfin_reclaim_p100()
                llm_blocked = False
                record_action("GPU idle %d min: unloaded %s, Immich ML on, Jellyfin back to %s"
                              % (idle // 60, ", ".join(model_names([{"id": u["model"]} for u in unloaded])),
                                 JELLYFIN_DEVICES.get(back.get("accel"), back.get("accel"))))
    else:
        llm_blocked = False
        # An idle ComfyUI hands the card back on its own.
        if comfy_up and state["comfy_idle_min"] and idle > state["comfy_idle_min"] * 60:
            if comfy_release("ComfyUI idle %d min: stopped, P100 free again" % (idle // 60)):
                comfy_up = False

    if state["ml_idle_min"] and idle > state["ml_idle_min"] * 60:
        if immich_ml_state() == "running" and immich_ml_queue_active() == 0:
            with contextlib.suppress(HTTPException):
                immich_ml("stop")
            record_action("Immich ML idle %d min: stopped to free VRAM" % (idle // 60))


async def auto_loop():
    while True:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(auto_tick)
        await asyncio.sleep(20)


@app.on_event("startup")
async def _startup():
    load_state()
    asyncio.create_task(auto_loop())


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
    ok = bool(PANEL_PASSWORD) and secrets.compare_digest(user, PANEL_USER) and secrets.compare_digest(password, PANEL_PASSWORD)
    if not ok:
        time.sleep(LOGIN_DELAY)
        raise HTTPException(401, "usuario o contraseña incorrectos")
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
    # ComfyUI would keep holding the card: stop it before leaving Image mode.
    if mode in ("llm", "media", "balanced"):
        with contextlib.suppress(Exception):
            if comfy_state() == "running":
                result["comfyui"] = comfy("stop")

    if mode == "llm":
        result["immich_ml"] = immich_ml("stop")        # CUDA: it needs the P100
        result["jellyfin"] = jellyfin_release_p100()   # iGPU keeps transcoding
        llm_blocked = True
        record_action("mode LLM: Immich ML stopped, Jellyfin on %s (P100 free for llama.cpp)"
                      % JELLYFIN_DEVICES[JELLYFIN_FALLBACK])
    elif mode == "media":
        result["llama"] = llama_unload_all()
        result["immich_ml"] = immich_ml("start")
        result["jellyfin"] = jellyfin_ensure_accel("nvenc")
        state["jellyfin_before_llm"] = None
        llm_blocked = False
        record_action("mode Media: llama.cpp unloaded, Immich ML + Jellyfin (NVENC on the P100) on")
    elif mode == "balanced":
        result["immich_ml"] = immich_ml("start")
        result["jellyfin"] = jellyfin_ensure_accel("nvenc")
        state["jellyfin_before_llm"] = None
        llm_blocked = False
        record_action("mode Shared: Immich ML + Jellyfin (NVENC on the P100, LLM not unloaded) on")
    elif mode == "image":
        # Image generation needs the whole card: unload llama.cpp, stop the CUDA
        # Immich workload and move Jellyfin to the iGPU (it keeps transcoding).
        result["llama"] = llama_unload_all()
        result["immich_ml"] = immich_ml("stop")
        result["jellyfin"] = jellyfin_release_p100()
        state["jellyfin_before_llm"] = None
        result["comfyui"] = comfy("start")
        llm_blocked = False
        record_action("mode Image: llama.cpp unloaded, Immich ML stopped, "
                      "Jellyfin on %s; ComfyUI started with the P100 to itself"
                      % JELLYFIN_DEVICES[JELLYFIN_FALLBACK])
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
    if not 125 <= watts <= 250:
        raise HTTPException(400, "watts must be between 125 and 250")
    p = ssh(PVE_HOST, f"nvidia-smi -pl {watts}")
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
    return {"unit": LLAMA_LOG_UNIT, "container": AI_CTID, "count": len(out), "lines": out}


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

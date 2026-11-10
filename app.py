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
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
        raise RuntimeError(f"{name} must be a number, not {raw!r}") from None


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
# it can reach. A router started with `--api-key` (or behind a proxy that wants
# one) needs `LLAMA_API_KEY`; nothing is sent when it is empty.
# `.rstrip("/")`: a URL pasted from a browser ends with one, and `//v1/models` is
# not what every router expects.
LLAMA_URL = required("LLAMA_URL").rstrip("/")
LLAMA_API_KEY = optional("LLAMA_API_KEY")
LLAMA_HEADERS = {"Authorization": f"Bearer {LLAMA_API_KEY}"} if LLAMA_API_KEY else {}

# llama.cpp and ComfyUI: the machine they run on, and optionally the Proxmox CT
# to enter there with `pct exec`. No CT id means they run on that host itself.
AI_HOST = optional("AI_HOST") or GPU_HOST
AI_CTID = optional("AI_CTID")
LLAMA_LOG_UNIT = optional("LLAMA_LOG_UNIT") or "llama-server"
LLAMA_LOG_MAX = 2000
# The folder the models live in, as llama.cpp sees it: inside the CT when there
# is an AI_CTID, on AI_HOST otherwise. Empty means the panel never reads a disk.
LLAMA_MODELS_DIR = optional("LLAMA_MODELS_DIR")

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
# The agent's web UI is an s6 service inside its container, next to the gateway:
# the panel moves the service, not the container, so channels and models stay up.
HERMES_WEBUI_CONTAINER = optional("HERMES_WEBUI_CONTAINER")
HERMES_WEBUI_SERVICE = optional("HERMES_WEBUI_SERVICE") or "dashboard"
HERMES_WEBUI_UI_URL = optional("HERMES_WEBUI_UI_URL")
COMFY_CONTAINER = optional("COMFY_CONTAINER")
COMFY_URL = optional("COMFY_URL")
COMFY_UI_URL = optional("COMFY_UI_URL")

# What this deployment actually has. The dashboard hides what is missing and the
# switch logic never claims an action it did not take.
IMMICH = bool(IMMICH_HOST)
JELLYFIN = bool(JELLYFIN_HOST and JELLYFIN_XML)
WEBUI = bool(HERMES_WEBUI_CONTAINER)
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
# The exact set, when a count is not enough: a PCI address (0000:02:00.0) or a
# UUID per card. Empty falls back to the count above.
GPU_EXPECTED_IDS = [x.strip() for x in optional("GPU_EXPECTED_IDS").split(",") if x.strip()]
# Kernel evidence (NVRM, Xid, AER, amdgpu resets) is collected when the card set
# changes. It is the difference between "a card is gone" and "the card fell off
# the bus at 03:12", and it costs one SSH only when something happened.
GPU_EVIDENCE = optional("GPU_EVIDENCE").strip().lower() not in ("0", "false", "no", "off")
# How many card events the state file keeps. The newest is the one the
# dashboard shows; the rest is what makes a flapping card visible.
GPU_EVENTS_MAX = 50

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

# PCI vendor ids, which is the only thing every vendor agrees on. The cards are
# read through the driver that owns them: nvidia-smi for NVIDIA, sysfs for AMD
# and Intel. Nothing here requires a card to be an NVIDIA one.
VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}
VENDOR_LABEL = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel"}

# One SSH, one report. The sysfs half is vendor-neutral (a card that is not on
# the bus is not in /sys/class/drm either) and nvidia-smi is what says whether
# the *driver* sees the cards it can see. When those two disagree the panel can
# finally tell a dead card from a crashed driver.
PROBE_SCRIPT = r"""
# One line, always: a multi-line value would break the record it lives in.
p() { head -1 "$1" 2>/dev/null || true; }
# The AER counters are a table; only the total is a reading.
aer() { awk -v key="$2" '$1 == key { print $2; exit }' "$1" 2>/dev/null; }
have() { command -v "$1" >/dev/null 2>&1 && echo 1 || echo 0; }
kv() { printf '\t%s=%s' "$1" "$2"; }

echo "@@tools"
echo "tool$(kv nvidia-smi "$(have nvidia-smi)")$(kv amd-smi "$(have amd-smi)")$(kv rocm-smi "$(have rocm-smi)")$(kv lspci "$(have lspci)")"

echo "@@sysfs"
for dev in /sys/class/drm/card[0-9]*/device; do
  [ -e "$dev/vendor" ] || continue
  bdf=$(basename "$(readlink -f "$dev" 2>/dev/null)" 2>/dev/null)
  drm=$(basename "$(dirname "$dev")")
  drv=$(basename "$(readlink -f "$dev/driver" 2>/dev/null)" 2>/dev/null)
  nm=$(p "$dev/product_name")
  if [ -z "$nm" ] && command -v lspci >/dev/null 2>&1; then
    nm=$(lspci -s "$bdf" 2>/dev/null | head -1 | sed -e 's/^[^ ]* //' -e 's/^[^:]*: //')
  fi
  hw=$(ls -d "$dev"/hwmon/hwmon* 2>/dev/null | head -1)
  echo "card$(kv bdf "$bdf")$(kv drm "$drm")$(kv vendor "$(p "$dev/vendor")")$(kv driver "$drv")$(kv name "$nm")$(kv vram_used "$(p "$dev/mem_info_vram_used")")$(kv vram_total "$(p "$dev/mem_info_vram_total")")$(kv busy "$(p "$dev/gpu_busy_percent")")$(kv temp "$(p "$hw/temp1_input")")$(kv power "$(p "$hw/power1_average")")$(kv cap "$(p "$hw/power1_cap")")$(kv cap_min "$(p "$hw/power1_cap_min")")$(kv cap_max "$(p "$hw/power1_cap_max")")$(kv clk "$(p "$hw/freq1_input")")$(kv clk_mem "$(p "$hw/freq2_input")")$(kv link_speed "$(p "$dev/current_link_speed")")$(kv link_width "$(p "$dev/current_link_width")")$(kv unique_id "$(p "$dev/unique_id")")$(kv aer_fatal "$(aer "$dev/aer_dev_fatal" TOTAL_ERR_FATAL)")$(kv aer_nonfatal "$(aer "$dev/aer_dev_nonfatal" TOTAL_ERR_NONFATAL)")$(kv aer_correctable "$(aer "$dev/aer_dev_correctable" TOTAL_ERR_COR)")"
done

echo "@@nvinfo"
for f in /proc/driver/nvidia/gpus/*/information; do
  [ -e "$f" ] || continue
  echo "nv$(kv uuid "$(sed -n 's/^GPU UUID:[[:space:]]*//p' "$f" | head -1)")$(kv bdf "$(sed -n 's/^Bus Location:[[:space:]]*//p' "$f" | head -1)")"
done

# A wedged card can make nvidia-smi hang; the sysfs half has already been
# printed by then, and the timeout keeps this probe from hanging with it.
NV=""
command -v timeout >/dev/null 2>&1 && NV="timeout 10"

echo "@@gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  out=$($NV nvidia-smi --query-gpu=@FIELDS@ --format=csv,noheader,nounits 2>&1); rc=$?
  printf '%s\n' "$out"
else
  rc=127
fi
echo "@@gpu_rc"
printf '%s\n' "$rc"

echo "@@procs"
if command -v nvidia-smi >/dev/null 2>&1; then
  out=$($NV nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1); rc=$?
  printf '%s\n' "$out"
else
  rc=127
fi
echo "@@procs_rc"
printf '%s\n' "$rc"

echo "@@end"
"""

# comfy_idle_min: with an empty queue and an idle card, the GPU goes back to
# llama.cpp by itself. 0 disables the timer.
#
# gpus_known is the baseline the inventory is compared against, gpus_names keeps
# what is known about every card ever seen (name, PCI address, UUID) including
# the ones that have since disappeared, gpus_events is a short history and
# gpus_down is the cards that are missing right now with the moment they went.
# last_busy is persisted so a container restart does not reset the idle timer.
DEFAULTS = {
    "enabled": True,
    "llama_idle_min": 60,
    "ml_idle_min": 0,
    "comfy_idle_min": 10,
    "last_action": None,
    "last_action_at": None,
    "revoked": {},
    "jellyfin_before_llm": None,
    "gpus_known": None,
    "gpus_names": {},
    "gpus_last_change": None,
    "gpus_events": [],
    "gpus_down": {},
    "gpus_blind": [],
    "gpus_aer": {},
    "last_busy": None,
}


log = logging.getLogger("gpu-panel")

# Derived from the credentials: rotating either one logs every browser out.
SESSION_KEY = hashlib.sha256(f"session|{PANEL_PASSWORD}|{PANEL_TOKEN}".encode()).digest()

state = dict(DEFAULTS)
# Reentrant: record_action() takes it and then saves, which takes it again.
STATE_LOCK = threading.RLock()
# The consumer probes are independent SSH/HTTP round trips; run them at once.
PROBE_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="probe")
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
    global last_busy
    with STATE_LOCK:
        state.update({k: v for k, v in stored.items() if k in DEFAULTS})
    if state.get("last_busy"):
        # A restart is not an idle period: keep counting from before it.
        last_busy = float(state["last_busy"])


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


def gather_probes(jobs):
    """Run independent probes at once and keep every answer, including failures.

    They are separate SSH and HTTP round trips with nothing in common, and the
    dashboard asks for all of them every five seconds: one after another was
    2.3 s per page load, most of it waiting.
    """
    futures = {name: PROBE_POOL.submit(job) for name, job in jobs.items()}
    out = {}
    for name, future in futures.items():
        try:
            out[name] = future.result(timeout=90)
        except Exception as exc:  # noqa: BLE001 - a failed probe is an answer
            out[name] = exc
    return out


def llama_request(method, path, **kwargs):
    """A call to the router, with the key if one is configured."""
    kwargs.setdefault("timeout", 8)
    return httpx.request(method, f"{LLAMA_URL}{path}", headers=dict(LLAMA_HEADERS), **kwargs)


# One TCP session, reused: every probe is 6 SSH calls and a fresh handshake is
# ~150 ms each. `%C` is a hash of host/user/port, so the path stays short.
SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=6",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=/tmp/gpu-panel-%C",
    "-o", "ControlPersist=60",
]


def ssh(host, command, timeout=30):
    return run(["ssh", *SSH_OPTS, host, command], timeout)


def ai_cmd(command):
    """A command where llama.cpp and ComfyUI live: on AI_HOST, or in its CT."""
    return f"pct exec {AI_CTID} -- {command}" if AI_CTID else command


def ssh_script(host, script, timeout=60):
    payload = base64.b64encode(script.encode()).decode()
    return run(["ssh", *SSH_OPTS, host, "base64 -d | sh"], timeout, input=payload)


def _tidy(value, scale=1, digits=None):
    """A sysfs number as the string nvidia-smi would have printed, or `N/A`.

    The dashboard was written against nvidia-smi's strings and keeps working on
    them: the AMD half of the probe produces the same ones, in the same units.
    """
    try:
        raw = float(str(value).strip())
    except (TypeError, ValueError):
        return "N/A"
    scaled = raw / scale
    return str(int(round(scaled))) if digits is None else f"{scaled:.{digits}f}"


def _pcie_gen(speed):
    """`16.0 GT/s PCIe` -> `4`. AMD reports the rate, not the generation."""
    for rate, gen in (("2.5", "1"), ("5.0", "2"), ("8.0", "3"), ("16.0", "4"), ("32.0", "5"), ("64.0", "6")):
        if str(speed or "").strip().startswith(rate):
            return gen
    return "N/A"


def _fields(parts):
    """The `k=v` fields of one probe line, without trusting any of them."""
    out = {}
    for item in parts:
        key, sep, value = item.partition("=")
        if sep and key:
            out[key] = value
    return out


def _aer(raw):
    """PCIe error counters for one card, when the kernel exposes them."""
    counters = {name: _tidy(raw.get(f"aer_{name}")) for name in ("fatal", "nonfatal", "correctable")}
    if all(value == "N/A" for value in counters.values()):
        return None
    return counters


def parse_probe(text):
    """The probe's sections. Anything unrecognised is dropped, not trusted."""
    sections = {"tools": {}, "sysfs": [], "nvinfo": [], "gpu": [], "procs": [], "rc": {}}
    current = ""
    for line in text.splitlines():
        if line.startswith("@@"):
            current = line[2:].strip()
            continue
        if current == "tools" and line.startswith("tool\t"):
            sections["tools"].update({k: v == "1" for k, v in _fields(line.split("\t")[1:]).items()})
        elif current == "sysfs" and line.startswith("card\t"):
            sections["sysfs"].append(_fields(line.split("\t")[1:]))
        elif current == "nvinfo" and line.startswith("nv\t"):
            sections["nvinfo"].append(_fields(line.split("\t")[1:]))
        elif current == "gpu":
            sections["gpu"].append(line)
        elif current == "procs":
            sections["procs"].append(line)
        elif current.endswith("_rc") and line.strip():
            sections["rc"][current[: -len("_rc")]] = line.strip()
    return sections


def sysfs_card(raw, vendor, tools):
    """A card read from sysfs: no vendor tool, no ROCm, no driver-specific code.

    AMD exposes everything the dashboard shows through /sys/class/drm, which is
    in the kernel and not in ROCm. Intel exposes presence and its driver; the
    panel reports it because Jellyfin's VAAPI fallback depends on that GPU, and
    not drawing it was the panel lying about the machine.
    """
    cap_min, cap_max = raw.get("cap_min"), raw.get("cap_max")
    setter = None
    if vendor == "amd" and _tidy(cap_min) != "N/A" and _tidy(cap_max) != "N/A":
        setter = "amd-smi" if tools.get("amd-smi") else ("rocm-smi" if tools.get("rocm-smi") else None)
    return {
        "index": str(raw.get("drm") or "").removeprefix("card") or None,
        # The UUID is what nvidia-smi calls identity; amdgpu has `unique_id` and
        # the PCI address is the one every vendor has.
        "uuid": raw.get("unique_id") or raw.get("bdf") or None,
        "pci": raw.get("bdf") or "",
        "vendor": vendor,
        "name": raw.get("name") or VENDOR_LABEL.get(vendor, "GPU"),
        "driver": raw.get("driver") or "",
        "mem_used": _tidy(raw.get("vram_used"), 2**20),
        "mem_total": _tidy(raw.get("vram_total"), 2**20),
        "util": _tidy(raw.get("busy")),
        "temp": _tidy(raw.get("temp"), 1000),
        "power_draw": _tidy(raw.get("power"), 10**6),
        "power_limit": _tidy(raw.get("cap"), 10**6, 2),
        "power_default": "N/A",
        "power_min": _tidy(raw.get("cap_min"), 10**6),
        "power_max": _tidy(raw.get("cap_max"), 10**6),
        "clk_sm": _tidy(raw.get("clk"), 10**6),
        "clk_mem": _tidy(raw.get("clk_mem"), 10**6),
        "pcie_gen": _pcie_gen(raw.get("link_speed")),
        "pcie_width": _tidy(raw.get("link_width")),
        "processes": [],
        "bus_visible": True,
        "driver_visible": bool(raw.get("driver")),
        "power_settable": bool(setter),
        "setter": setter,
        "aer": _aer(raw),
    }


def build_cards(sections):
    """One card list from the two views the probe returns, and the errors.

    nvidia-smi is the driver's view and sysfs is the bus's. A card in both is a
    healthy card; a card only on the bus is one the driver lost (`Xid 79`, a
    crashed module); a card in neither is off the bus. The list keeps the two
    flags so the dashboard can say which of the three happened.
    """
    tools = sections["tools"]
    errors = []
    bus = {raw["bdf"]: raw for raw in sections["sysfs"] if raw.get("bdf")}
    # /proc/driver/nvidia is the only place that ties nvidia-smi's UUID to the
    # PCI address the bus reports, and both directions are needed: one to attach
    # the address to the driver's row, the other to keep the UUID of a card the
    # driver stopped listing.
    uuid_by_bdf = {r["bdf"]: r["uuid"] for r in sections["nvinfo"] if r.get("bdf") and r.get("uuid")}
    bdf_by_uuid = {r["uuid"]: r["bdf"] for r in sections["nvinfo"] if r.get("bdf") and r.get("uuid")}
    cards = []
    claimed = set()

    for line in sections["gpu"]:
        parts = [value.strip() for value in line.split(",")]
        if len(parts) != len(GPU_KEYS):
            if line.strip():
                # `Unable to determine the device handle for GPU 0000:01:00.0:
                # GPU is lost.` lands here, and it is evidence, not noise.
                errors.append(line.strip()[:300])
            continue
        card = dict(zip(GPU_KEYS, parts, strict=True))
        card["pci"] = bdf_by_uuid.get(card.get("uuid")) or ""
        card["vendor"] = "nvidia"
        card["bus_visible"] = (card["pci"] in bus) if card["pci"] else None
        card["driver_visible"] = True
        card["setter"] = "nvidia-smi" if tools.get("nvidia-smi", True) else None
        card["power_settable"] = (
            as_number(card.get("power_min"), None) is not None
            and as_number(card.get("power_max"), None) is not None
        )
        card["aer"] = _aer(bus.get(card["pci"]) or {})
        card["processes"] = []
        if card["pci"]:
            claimed.add(card["pci"])
        cards.append(card)

    by_uuid = {card.get("uuid"): card for card in cards}
    for line in sections["procs"]:
        parts = [value.strip() for value in line.split(",")]
        if len(parts) != 4 or not parts[1].isdigit():
            if line.strip():
                errors.append(line.strip()[:200])
            continue
        proc = {"pid": parts[1], "name": parts[2], "mem_mib": int(parts[3]) if parts[3].isdigit() else 0}
        target = by_uuid.get(parts[0])
        if target is not None:
            proc["gpu"] = target["index"]
            target["processes"].append(proc)

    for bdf, raw in bus.items():
        if bdf in claimed:
            continue
        vendor = VENDORS.get(str(raw.get("vendor") or "").lower(), "unknown")
        card = sysfs_card(raw, vendor, tools)
        if uuid_by_bdf.get(bdf):
            card["uuid"] = uuid_by_bdf[bdf]
        if vendor == "nvidia":
            # On the bus, invisible to nvidia-smi: this is a card the driver
            # gave up on, not a card that is missing from the machine.
            card["driver_visible"] = False
        cards.append(card)

    order = {"nvidia": 0, "amd": 1, "intel": 2}
    cards.sort(key=lambda card: (order.get(card.get("vendor"), 3), as_number(card.get("index"), 99)))
    return cards, errors


def power_command(card, watts):
    """The vendor command that sets a power limit, or None when there is none.

    NVIDIA answers to `nvidia-smi -i <index> -pl`. AMD answers to `amd-smi set`
    when that tool is installed, and to the older `rocm-smi --setpoweroverdrive`
    (one machine-wide value, not one per card) when it is not. Intel exposes no
    power cap at all, and that is not a failure: it is a card the dashboard must
    not offer a slider for.
    """
    setter = card.get("setter")
    if setter is None:
        setter = "nvidia-smi" if card.get("vendor") in (None, "nvidia") else ""
    index = card.get("index")
    if setter == "nvidia-smi" and index is not None:
        return f"nvidia-smi -i {index} -pl {watts}"
    if setter == "amd-smi" and index is not None:
        return f"amd-smi set -g {index} --power-cap {watts}"
    if setter == "rocm-smi":
        return f"rocm-smi --setpoweroverdrive {watts}"
    return None


def probe_cards():
    """Every card, or why there are none. This one does not raise.

    A host that answers with zero cards is not a broken host: it is a machine
    whose driver is unloaded or whose cards are gone, and that is the event the
    inventory exists to record. Only a host that cannot be read at all is an
    error, and it is raised by `gpu_status`.
    """
    p = ssh_script(GPU_HOST, PROBE_SCRIPT.replace("@FIELDS@", GPU_FIELDS), timeout=30)
    text = p.stdout or ""
    if "@@end" not in text:
        detail = (p.stderr or p.stdout or "").strip().splitlines()
        return {
            "host_ok": False,
            "cards": [],
            "errors": [detail[-1][:300]] if detail else [f"the probe returned nothing (exit {p.returncode})"],
            "tools": {},
            "rc": {},
        }
    sections = parse_probe(text)
    cards, errors = build_cards(sections)
    return {
        "host_ok": True,
        "cards": cards,
        "errors": errors,
        "tools": sections["tools"],
        "rc": sections["rc"],
    }


def probe_view(probe):
    """The part of a probe that belongs in /api/status: what failed and why."""
    return {
        "host_ok": probe.get("host_ok"),
        "tools": probe.get("tools") or {},
        "errors": list(probe.get("errors") or [])[:5],
        "nvidia_smi": (probe.get("rc") or {}).get("gpu"),
    }


def single_card(cards, probe=None):
    """The first card, with every card attached: what the controls read."""
    if not cards:
        empty = dict.fromkeys(GPU_KEYS, "N/A")
        empty.update({
            "index": None, "uuid": None, "name": "no cards", "driver": "", "vendor": None,
            "pci": "", "processes": [], "util_max": 0, "power_limits": [], "gpus": [],
            "power_settable": False, "bus_visible": None, "driver_visible": None,
        })
        if probe is not None:
            empty["probe"] = probe_view(probe)
        return empty
    gpu = dict(cards[0])
    # A card is busy if any card is busy. `N/A` counts as idle here: nothing is
    # being computed, and it must not be read as a number.
    gpu["util_max"] = max(int(as_number(card.get("util"))) for card in cards)
    gpu["power_limits"] = [card.get("power_limit") for card in cards]
    gpu["gpus"] = cards
    if probe is not None:
        gpu["probe"] = probe_view(probe)
    return gpu


def gpu_status():
    """Every card, plus the single-card shape the power controls still read.

    Processes are attributed by UUID: without that the second card looks idle.
    Zero cards is not a 502: the inventory turns it into an event with a time.
    """
    probe = probe_cards()
    if not probe["host_ok"]:
        errors = probe.get("errors") or ["no answer"]
        raise HTTPException(502, f"card probe failed: {errors[0]}")
    return single_card(probe["cards"], probe)


def llama_models():
    r = llama_request("GET", "/v1/models")
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
        r = llama_request("POST", "/models/unload", json={"model": m["id"]}, timeout=60)
        out.append({"model": m["id"], "http": r.status_code})
    return out


def llama_logs(lines=200):
    """Last N journald lines of the llama.cpp unit, capped here and not in the browser."""
    try:
        lines = max(20, min(int(lines), LLAMA_LOG_MAX))
    except (TypeError, ValueError):
        raise HTTPException(400, "lines must be an integer") from None
    p = ssh(
        AI_HOST,
        ai_cmd(f"journalctl -u {LLAMA_LOG_UNIT} -n {lines} --no-pager -o short-iso"),
        timeout=25,
    )
    if p.returncode != 0:
        raise HTTPException(502, (p.stderr or p.stdout).strip() or "journalctl failed")
    return [line for line in p.stdout.splitlines() if line.strip()]


def models_on_disk():
    """The .gguf files in LLAMA_MODELS_DIR, as llama.cpp sees them.

    `-L` is what makes this useful: several models here are symlinks into the
    pool, and without it `%s` is the size of the link, not of the file.
    """
    command = (
        f"find -L {shlex.quote(LLAMA_MODELS_DIR)} -maxdepth 1 -type f -name '*.gguf' "
        "-printf '%f\\t%s\\t%T@\\n'"
    )
    # Eight seconds: the dashboard polls every five, and PROBE_POOL has six
    # workers for six probes, so a probe slower than the poll never gives its
    # worker back. A hung SSH is not worth the whole card.
    p = ssh(AI_HOST, ai_cmd(command), timeout=8)
    rows = [line for line in p.stdout.splitlines() if line.strip()]
    models = []
    for line in rows:
        parts = line.split("\t")
        if len(parts) != 3:
            log.warning("models folder: skipping unreadable line %r", line)
            continue
        name, size, mtime = parts
        try:
            model = {"id": name.removesuffix(".gguf"), "size": int(size), "mtime": int(float(mtime))}
        except ValueError:
            log.warning("models folder: skipping unreadable line %r", line)
            continue
        # A projector is not a model: the router does not publish it as one.
        if not name.lower().startswith("mmproj"):
            models.append(model)
    # A folder with one unreadable subdirectory still lists the rest: `find`
    # says so on stderr and exits non-zero, but what it printed is a listing.
    if p.returncode != 0:
        if not rows:
            raise RuntimeError((p.stderr or p.stdout).strip() or "find failed")
        log.debug("find -L %s exited %s: %s", LLAMA_MODELS_DIR, p.returncode,
                  (p.stderr or p.stdout).strip())
    return sorted(models, key=lambda m: m["id"])


def public_model(model):
    """Whether the panel may name this id at all (see LLAMA_PUBLIC_MODELS)."""
    if not LLAMA_PUBLIC_MODELS:
        return True
    name = str(model.get("id", "")).lower()
    return any(p in name for p in LLAMA_PUBLIC_MODELS)


def listed_models(models):
    """Keep only the models the panel is allowed to name (see LLAMA_PUBLIC_MODELS)."""
    return [m for m in models if public_model(m)]


def model_key(name):
    """A model's name on both sides: no directory, no `.gguf`, no case.

    The router answers with whatever its presets say (`x`, `x.gguf`, a full
    path); the folder answers with `%f`, already bare.
    """
    base = str(name or "").rsplit("/", 1)[-1]
    if base.lower().endswith(".gguf"):
        base = base[:-len(".gguf")]
    return base.lower()


def disk_models_view(models, served=()):
    """The folder says what there is, the router what it serves: mark both.

    A router that does not answer costs the marks, never the listing.
    """
    states = {model_key(m.get("id")): str(m.get("status") or "") for m in served or ()}
    view = []
    for model in listed_models(models):
        key = model_key(model.get("id"))
        view.append({**model, "served": key in states, "loaded": states.get(key, "").startswith("load")})
    return view


def model_names(models):
    """Names for messages: the listed ids, plus how many unlisted ones were touched."""
    listed = listed_models(models)
    names = [m.get("id") for m in listed]
    unlisted = len(models) - len(listed)
    if unlisted:
        names.append(f"{unlisted} unlisted")
    return names


def immich_ml_state():
    if not IMMICH:
        return "unconfigured"
    p = ssh(IMMICH_HOST, f"docker inspect -f '{{{{.State.Status}}}}' {IMMICH_ML_CONTAINER}")
    return p.stdout.strip() or "unknown"


# A stopped ComfyUI has no API to answer with, so its state comes from the
# container: `pct exec` plus `docker inspect` is over a second, and a container
# does not change state between two dashboard polls. Actions force it.
COMFY_STATE_TTL = 15
COMFY_STATE = {"at": 0.0, "value": None}


def comfy_state(force=False):
    """Whether ComfyUI is up. HTTP first: it answers in five ms.

    The container is the fallback, because a ComfyUI that is still booting has
    no API yet but is not stopped either. That fallback is remembered for
    COMFY_STATE_TTL seconds — and a start/stop/restart asks again.
    """
    if not COMFY:
        return "unconfigured"
    if COMFY_API:
        try:
            httpx.get(f"{COMFY_URL}/system_stats", timeout=3)
            COMFY_STATE.update({"at": time.time(), "value": "running"})
            return "running"
        except Exception:  # noqa: BLE001 - not answering: ask the container
            pass
    now = time.time()
    if not force and COMFY_STATE["value"] and now - COMFY_STATE["at"] < COMFY_STATE_TTL:
        return COMFY_STATE["value"]
    p = ssh(AI_HOST, ai_cmd(f"docker inspect -f '{{{{.State.Status}}}}' {COMFY_CONTAINER}"))
    value = p.stdout.strip() or "unknown"
    COMFY_STATE.update({"at": now, "value": value})
    return value


WEBUI_STATE_TTL = 5
WEBUI_STATE = {"at": 0.0, "value": None}


def hermes_webui_state(force=False):
    """`up`/`down` from s6's own view of the slot, `unconfigured` without a container.

    `s6-svstat -o up` prints exactly `true`/`false`: the human form of the same
    command says `down (exitcode 0)` or `down (signal SIGTERM)` depending on how
    the slot died, and parsing that is how a probe starts lying.
    """
    if not WEBUI:
        return "unconfigured"
    now = time.time()
    if not force and WEBUI_STATE["value"] and now - WEBUI_STATE["at"] < WEBUI_STATE_TTL:
        return WEBUI_STATE["value"]
    p = ssh(AI_HOST, ai_cmd(
        f"docker exec {HERMES_WEBUI_CONTAINER} /command/s6-svstat -o up "
        f"/run/service/{HERMES_WEBUI_SERVICE}"))
    raw = p.stdout.strip()
    if p.returncode != 0 or raw not in ("true", "false"):
        raise RuntimeError((p.stderr or p.stdout).strip() or "s6-svstat failed")
    value = "up" if raw == "true" else "down"
    WEBUI_STATE.update({"at": now, "value": value})
    return value


def hermes_webui_stats():
    """The slot's state plus the address of the dashboard, like `comfyui`."""
    return {"state": hermes_webui_state(), "url": HERMES_WEBUI_UI_URL}


def hermes_webui(action):
    if action not in ("start", "stop", "restart"):
        raise HTTPException(400, "action must be start, stop or restart")
    if not WEBUI:
        return {"action": action, "skipped": "the agent's web UI is not configured"}
    # restart is stop-then-start: the service has no restart signal of its own.
    for flag in (["-d"] if action == "stop" else ["-u"] if action == "start" else ["-d", "-u"]):
        p = ssh(AI_HOST, ai_cmd(
            f"docker exec {HERMES_WEBUI_CONTAINER} /command/s6-svc {flag} "
            f"/run/service/{HERMES_WEBUI_SERVICE}"), timeout=90)
        if p.returncode != 0:
            raise HTTPException(502, (p.stderr or p.stdout).strip() or "s6-svc failed")
    return {"action": action, "ok": True, "state": hermes_webui_state(force=True)}


def comfy(action):
    if action not in ("start", "stop", "restart"):
        raise HTTPException(400, "action must be start, stop or restart")
    if not COMFY:
        return {"action": action, "skipped": "ComfyUI is not configured"}
    p = ssh(AI_HOST, ai_cmd(f"docker {action} {COMFY_CONTAINER}"), timeout=90)
    if p.returncode != 0:
        raise HTTPException(502, p.stderr.strip() or p.stdout.strip())
    # The remembered state is now wrong on purpose: ask again.
    comfy_state(force=True)
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


def card_key(card):
    """Which card this is, in a way the vendor does not change.

    NVIDIA has UUIDs and they are the best identity there is, but they come
    from the driver: a card the driver stopped seeing loses its UUID too, and
    keying on it would read as a removal plus an arrival. The PCI address is on
    the bus and is what the absence of a card looks like, so it is the fallback
    that makes those two cases the same card. AMD and Intel are keyed on it
    directly.
    """
    uuid = str(card.get("uuid") or "").strip()
    if card.get("vendor") in (None, "nvidia") and uuid.startswith(("GPU-", "MIG-")):
        return uuid
    pci = str(card.get("pci") or "").strip()
    if pci and pci != "N/A":
        return pci
    if uuid and uuid != "N/A":
        return uuid
    return "?"


def _short(key):
    """The part of an identity that tells two identical cards apart.

    A PCI UUID is `GPU-<serial>-<more>`, but a MIG device is `MIG-<parent>-...`
    and the serial is not always the same length: cut at the dash instead of at
    a fixed offset, which used to leave `bbbb-222` in a message. A PCI address
    is already short and is printed whole.
    """
    text = str(key)
    if text.startswith(("GPU-", "MIG-")):
        parts = text.split("-")
        return (parts[1] if len(parts) > 1 else text)[:12] or text
    return text[:12]


def _meta(key, value):
    """What is known about a card: dicts now, plain names in older state files."""
    if isinstance(value, dict):
        uuid = value.get("uuid") or (key if str(key).startswith(("GPU-", "MIG-")) else "")
        return {"name": value.get("name") or "GPU", "pci": value.get("pci") or "",
                "uuid": uuid, "vendor": value.get("vendor") or ""}
    uuid = key if str(key).startswith(("GPU-", "MIG-")) else ""
    return {"name": value or "GPU", "pci": "", "uuid": uuid, "vendor": ""}


def gpu_evidence():
    """Kernel lines that explain a card that just changed.

    `Xid 79` is a card that fell off the bus, an `amdgpu` ring reset is a driver
    that gave up, and `pcieport` AER lines are the link: the panel cannot fix
    any of it, but it cannot call them the same thing either. One SSH, and only
    when something actually changed.
    """
    if not GPU_EVIDENCE:
        return []
    p = ssh(
        GPU_HOST,
        "journalctl -k --since '-10 min' -o short-iso --no-pager 2>/dev/null "
        "| grep -Ei 'NVRM|Xid|AER|pcieport|amdgpu.*(error|reset|ring)' | tail -20",
        timeout=20,
    )
    return [line.strip() for line in (p.stdout or "").splitlines() if line.strip()][-20:]


def track_gpus(gpus):
    """Remember the card set, and record the moment it changes.

    Two card views matter and they fail differently: a card that is not on the
    bus is gone from /sys/class/drm as well (hardware, power, riser), while a
    card the driver stopped seeing is still there with its PCI address (`Xid
    79`, a crashed nvidia module). The event says which one happened, when, and
    with the kernel lines that go with it.

    The first observation after a fresh state file is the baseline, not news.
    """
    keys = [card_key(card) for card in gpus]
    meta = {
        card_key(card): {
            "name": card.get("name") or "GPU",
            "pci": card.get("pci") or "",
            "uuid": str(card.get("uuid") or ""),
            "vendor": card.get("vendor") or "",
        }
        for card in gpus
    }
    aers = {card_key(card): card.get("aer") for card in gpus if card.get("aer")}
    if aers:
        state["gpus_aer"] = {**(state.get("gpus_aer") or {}), **aers}

    known = state.get("gpus_known")
    if known is None:
        with STATE_LOCK:
            state["gpus_known"] = keys
            state["gpus_names"] = meta
            state["gpus_down"] = {}
            state["gpus_blind"] = sorted(k for k, v in zip(keys, gpus, strict=True) if v.get("driver_visible") is False)
        save_state()
        return

    at = time.strftime("%Y-%m-%d %H:%M:%S")
    now = time.time()
    seen = {key: _meta(key, value) for key, value in (state.get("gpus_names") or {}).items()}

    added = [card for card in gpus if card_key(card) not in known]
    removed = [key for key in known if key not in keys]
    blind_now = sorted(card_key(card) for card in gpus if card.get("driver_visible") is False)
    blind_before = sorted(state.get("gpus_blind") or [])
    went_blind = [key for key in blind_now if key not in blind_before]
    came_back = [key for key in blind_before if key not in blind_now and key in keys]
    with STATE_LOCK:
        state["gpus_blind"] = blind_now
    if not (added or removed or went_blind or came_back):
        # Same cards, maybe reported in another order: not a change.
        return

    evidence = gpu_evidence()
    events = []
    down = dict(state.get("gpus_down") or {})
    for key in removed:
        info = seen.get(key) or _meta(key, "GPU")
        previous = down.get(key) or {}
        down[key] = {
            "key": key,
            "uuid": info.get("uuid") or "",
            "pci": info.get("pci") or "",
            "name": info.get("name") or "GPU",
            "vendor": info.get("vendor") or "",
            # `since` survives the event history, so a card that stays gone
            # says for how long instead of only when it was noticed.
            "since": previous.get("since") or at,
            "since_ts": previous.get("since_ts") or now,
            "lost": int(previous.get("lost") or 0) + 1,
            "aer": previous.get("aer") or (state.get("gpus_aer") or {}).get(key),
        }
        events.append({**{k: down[key][k] for k in ("key", "uuid", "pci", "name", "vendor")},
                       "at": at, "ts": now, "kind": "removed", "count": len(keys), "evidence": evidence})
    for card in added:
        key = card_key(card)
        down.pop(key, None)
        events.append({
            "at": at, "ts": now, "kind": "added", "key": key, "uuid": str(card.get("uuid") or ""),
            "pci": card.get("pci") or "", "name": card.get("name") or "GPU",
            "vendor": card.get("vendor") or "", "count": len(keys),
            "evidence": evidence if removed else None,
        })
    for key in went_blind:
        info = meta.get(key) or seen.get(key) or _meta(key, "GPU")
        events.append({"at": at, "ts": now, "kind": "blind", "key": key, "uuid": info.get("uuid") or "",
                       "pci": info.get("pci") or "", "name": info.get("name") or "GPU",
                       "vendor": info.get("vendor") or "", "count": len(keys), "evidence": evidence})
    for key in came_back:
        info = meta.get(key) or seen.get(key) or _meta(key, "GPU")
        down.pop(key, None)
        events.append({"at": at, "ts": now, "kind": "visible", "key": key, "uuid": info.get("uuid") or "",
                       "pci": info.get("pci") or "", "name": info.get("name") or "GPU",
                       "vendor": info.get("vendor") or "", "count": len(keys)})

    def listed(kind):
        return [{"key": e["key"], "uuid": e["uuid"], "name": e["name"], "pci": e["pci"]}
                for e in events if e["kind"] == kind]

    with STATE_LOCK:
        state["gpus_known"] = keys
        # A card that is gone keeps its name, which is what makes the next
        # "lost" message readable.
        state["gpus_names"] = {**{key: value for key, value in seen.items() if key in keys or key in down}, **meta}
        state["gpus_down"] = down
        state["gpus_events"] = (list(state.get("gpus_events") or []) + events)[-GPU_EVENTS_MAX:]
        state["gpus_last_change"] = {
            "at": at,
            "count": len(keys),
            "added": listed("added"),
            "removed": listed("removed"),
        }

    def label(cards):
        return ", ".join(f"{card['name']} ({_short(card['key'])})" for card in cards)

    parts = []
    if added:
        parts.append(f"{len(added)} new GPU: {label(listed('added'))}")
    if removed:
        parts.append(f"{len(removed)} GPU lost: {label(listed('removed'))}")
    if went_blind:
        parts.append(f"{len(went_blind)} GPU on the bus but not in the driver: {label(listed('blind'))}")
    record_action(f"cards changed (now {len(keys)}): {'; '.join(parts)}")


def inventory_view(gpus, probe=None):
    """What the dashboard needs to say whether the card set is complete.

    `missing` is the count the settings promise; `absent` is the same idea with
    the identity and the moment each card went, which is the only version that
    can be acted on. `state` turns the three into one word.
    """
    keys = {card_key(card) for card in gpus}
    now = time.time()
    down = {key: value for key, value in (state.get("gpus_down") or {}).items() if key not in keys}
    absent = []
    for key, info in down.items():
        since_ts = info.get("since_ts") or now
        absent.append({
            **info,
            "seconds": int(max(0, now - since_ts)),
            "wanted": bool(GPU_EXPECTED_IDS) and key in GPU_EXPECTED_IDS,
        })
    # A card the settings require and this machine has never shown is not an
    # absence with a timestamp: it is a card that has never been there, and the
    # dashboard has to be able to say exactly that.
    seen = set(state.get("gpus_known") or []) | set((state.get("gpus_names") or {}).keys())
    for wanted in GPU_EXPECTED_IDS:
        if wanted in keys or wanted in down or wanted in seen:
            continue
        absent.append({
            "key": wanted, "uuid": "" if not wanted.startswith(("GPU-", "MIG-")) else wanted,
            "pci": "" if wanted.startswith(("GPU-", "MIG-")) else wanted,
            "name": "not seen", "since": None, "seconds": None, "wanted": True, "lost": 0,
        })
    absent.sort(key=lambda info: info.get("since_ts") or 0)
    # A card that comes and goes is a different problem from a card that is
    # gone, and the history is the only place that remembers the difference.
    drops = {}
    for event in state.get("gpus_events") or []:
        if event.get("kind") in ("removed", "blind") and float(event.get("ts") or 0) >= now - 3600:
            drops[event["key"]] = drops.get(event["key"], 0) + 1
    flapping = []
    for key, count in drops.items():
        if count < 3:
            continue
        known = (state.get("gpus_names") or {}).get(key)
        flapping.append({"key": key, "name": _meta(key, known)["name"] if known else "GPU", "lost": count})
    count = len(gpus)
    missing = max(0, GPU_EXPECTED - count) if GPU_EXPECTED else len(absent)
    health = "empty" if count == 0 else ("partial" if (missing or absent or flapping) else "ok")
    return {
        "count": count,
        "expected": GPU_EXPECTED,
        "expected_ids": GPU_EXPECTED_IDS,
        "missing": missing,
        "state": health,
        "present": [
            {
                "key": card_key(card), "uuid": card.get("uuid"), "index": card.get("index"),
                "name": card.get("name"), "pci": card.get("pci"), "vendor": card.get("vendor"),
                "driver": card.get("driver"), "bus_visible": card.get("bus_visible"),
                "driver_visible": card.get("driver_visible"),
                "power_settable": card.get("power_settable"), "aer": card.get("aer"),
            }
            for card in gpus
        ],
        "absent": absent,
        "events": list(state.get("gpus_events") or [])[-10:],
        "flapping": flapping,
        "last_change": state.get("gpus_last_change"),
        # `gpu_status` already reduced the probe to what a client can use.
        "probe": probe if probe is not None else {},
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
        # Written down at most once a minute: a container restart used to reset
        # the idle timer, so a panel that came back saw "idle since now".
        if last_busy - float(state.get("last_busy") or 0) > 60:
            state["last_busy"] = last_busy
            save_state()
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
                    done.append(f"Jellyfin on {JELLYFIN_DEVICES[JELLYFIN_FALLBACK]}")
            llm_blocked = True
        # Outside the `llm_blocked` check: a ComfyUI that was stopped on a later
        # tick still happened, and the last-action line is the only record of it.
        if done:
            record_action(f"LLM in use: {', '.join(done)} (card free for llama.cpp)")
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
                        back.append(f"Jellyfin back to {JELLYFIN_DEVICES.get(accel, accel)}")
                llm_blocked = False
                names = ", ".join(model_names([{"id": u["model"]} for u in unloaded]))
                record_action(f"GPU idle {idle // 60} min: unloaded {names}"
                              f"{('; ' + ', '.join(back)) if back else ''}")
    else:
        llm_blocked = False
        # An idle ComfyUI hands the card back on its own.
        if comfy_up and state["comfy_idle_min"] and idle > state["comfy_idle_min"] * 60:
            if comfy_release(f"ComfyUI idle {idle // 60} min: stopped, card free again"):
                comfy_up = False

    if IMMICH and state["ml_idle_min"] and idle > state["ml_idle_min"] * 60:
        if immich_ml_state() == "running" and immich_ml_queue_active() == 0:
            with contextlib.suppress(HTTPException):
                immich_ml("stop")
            record_action(f"Immich ML idle {idle // 60} min: stopped to free VRAM")


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


# Guessing the password from the LAN is slow already (LOGIN_DELAY); this makes
# it stop instead. In memory on purpose: a restart is not a reason to forget.
LOGIN_MAX_FAILURES = number("LOGIN_MAX_FAILURES", 5)
LOGIN_WINDOW = number("LOGIN_WINDOW", 300)
LOGIN_FAILURES = {}
LOGIN_LOCK = threading.Lock()


def login_throttled(where):
    """True when this caller has used up its failures in the window."""
    if not LOGIN_MAX_FAILURES:
        return False
    now = time.time()
    with LOGIN_LOCK:
        recent = [when for when in LOGIN_FAILURES.get(where, []) if now - when < LOGIN_WINDOW]
        LOGIN_FAILURES[where] = recent
        return len(recent) >= LOGIN_MAX_FAILURES


def login_failed(where):
    with LOGIN_LOCK:
        LOGIN_FAILURES.setdefault(where, []).append(time.time())


@app.post("/api/login")
def login(request: Request, body: dict = Body(...)):
    client = getattr(request, "client", None)
    where = getattr(client, "host", None) or "unknown"
    if login_throttled(where):
        raise HTTPException(429, "too many failed logins, try again later")
    user = str(body.get("user") or "").strip()
    password = str(body.get("password") or "")
    ok = bool(PANEL_PASSWORD) and same_text(user, PANEL_USER) and same_text(password, PANEL_PASSWORD)
    if not ok:
        login_failed(where)
        time.sleep(LOGIN_DELAY)
        raise HTTPException(401, "wrong user or password")
    with LOGIN_LOCK:
        LOGIN_FAILURES.pop(where, None)
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
    probe = gpu.pop("probe", None)
    # A page load is also an observation: the set is compared here, not only on
    # the 20-second tick, so a card that just came back shows up at once.
    track_gpus(out["gpus"])
    out["inventory"] = inventory_view(out["gpus"], probe)
    # What this deployment has: the dashboard says "not configured" instead of
    # showing a consumer's controls as broken.
    out["configured"] = {
        "immich": IMMICH,
        "jellyfin": JELLYFIN,
        "comfyui": COMFY,
        "comfyui_api": COMFY_API,
        "hermes_webui": WEBUI,
        "llama_logs": bool(AI_HOST),
        "llama_models_dir": bool(LLAMA_MODELS_DIR),
    }
    jobs = {
        "immich_ml": immich_ml_state,
        "immich_ffmpeg": immich_transcode,
        "jellyfin_accel": jellyfin_accel,
        "llama": llama_models,
        "comfyui": comfy_stats,
        "hermes_webui": hermes_webui_stats,
    }
    if LLAMA_MODELS_DIR:
        jobs["disk"] = models_on_disk
    probes = gather_probes(jobs)
    # Popped before the loop: the answers land under their own name (`llama`),
    # and this one is crossed with the router's before it is published.
    disk = probes.pop("disk", [])
    for key, value in probes.items():
        # Each answer is kept, including the ones that are exceptions: a probe
        # that failed is a reading too, and it is not worth a 500.
        if isinstance(value, Exception):
            out[key] = {"error": str(value)}
        else:
            out[key] = value

    models = out.get("llama")
    if isinstance(models, dict):
        out["llama_counts"] = {"error": models.get("error")}
    else:
        # Counts cover every model, not only the listed ones.
        out["llama"] = listed_models(models)
        out["llama_counts"] = {
            "total": len(models),
            "loaded": sum(1 for model in models if model["status"] == "loaded"),
        }
    if isinstance(disk, Exception):
        out["disk_models"] = {"error": str(disk)}
    else:
        out["disk_models"] = disk_models_view(disk, models if isinstance(models, list) else [])
    comfyui = out.get("comfyui")
    if not isinstance(comfyui, dict):
        out["comfyui"] = {"state": "unknown", "error": "unexpected answer"}
    elif "error" in comfyui and "state" not in comfyui:
        out["comfyui"] = {"state": "unknown", **comfyui}
    out["comfyui"]["url"] = COMFY_UI_URL
    return out


def attempt(result, key, action, done, label=None):
    """Run one consumer action, keep its answer, and never claim a skip.

    The mode endpoints used to raise halfway through: ComfyUI stopped, the
    router down, and `last_action` still saying nothing happened. Each action
    now answers for itself and the message says only what ran.
    """
    try:
        result[key] = action()
    except Exception as exc:  # noqa: BLE001 - one consumer failing is not the mode
        result[key] = {"error": str(exc)}
        return False
    if label:
        done.append(label)
    return True


def unload_models(result, done):
    """Unload llama.cpp, and only say so when something was actually unloaded."""
    try:
        unloaded = llama_unload_all()
    except Exception as exc:  # noqa: BLE001 - a router that is down is an answer
        result["llama"] = {"error": str(exc)}
        return False
    result["llama"] = unloaded
    if unloaded:
        done.append("llama.cpp unloaded")
        return True
    return False


@app.post("/api/mode/{mode}")
def set_mode(mode: str, _: str = Depends(require_auth)):
    global llm_blocked
    # Validated before anything is stopped: an unknown mode used to stop ComfyUI
    # and then answer 400.
    if mode not in ("llm", "media", "balanced", "image"):
        raise HTTPException(400, "mode must be llm, media, balanced or image")
    result = {}
    done = []
    # ComfyUI would keep holding the card: stop it before leaving Image mode.
    if mode in ("llm", "media", "balanced") and COMFY and comfy_state() == "running":
        attempt(result, "comfyui", lambda: comfy("stop"), done, "ComfyUI stopped")

    if mode == "llm":
        if IMMICH:
            attempt(result, "immich_ml", lambda: immich_ml("stop"), done, "Immich ML stopped")
        if JELLYFIN:
            attempt(result, "jellyfin", jellyfin_release_gpu, done,
                    f"Jellyfin on {JELLYFIN_DEVICES[JELLYFIN_FALLBACK]}")
        llm_blocked = True
        record_action(f"mode LLM: {', '.join(done) or 'nothing to move'} (card free for llama.cpp)")
    elif mode == "media":
        unload_models(result, done)
        if IMMICH:
            attempt(result, "immich_ml", lambda: immich_ml("start"), done, "Immich ML on")
        if JELLYFIN:
            attempt(result, "jellyfin", lambda: jellyfin_ensure_accel("nvenc"), done,
                    f"Jellyfin on {JELLYFIN_DEVICES['nvenc']}")
        state["jellyfin_before_llm"] = None
        llm_blocked = False
        record_action(f"mode Media: {', '.join(done) or 'nothing to move'}")
    elif mode == "balanced":
        if IMMICH:
            attempt(result, "immich_ml", lambda: immich_ml("start"), done, "Immich ML on")
        if JELLYFIN:
            attempt(result, "jellyfin", lambda: jellyfin_ensure_accel("nvenc"), done,
                    f"Jellyfin on {JELLYFIN_DEVICES['nvenc']}")
        state["jellyfin_before_llm"] = None
        llm_blocked = False
        record_action(f"mode Shared: {', '.join(done) or 'nothing to move'} (llama.cpp not unloaded)")
    else:
        # Image generation needs the whole card: unload llama.cpp, stop the CUDA
        # Immich workload and move Jellyfin off (it keeps transcoding).
        unload_models(result, done)
        if IMMICH:
            attempt(result, "immich_ml", lambda: immich_ml("stop"), done, "Immich ML stopped")
        if JELLYFIN:
            attempt(result, "jellyfin", jellyfin_release_gpu, done,
                    f"Jellyfin on {JELLYFIN_DEVICES[JELLYFIN_FALLBACK]}")
        state["jellyfin_before_llm"] = None
        if COMFY:
            attempt(result, "comfyui", lambda: comfy("start"), done,
                    "ComfyUI started with the card to itself")
        llm_blocked = False
        record_action(f"mode Image: {', '.join(done) or 'nothing to move'}")
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
        raise HTTPException(400, "watts must be an integer") from None
    # The range is the cards', not a constant, and two cards in one machine can
    # differ: validate against the widest range and then let each card answer for
    # itself, because `nvidia-smi -pl` without `-i` fails as a whole as soon as
    # one of them does not accept the value.
    cards = gpu_status()["gpus"]
    commands = [(card, power_command(card, watts)) for card in cards]
    settable = [(card, command) for card, command in commands if command]
    skipped = [(card, command) for card, command in commands if not command]
    ranges = [(card.get("index"), as_number(card.get("power_min"), None), as_number(card.get("power_max"), None))
              for card, _ in settable]
    known = [(index, low, high) for index, low, high in ranges if low is not None and high is not None]
    if known:
        low, high = min(low for _, low, _ in known), max(high for _, _, high in known)
        if not low <= watts <= high:
            raise HTTPException(400, f"watts must be between {int(low)} and {int(high)}")
    if not settable:
        raise HTTPException(400, "no card on this host exposes a settable power limit")

    results = []
    for card, command in settable:
        p = ssh(GPU_HOST, command)
        entry = {"index": card.get("index"), "ok": p.returncode == 0}
        if p.returncode != 0:
            entry["error"] = (p.stderr or p.stdout).strip()[:200] or "the driver refused the limit"
        results.append(entry)
    for card, _ in skipped:
        # Not a refusal: a vendor with no power cap is a different answer than
        # a card that was asked and said no.
        results.append({"index": card.get("index"), "vendor": card.get("vendor"),
                        "skipped": "this vendor exposes no power limit"})
    if not any(entry.get("ok") for entry in results):
        raise HTTPException(502, results[0].get("error") or "no card accepted the limit")

    refused = [entry["index"] for entry in results if entry.get("ok") is False]
    record_action(
        f"power limit = {watts} W"
        + (f" ({len(refused)} card(s) refused: {', '.join(str(i) for i in refused)})" if refused else "")
    )
    return {"ok": not refused, "watts": watts, "cards": results}


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


@app.post("/api/hermes-webui/{action}")
def hermes_webui_action(action: str, _: str = Depends(require_auth)):
    return hermes_webui(action)


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

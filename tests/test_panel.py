"""What the panel promises to a deployment that is not this one.

No GPU, no SSH: `app` runs with the required variables and every probe is
replaced by a stub, which is how a setup with two cards, none of the consumers,
or a card that answers `N/A` can be exercised.
"""

import json
import subprocess
import types

import pytest
from conftest import load


class Proc:
    """What `subprocess.run` returns, which is what every probe checks."""

    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


CARD = {
    "index": "0",
    "uuid": "GPU-aaaa-bbbb",
    "name": "Some Card",
    "driver": "1.0",
    "mem_used": "100",
    "mem_total": "16384",
    "util": "0",
    "temp": "40",
    "power_draw": "30",
    "power_limit": "200",
    "power_default": "250",
    "power_min": "100",
    "power_max": "300",
    "clk_sm": "1328",
    "clk_mem": "715",
    "pcie_gen": "3",
    "pcie_width": "16",
    "processes": [],
}


def card(**overrides):
    one = dict(CARD, **overrides)
    one["util_max"] = int(float(one["util"])) if one["util"] not in ("N/A", None) else 0
    one["power_limits"] = [one["power_limit"]]
    one["gpus"] = [dict(one, gpus=[])] if False else [one]
    return one


def stub_gpu(panel, *cards):
    """Point the panel at these cards; `/api/status` reads GPU 0 plus the list."""
    single = dict(cards[0])
    single["gpus"] = [dict(c) for c in cards]
    single["util_max"] = max(c["util_max"] for c in cards)
    single["power_limits"] = [c["power_limit"] for c in cards]
    panel.gpu_status = lambda: dict(single)


# --- configuration -----------------------------------------------------------
def test_required_variables_name_themselves(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="GPU_HOST"):
        load(monkeypatch, tmp_path, GPU_HOST="")
    with pytest.raises(RuntimeError, match="PANEL_PASSWORD"):
        load(monkeypatch, tmp_path, PANEL_PASSWORD="")
    with pytest.raises(RuntimeError, match="LLAMA_URL"):
        load(monkeypatch, tmp_path, LLAMA_URL="")


def test_old_names_are_aliases(monkeypatch, tmp_path):
    panel = load(monkeypatch, tmp_path, GPU_HOST="", PVE_HOST="root@old", FRONTEND_HOST="root@jf")
    assert panel.GPU_HOST == "root@old"
    assert panel.JELLYFIN_HOST == "root@jf"
    assert panel.JELLYFIN is False  # it also needs the config file


def test_ai_host_defaults_to_the_gpu_host(monkeypatch, tmp_path):
    panel = load(monkeypatch, tmp_path)
    assert panel.AI_HOST == panel.GPU_HOST
    assert panel.AI_CTID == ""
    assert panel.ai_cmd("docker ps") == "docker ps"


def test_ai_ctid_wraps_the_command(monkeypatch, tmp_path):
    panel = load(monkeypatch, tmp_path, AI_CTID="126")
    assert panel.ai_cmd("docker ps") == "pct exec 126 -- docker ps"


def test_bad_numbers_fail_loudly(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="GPU_EXPECTED must be a number"):
        load(monkeypatch, tmp_path, GPU_EXPECTED="two")


# --- a deployment with no consumers -----------------------------------------
def test_minimal_setup_reports_consumers_unconfigured(panel):
    stub_gpu(panel, card())
    status = panel.status("t")

    assert status["configured"] == {
        "immich": False,
        "jellyfin": False,
        "comfyui": False,
        "comfyui_api": False,
        "llama_logs": True,
    }
    assert status["immich_ml"] == "unconfigured"
    assert status["jellyfin_accel"] == "unconfigured"
    assert status["comfyui"]["state"] == "unconfigured"
    assert status["immich_ffmpeg"] == {}
    assert status["inventory"]["count"] == 1
    assert status["inventory"]["expected"] == 0
    assert status["inventory"]["missing"] == 0


def test_modes_take_no_action_that_does_not_exist(panel):
    stub_gpu(panel, card())
    panel.llama_unload_all = lambda: [{"model": "big", "http": 200}]

    for mode in ("llm", "balanced"):
        assert panel.set_mode(mode, "t") == {}
    assert "nothing to move" in panel.state["last_action"]

    panel.set_mode("image", "t")
    assert panel.state["last_action"] == "mode Image: llama.cpp unloaded"


def test_auto_switch_leaves_absent_consumers_alone(panel):
    stub_gpu(panel, card(util="90"))
    panel.llama_models = lambda: [{"id": "big", "status": "loading"}]
    calls = []
    panel.run = lambda cmd, timeout=30, **kw: calls.append(cmd) or Proc()

    panel.auto_tick()

    # A busy card with a model loading: nothing to stop, because nothing exists.
    assert calls == []
    assert panel.state["last_action"] is None


def test_a_comfy_only_stop_is_still_recorded(monkeypatch, tmp_path):
    panel = load(monkeypatch, tmp_path, COMFY_CONTAINER="comfyui")
    stub_gpu(panel, card(util="90"))
    panel.llama_models = lambda: [{"id": "big", "status": "loaded"}]
    # Already in LLM priority: this tick only has ComfyUI left to move.
    monkeypatch.setattr(panel, "llm_blocked", True)
    monkeypatch.setattr(panel, "comfy_state", lambda: "running")
    monkeypatch.setattr(panel, "comfy", lambda action: {"action": action, "ok": True})

    panel.auto_tick()

    assert panel.state["last_action"].startswith("LLM in use: ComfyUI stopped")


# --- readings the card does not report --------------------------------------
def test_na_readings_do_not_crash_the_status(panel):
    stub_gpu(panel, card(util="N/A", power_min="N/A", power_max="N/A", power_limit="N/A"))
    status = panel.status("t")
    assert status["gpu"]["util_max"] == 0
    assert status["inventory"]["count"] == 1


def test_na_power_range_is_left_to_nvidia_smi(panel):
    stub_gpu(panel, card(power_min="N/A", power_max="N/A"))
    seen = []

    def fake_ssh(host, command, timeout=30):
        seen.append(command)
        return Proc()

    panel.ssh = fake_ssh
    out = panel.set_power({"watts": 150}, "t")
    assert (out["ok"], out["watts"]) == (True, 150)
    assert seen == ["nvidia-smi -i 0 -pl 150"]


def test_power_limit_is_bounded_by_the_card(panel):
    stub_gpu(panel, card(power_min="100", power_max="300"))
    panel.ssh = lambda host, command, timeout=30: Proc()

    for watts in (99, 301):
        with pytest.raises(panel.HTTPException) as exc:
            panel.set_power({"watts": watts}, "t")
        assert exc.value.status_code == 400
        assert exc.value.detail == "watts must be between 100 and 300"

    assert panel.set_power({"watts": 250}, "t")["ok"] is True


def test_power_is_applied_to_every_card(panel):
    stub_gpu(panel, card(index="0"), card(index="1", uuid="GPU-bbbb"))
    calls = []
    panel.ssh = lambda host, command, timeout=30: calls.append(command) or Proc()

    out = panel.set_power({"watts": 250}, "t")

    assert calls == ["nvidia-smi -i 0 -pl 250", "nvidia-smi -i 1 -pl 250"]
    assert out["ok"] is True
    assert [c["index"] for c in out["cards"]] == ["0", "1"]


def test_a_card_that_refuses_the_limit_is_reported(panel):
    # Two different cards: the second one tops out below what was asked.
    stub_gpu(panel, card(index="0"), card(index="1", uuid="GPU-bbbb", power_max="200"))

    def fake_ssh(host, command, timeout=30):
        if "-i 1" in command:
            return Proc(returncode=1, stderr="Setting power limit for GPU 0000:02:00.0 failed")
        return Proc()

    panel.ssh = fake_ssh
    out = panel.set_power({"watts": 250}, "t")

    assert out["ok"] is False
    assert out["cards"][0] == {"index": "0", "ok": True}
    assert out["cards"][1]["error"].startswith("Setting power limit")
    assert "1 card(s) refused" in panel.state["last_action"]


def test_no_card_accepting_the_limit_is_a_502(panel):
    stub_gpu(panel, card())
    panel.ssh = lambda host, command, timeout=30: Proc(returncode=1, stderr="not supported")
    with pytest.raises(panel.HTTPException) as exc:
        panel.set_power({"watts": 200}, "t")
    assert exc.value.status_code == 502
    assert exc.value.detail == "not supported"


def test_a_value_outside_every_range_is_refused_before_any_command(panel):
    stub_gpu(panel, card(index="0", power_max="200"), card(index="1", uuid="GPU-bbbb", power_max="200"))
    calls = []
    panel.ssh = lambda host, command, timeout=30: calls.append(command) or Proc()
    with pytest.raises(panel.HTTPException) as exc:
        panel.set_power({"watts": 250}, "t")
    assert exc.value.detail == "watts must be between 100 and 200"
    assert calls == []


def test_a_card_without_a_settable_range_still_reports(panel):
    stub_gpu(panel, card(power_min="N/A", power_max="N/A"))
    assert panel.as_number("N/A", None) is None
    assert panel.as_number(" 125.00 ") == 125.0
    assert panel.as_number(None, 7) == 7


# --- timeouts and dead hosts ------------------------------------------------
def test_unreachable_host_is_a_result_not_a_traceback(panel, monkeypatch):
    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 6)

    monkeypatch.setattr(subprocess, "run", boom)
    result = panel.run(["ssh", "root@nowhere", "true"], timeout=6)
    assert result.returncode == 124
    assert "timed out" in result.stderr

    def missing(*args, **kwargs):
        raise FileNotFoundError("ssh")

    monkeypatch.setattr(subprocess, "run", missing)
    assert panel.run(["ssh"]).returncode == 127


# --- talking to the router ---------------------------------------------------
def test_a_trailing_slash_in_the_router_url_is_harmless(monkeypatch, tmp_path):
    panel = load(monkeypatch, tmp_path, LLAMA_URL="http://llama.test:8080/")
    seen = []

    class Response:
        status_code = 200

        def json(self):
            return {"data": []}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(panel.httpx, "request", lambda method, url, **kw: seen.append(url) or Response())
    panel.llama_models()
    assert seen == ["http://llama.test:8080/v1/models"]


def test_router_auth_is_optional(panel, monkeypatch):
    seen = []

    class Response:
        status_code = 200

        def json(self):
            return {"data": [{"id": "m", "status": {"value": "loaded"}}]}

        def raise_for_status(self):
            return None

    def fake_request(method, url, **kwargs):
        seen.append((method, url, kwargs.get("headers") or {}))
        return Response()

    monkeypatch.setattr(panel.httpx, "request", fake_request)
    assert panel.llama_models()[0]["id"] == "m"
    assert seen[0][2] == {}  # no key configured, no header

    panel.LLAMA_HEADERS.clear()
    panel.LLAMA_HEADERS.update({"Authorization": "Bearer secret"})
    panel.llama_models()
    assert seen[1][2] == {"Authorization": "Bearer secret"}


def test_the_key_reaches_the_unload_call(monkeypatch, tmp_path):
    panel = load(monkeypatch, tmp_path, LLAMA_API_KEY="secret")
    seen = []

    class Response:
        status_code = 200

        def json(self):
            return {"data": [{"id": "big", "status": {"value": "loaded"}}]}

        def raise_for_status(self):
            return None

    def fake_request(method, url, **kwargs):
        seen.append((method, url, kwargs.get("headers") or {}, kwargs.get("json")))
        return Response()

    monkeypatch.setattr(panel.httpx, "request", fake_request)
    assert panel.llama_unload_all() == [{"model": "big", "http": 200}]
    assert seen[-1] == ("POST", "http://llama.test:8080/models/unload",
                        {"Authorization": "Bearer secret"}, {"model": "big"})


# --- inventory --------------------------------------------------------------
def test_inventory_tracks_the_card_set(panel, tmp_path):
    a = {"uuid": "GPU-aaaa-1111", "name": "Card A", "index": "0"}
    b = {"uuid": "GPU-bbbb-2222", "name": "Card B", "index": "1"}

    panel.track_gpus([a, b])
    assert panel.state["gpus_known"] == [a["uuid"], b["uuid"]]
    assert panel.state["gpus_last_change"] is None  # first look is the baseline

    panel.track_gpus([b, a])
    assert panel.state["gpus_last_change"] is None  # order is not a change

    panel.track_gpus([a])
    change = panel.state["gpus_last_change"]
    assert [(card["uuid"], card["name"], card["key"]) for card in change["removed"]] == [
        (b["uuid"], "Card B", b["uuid"])
    ]
    assert "1 GPU lost: Card B (bbbb)" in panel.state["last_action"]
    # The card is remembered as missing, with the moment it went.
    assert panel.state["gpus_down"][b["uuid"]]["since"]
    assert panel.inventory_view([a])["state"] == "partial"

    panel.track_gpus([a, b])
    assert [(card["uuid"], card["name"]) for card in panel.state["gpus_last_change"]["added"]] == [
        (b["uuid"], "Card B")
    ]
    assert "1 new GPU: Card B (bbbb)" in panel.state["last_action"]
    assert panel.state["gpus_down"] == {}

    view = panel.inventory_view([a, b])
    assert (view["count"], view["expected"], view["missing"]) == (2, 0, 0)
    assert view["state"] == "ok"
    assert json.loads((tmp_path / "auto.json").read_text())["gpus_known"] == [a["uuid"], b["uuid"]]


def test_expected_count_is_what_makes_a_gap_visible(monkeypatch, tmp_path):
    panel = load(monkeypatch, tmp_path, GPU_EXPECTED="2")
    view = panel.inventory_view([{"uuid": "GPU-a", "index": "0", "name": "Card"}])
    assert (view["count"], view["expected"], view["missing"]) == (1, 2, 1)


# --- auth -------------------------------------------------------------------
def request(**kwargs):
    """The smallest thing that behaves like a Starlette Request for auth."""
    defaults = {"cookies": {}, "headers": {}, "method": "GET", "url": "http://panel.test/"}
    return types.SimpleNamespace(**{**defaults, **kwargs})


def test_session_roundtrip(panel):
    cookie = panel.make_session("admin")
    assert panel.session_payload(request(cookies={panel.SESSION_COOKIE: cookie}))["u"] == "admin"

    body, _, signature = cookie.partition(".")
    tampered = body + "." + ("A" if signature[0] != "A" else "B") + signature[1:]
    assert panel.session_payload(request(cookies={panel.SESSION_COOKIE: tampered})) == {}
    assert panel.session_payload(request()) == {}


def test_expired_and_revoked_sessions_are_refused(panel):
    cookie = panel.make_session("admin")
    nonce = panel.session_payload(request(cookies={panel.SESSION_COOKIE: cookie}))["n"]

    panel.state["revoked"] = {nonce: 9999999999}
    assert panel.session_payload(request(cookies={panel.SESSION_COOKIE: cookie})) == {}

    panel.state["revoked"] = {}
    panel.SESSION_TTL = -1
    assert panel.session_payload(request(cookies={panel.SESSION_COOKIE: panel.make_session("admin")})) == {}


def test_logout_revokes_the_nonce(panel, tmp_path):
    cookie = panel.make_session("admin")
    panel.logout(request(cookies={panel.SESSION_COOKIE: cookie}))
    stored = json.loads((tmp_path / "auto.json").read_text())
    assert len(stored["revoked"]) == 1
    assert panel.session_payload(request(cookies={panel.SESSION_COOKIE: cookie})) == {}


def test_bearer_token_and_csrf_guard(panel):
    token = request(headers={"authorization": "Bearer token"})
    assert panel.require_auth(token) == "token"
    with pytest.raises(panel.HTTPException) as exc:
        panel.require_auth(request(headers={"authorization": "Bearer nope"}))
    assert exc.value.status_code == 401

    cookie = panel.make_session("admin")
    with pytest.raises(panel.HTTPException) as exc:
        panel.require_auth(
            request(
                method="POST",
                cookies={panel.SESSION_COOKIE: cookie},
                headers={"origin": "http://evil.test", "host": "panel.test"},
            )
        )
    assert exc.value.status_code == 403

    same_origin = request(
        method="POST",
        cookies={panel.SESSION_COOKIE: cookie},
        headers={"origin": "http://panel.test", "host": "panel.test"},
    )
    assert panel.require_auth(same_origin) == "admin"


def test_non_ascii_password_is_refused_not_crashed(panel):
    with pytest.raises(panel.HTTPException) as exc:
        panel.login(request(), {"user": "admin", "password": "contraseña"})
    assert exc.value.status_code == 401


# --- state on disk ----------------------------------------------------------
def test_state_is_written_atomically(panel, tmp_path):
    panel.record_action("something happened")
    stored = json.loads((tmp_path / "auto.json").read_text())
    assert stored["last_action"] == "something happened"
    assert not (tmp_path / "auto.json.tmp").exists()


def test_a_broken_state_file_is_ignored(panel, tmp_path):
    (tmp_path / "auto.json").write_text("{ this is not json")
    panel.load_state()
    assert panel.state["llama_idle_min"] == 60
    # And the panel keeps working: the next save replaces the broken file.
    panel.record_action("recovered")
    assert json.loads((tmp_path / "auto.json").read_text())["last_action"] == "recovered"


def test_state_path_without_a_directory(panel, tmp_path, monkeypatch):
    panel.STATE_PATH = "auto.json"
    monkeypatch.chdir(tmp_path)
    panel.record_action("relative")
    assert json.loads((tmp_path / "auto.json").read_text())["last_action"] == "relative"


# --- validation -------------------------------------------------------------
def test_auto_settings_are_validated(panel):
    with pytest.raises(panel.HTTPException) as exc:
        panel.set_auto({"llama_idle_min": 5000}, "t")
    assert exc.value.status_code == 400

    assert panel.set_auto({"llama_idle_min": 30, "enabled": False}, "t")["llama_idle_min"] == 30


def test_llama_logs_are_capped(panel):
    seen = {}

    def fake_ssh(host, command, timeout=30):
        seen["command"] = command
        return Proc("line\n")

    panel.ssh = fake_ssh
    panel.llama_logs(10**6)
    assert "-n 2000" in seen["command"]
    with pytest.raises(panel.HTTPException):
        panel.llama_logs("many")


# --- the probe: two views of the same machine -------------------------------
PROBE_TOOLS = {"nvidia-smi": True}


def probe(sysfs=(), nvinfo=(), gpu_lines=(), proc_lines=(), tools=None, gpu_rc="0"):
    """What the probe script prints, section by section."""
    tools = PROBE_TOOLS if tools is None else tools
    lines = ["@@tools", "tool" + "".join(f"\t{name}={int(ok)}" for name, ok in tools.items())]
    lines.append("@@sysfs")
    lines += ["card" + "".join(f"\t{key}={value}" for key, value in card.items()) for card in sysfs]
    lines.append("@@nvinfo")
    lines += ["nv" + "".join(f"\t{key}={value}" for key, value in info.items()) for info in nvinfo]
    lines.append("@@gpu")
    lines += list(gpu_lines)
    lines += ["@@gpu_rc", gpu_rc, "@@procs"]
    lines += list(proc_lines)
    lines += ["@@procs_rc", "0", "@@end"]
    return "\n".join(lines)


def stub_probe(panel, **kwargs):
    panel.ssh_script = lambda host, script, timeout=60: Proc(probe(**kwargs))


def nvidia_row(index="0", uuid="GPU-1111-2222", name="Tesla P100", **overrides):
    values = {
        "index": index, "uuid": uuid, "name": name, "driver": "580.1",
        "mem_used": "100", "mem_total": "16384", "util": "0", "temp": "40",
        "power_draw": "30", "power_limit": "175.00", "power_default": "250",
        "power_min": "100", "power_max": "250", "clk_sm": "1328", "clk_mem": "715",
        "pcie_gen": "3", "pcie_width": "16",
    }
    values.update(overrides)
    return ",".join(values[key] for key in values)


def test_the_probe_script_formats_nothing_it_does_not_mean(panel):
    # `%` in the script would collide with the CSV substitution, so the fields
    # are a token and not a format string.
    assert "@@end" in panel.PROBE_SCRIPT
    assert "@FIELDS@" in panel.PROBE_SCRIPT
    assert "%s" not in panel.PROBE_SCRIPT.split("@@tools")[1].split("@@end")[0].replace("printf '%s\\n'", "")


def test_a_card_the_driver_lost_is_not_a_refused_probe(panel):
    # On the bus, missing from nvidia-smi: the driver gave up on it. That is an
    # event with an identity, not a 502 with a nvidia-smi error in it.
    stub_probe(
        panel,
        sysfs=[{"bdf": "0000:01:00.0", "drm": "card0", "vendor": "0x10de", "driver": "nvidia",
                "name": "Tesla P100 PCIe 16GB"}],
        nvinfo=[{"uuid": "GPU-1111-2222", "bdf": "0000:01:00.0"}],
        gpu_lines=[],
    )
    gpu = panel.gpu_status()

    card = gpu["gpus"][0]
    assert (card["bus_visible"], card["driver_visible"]) == (True, False)
    assert panel.card_key(card) == "GPU-1111-2222"  # the bus entry keeps its UUID
    assert gpu["util_max"] == 0


def test_a_machine_whose_cards_all_went_is_an_event_not_a_502(monkeypatch, tmp_path):
    panel = load(monkeypatch, tmp_path, GPU_EXPECTED="2")
    board = [
        {"uuid": "GPU-aaaa-1111", "name": "Card A", "pci": "0000:01:00.0", "index": "0"},
        {"uuid": "GPU-bbbb-2222", "name": "Card B", "pci": "0000:02:00.0", "index": "1"},
    ]
    panel.track_gpus(board)  # baseline
    stub_probe(panel)  # the host answers, with no card at all

    status = panel.status("t")

    assert status["gpus"] == []
    assert status["gpu"]["name"] == "no cards"
    assert (status["inventory"]["count"], status["inventory"]["state"]) == (0, "empty")
    assert {card["key"] for card in status["inventory"]["absent"]} == {"GPU-aaaa-1111", "GPU-bbbb-2222"}
    assert panel.state["gpus_last_change"]["removed"]
    absent = status["inventory"]["absent"]
    assert all(card["seconds"] >= 0 for card in absent)
    assert all(card["name"] for card in absent)


def test_an_amd_card_is_read_from_sysfs(panel):
    stub_probe(
        panel,
        tools={"amd-smi": True},
        sysfs=[{
            "bdf": "0000:03:00.0", "drm": "card0", "vendor": "0x1002", "driver": "amdgpu",
            "name": "Radeon RX 7900 XTX", "vram_used": str(2048 * 2**20), "vram_total": str(24576 * 2**20),
            "busy": "42", "temp": "55000", "power": "120000000", "cap": "250000000",
            "cap_min": "150000000", "cap_max": "300000000", "clk": "2500000000",
            "clk_mem": "1000000000", "link_speed": "16.0 GT/s PCIe", "link_width": "16",
            "aer_fatal": "0", "aer_nonfatal": "1", "aer_correctable": "0",
        }],
    )
    card = panel.gpu_status()["gpus"][0]

    assert (card["vendor"], card["driver"], card["name"]) == ("amd", "amdgpu", "Radeon RX 7900 XTX")
    assert (card["mem_used"], card["mem_total"], card["util"], card["temp"]) == ("2048", "24576", "42", "55")
    assert (card["power_draw"], card["power_limit"]) == ("120", "250.00")
    assert (card["clk_sm"], card["clk_mem"]) == ("2500", "1000")
    assert (card["pcie_gen"], card["pcie_width"]) == ("4", "16")
    assert card["aer"]["nonfatal"] == "1"
    assert (card["setter"], card["power_settable"]) == ("amd-smi", True)
    assert panel.card_key(card) == "0000:03:00.0"


def test_the_amd_power_limit_goes_through_amd_smi(panel):
    card = dict(CARD, vendor="amd", pci="0000:03:00.0", setter="amd-smi", power_min="150", power_max="300")
    card.update({"util_max": 0, "power_limits": ["250.00"], "gpus": [card]})
    panel.gpu_status = lambda: dict(card, gpus=[dict(card)])
    calls = []
    panel.ssh = lambda host, command, timeout=30: calls.append(command) or Proc()

    assert panel.set_power({"watts": 250}, "t")["ok"] is True
    assert calls == ["amd-smi set -g 0 --power-cap 250"]


def test_an_intel_card_has_no_power_control(panel):
    stub_probe(
        panel,
        sysfs=[{"bdf": "0000:00:02.0", "drm": "card0", "vendor": "0x8086", "driver": "i915",
                "name": "Xeon E3-1200 v2 Graphics", "link_width": "1"}],
    )
    card = panel.gpu_status()["gpus"][0]

    assert (card["vendor"], card["driver"], card["power_settable"]) == ("intel", "i915", False)
    assert card["mem_total"] == "N/A"
    with pytest.raises(panel.HTTPException) as exc:
        panel.set_power({"watts": 100}, "t")
    assert exc.value.status_code == 400
    assert "no card" in exc.value.detail


def test_a_card_that_went_blind_is_its_own_event(panel):
    board = [{"uuid": "GPU-1111-2222", "name": "Tesla P100", "pci": "0000:01:00.0",
              "index": "0", "driver_visible": True}]
    panel.track_gpus(board)
    panel.track_gpus([{**board[0], "driver_visible": False}])

    kinds = [event["kind"] for event in panel.state["gpus_events"]]
    assert kinds == ["blind"]
    assert "not in the driver" in panel.state["last_action"]
    assert panel.state["gpus_down"] == {}  # it is still there

    panel.track_gpus(board)
    assert [event["kind"] for event in panel.state["gpus_events"]] == ["blind", "visible"]


def test_a_change_carries_the_kernel_lines_that_explain_it(panel):
    board = [{"uuid": "GPU-1111-2222", "name": "Tesla P100", "pci": "0000:01:00.0", "index": "0"}]
    panel.track_gpus(board)
    panel.ssh = lambda host, command, timeout=30: Proc("NVRM: Xid 79, GPU has fallen off the bus\n")

    panel.track_gpus([])

    event = panel.state["gpus_events"][-1]
    assert event["kind"] == "removed"
    assert event["evidence"] == ["NVRM: Xid 79, GPU has fallen off the bus"]


def test_a_dropped_card_that_keeps_crossing_the_bus_is_named_once(panel):
    board = [{"uuid": "GPU-1111-2222", "name": "Tesla P100", "pci": "0000:01:00.0", "index": "0"}]
    panel.track_gpus(board)
    panel.ssh = lambda host, command, timeout=30: Proc()

    for _ in range(3):
        panel.track_gpus([])
        panel.track_gpus(board)

    # It is back, so nothing is missing; the flapping is what the history says.
    view = panel.inventory_view(board)
    assert view["absent"] == []
    assert view["flapping"] == [{"key": "GPU-1111-2222", "name": "Tesla P100", "lost": 3}]
    # The history is bounded, however often a card flaps.
    assert len(panel.state["gpus_events"]) <= panel.GPU_EVENTS_MAX


def test_expected_ids_name_the_cards_that_should_be_there(monkeypatch, tmp_path):
    panel = load(monkeypatch, tmp_path, GPU_EXPECTED_IDS="0000:01:00.0,0000:02:00.0")
    board = [{"uuid": "0000:01:00.0", "name": "Card A", "pci": "0000:01:00.0", "index": "0", "vendor": "amd"}]
    gone = {"uuid": "0000:02:00.0", "name": "Card B", "pci": "0000:02:00.0", "index": "1", "vendor": "amd"}
    panel.track_gpus(board + [gone])  # baseline
    panel.track_gpus(board)  # the second card went
    view = panel.inventory_view(board)

    assert [card["key"] for card in view["absent"]] == ["0000:02:00.0"]
    assert view["absent"][0]["wanted"] is True
    assert view["missing"] == 1


# --- actions that used to be claimed ----------------------------------------
def test_a_router_that_is_down_is_not_a_claimed_unload(panel):
    def boom():
        raise RuntimeError("router down")

    panel.llama_unload_all = boom

    out = panel.set_mode("media", "t")

    assert "error" in out["llama"]
    assert "unloaded" not in panel.state["last_action"]


def test_an_unknown_mode_stops_nothing(monkeypatch, tmp_path):
    panel = load(monkeypatch, tmp_path, COMFY_CONTAINER="comfyui")
    calls = []
    panel.comfy_state = lambda: "running"
    panel.comfy = lambda action: calls.append(action) or {"ok": True}

    with pytest.raises(panel.HTTPException) as exc:
        panel.set_mode("nonsense", "t")

    assert (exc.value.status_code, calls) == (400, [])


def test_guessing_the_password_stops(panel):
    for _ in range(5):
        with pytest.raises(panel.HTTPException) as exc:
            panel.login(request(), {"user": "admin", "password": "nope"})
        assert exc.value.status_code == 401

    with pytest.raises(panel.HTTPException) as exc:
        panel.login(request(), {"user": "admin", "password": "nope"})
    assert exc.value.status_code == 429


def test_a_reading_the_vendor_does_not_report_is_not_a_number(panel):
    assert panel._tidy("55000", 1000) == "55"
    assert panel._tidy("") == "N/A"
    assert panel._pcie_gen("16.0 GT/s PCIe") == "4"
    assert panel._pcie_gen("N/A") == "N/A"


def test_the_drivers_rows_learn_their_pci_address(panel):
    # /proc/driver/nvidia is what ties a UUID to the bus; without it the bus
    # entries looked like a second pair of cards.
    board = [
        {"bdf": "0000:01:00.0", "drm": "card0", "vendor": "0x10de", "driver": "nvidia",
         "aer_fatal": "0", "aer_nonfatal": "0", "aer_correctable": "0"},
        {"bdf": "0000:02:00.0", "drm": "card1", "vendor": "0x10de", "driver": "nvidia"},
    ]
    stub_probe(
        panel,
        gpu_lines=[nvidia_row(uuid="GPU-1111-2222"), nvidia_row(index="1", uuid="GPU-3333-4444")],
        sysfs=board,
        nvinfo=[{"uuid": "GPU-1111-2222", "bdf": "0000:01:00.0"},
                {"uuid": "GPU-3333-4444", "bdf": "0000:02:00.0"}],
    )
    cards = panel.gpu_status()["gpus"]

    assert len(cards) == 2  # no duplicate for the bus view
    assert [card["pci"] for card in cards] == ["0000:01:00.0", "0000:02:00.0"]
    assert all(card["driver_visible"] and card["bus_visible"] for card in cards)
    assert cards[0]["aer"] == {"fatal": "0", "nonfatal": "0", "correctable": "0"}
    assert cards[1]["aer"] is None


def test_a_stray_line_does_not_break_the_records(panel):
    text = probe(sysfs=[{"bdf": "0000:01:00.0", "vendor": "0x1002", "driver": "amdgpu", "busy": "5"}])
    text = text.replace("@@nvinfo", "garbage from a multi-line file\n@@nvinfo")
    panel.ssh_script = lambda *args, **kwargs: Proc(text)

    card = panel.gpu_status()["gpus"][0]
    assert (card["pci"], card["util"]) == ("0000:01:00.0", "5")


def test_the_inventory_carries_the_probe_answer(panel):
    # It used to be reduced twice and the nvidia-smi answer was lost on the way.
    stub_probe(panel)
    status = panel.status("t")
    assert status["inventory"]["probe"]["nvidia_smi"] == "0"
    assert status["inventory"]["probe"]["host_ok"] is True


def test_a_stopped_comfyui_is_not_asked_twice_in_a_row(monkeypatch, tmp_path):
    panel = load(monkeypatch, tmp_path, COMFY_CONTAINER="comfyui", COMFY_URL="http://comfy.test")

    def no_api(*args, **kwargs):
        raise RuntimeError("ComfyUI is stopped")

    monkeypatch.setattr(panel.httpx, "get", no_api)
    calls = []
    panel.ssh = lambda host, command, timeout=30: calls.append(command) or Proc("created\n")

    assert panel.comfy_state() == "created"
    assert panel.comfy_state() == "created"  # remembered, not asked again
    assert len(calls) == 1
    assert panel.comfy_state(force=True) == "created"
    assert len(calls) == 2

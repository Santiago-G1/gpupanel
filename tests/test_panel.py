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
    assert panel.state["gpus_last_change"]["removed"] == [{"uuid": b["uuid"], "name": "Card B"}]
    assert "1 GPU lost: Card B (bbbb)" in panel.state["last_action"]

    panel.track_gpus([a, b])
    assert panel.state["gpus_last_change"]["added"] == [{"uuid": b["uuid"], "name": "Card B"}]
    assert "1 new GPU: Card B (bbbb)" in panel.state["last_action"]

    view = panel.inventory_view([a, b])
    assert (view["count"], view["expected"], view["missing"]) == (2, 0, 0)
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

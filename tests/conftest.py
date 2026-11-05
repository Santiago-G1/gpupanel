"""Shared setup: every test imports `app` fresh, with its own environment.

The module reads its configuration at import time, so a test gets a clean panel
by setting the variables and re-importing. Nothing here needs a GPU, a network
or an SSH key: the probes are monkeypatched per test.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASE = {
    "GPU_HOST": "root@gpu.test",
    "LLAMA_URL": "http://llama.test:8080",
    "PANEL_PASSWORD": "password",
    "PANEL_TOKEN": "token",
}

# Everything `app` reads. A test that wants a different setup starts from an
# empty board, which is also what the container gets.
PREFIXES = ("PANEL_", "IMMICH_", "JELLYFIN_", "COMFY_", "LLAMA_", "AI_", "GPU_", "SESSION_", "LOGIN_", "STATE_")
OLD_NAMES = {"PVE_HOST", "FRONTEND_HOST"}


def load(monkeypatch, tmp_path, **env):
    """Import `app` with the required variables plus whatever the test adds."""
    for key in list(os.environ):
        if key.startswith(PREFIXES) or key in OLD_NAMES:
            monkeypatch.delenv(key, raising=False)
    values = dict(BASE, STATE_PATH=str(tmp_path / "auto.json"), LOGIN_DELAY="0", **env)
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("app", None)
    import app  # noqa: PLC0415 - the point of the fixture

    return app


@pytest.fixture
def panel(monkeypatch, tmp_path):
    """A panel with only the required variables set."""
    return load(monkeypatch, tmp_path)

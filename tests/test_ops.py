"""The container gets every setting the panel reads.

A variable that `app.py` honours but the Compose file never forwards is a
setting that silently does nothing: it was in the README's table, it was in
`.env`, and the panel still ran with its default.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return (ROOT / name).read_text()


def app_variables():
    """Every variable the panel reads: the first name of each `optional()` call.

    A second name in the same call is an alias the panel still accepts for an
    installation that has not been renamed yet, and that one is not forwarded.
    """
    names = set()
    for args in re.findall(r"(?:required|optional|number)\(([^)]*)\)", read("app.py")):
        found = re.findall(r'"([A-Z][A-Z0-9_]*)"', args)
        if found:
            names.add(found[0])
    return names


def composed_variables():
    return set(re.findall(r"^\s*-\s+([A-Z][A-Z0-9_]*)=", read("docker-compose.yml"), re.M))


def template_variables():
    return set(re.findall(r"^([A-Z][A-Z0-9_]*)=", read(".env.template"), re.M))


def test_the_panel_reads_more_than_the_required_four():
    # Guards the regex above: a broken parse would make the two checks below pass.
    assert {"GPU_HOST", "LLAMA_URL", "PANEL_PASSWORD", "PANEL_TOKEN", "GPU_EXPECTED_IDS"} <= app_variables()
    assert "PVE_HOST" not in app_variables()


def test_compose_forwards_every_setting_the_panel_reads():
    missing = app_variables() - composed_variables()
    assert not missing, f"docker-compose.yml does not forward: {sorted(missing)}"


def test_the_template_names_every_setting_the_panel_reads():
    missing = app_variables() - template_variables()
    assert not missing, f".env.template does not document: {sorted(missing)}"


def test_compose_forwards_nothing_the_panel_ignores():
    # The other direction: a stale variable is a setting the operator will
    # change and see nothing happen.
    extra = composed_variables() - app_variables() - {"PUBLISH_ADDR", "PUBLISH_PORT", "TZ", "SSH_PATH"}
    assert not extra, f"docker-compose.yml forwards unused: {sorted(extra)}"

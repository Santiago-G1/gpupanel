"""The dashboard is wired by id: a missing element is a silent JavaScript error."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "static"


def ids(html: str) -> set[str]:
    return set(re.findall(r'id="([^"]+)"', html))


def looked_up(html: str) -> set[str]:
    script = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
    return set(re.findall(r"\$\('([^']+)'\)", script))


@pytest.mark.parametrize("page", ["index.html", "howto.html", "login.html"])
def test_the_pages_and_their_assets_are_there(page):
    assert (STATIC / page).is_file()
    html = (STATIC / page).read_text()
    for asset in re.findall(r'(?:href|src)="(/static/[^"]+)"', html):
        assert (STATIC / asset.removeprefix("/static/")).is_file(), f"{page} links {asset}"


def test_every_element_the_dashboard_looks_up_exists():
    html = (STATIC / "index.html").read_text()
    missing = looked_up(html) - ids(html)
    assert not missing, f"the script looks up ids that are not in the markup: {sorted(missing)}"


def test_the_guest_pages_do_not_load_an_authenticated_asset():
    # login.html is served without a session, so its CSS is inline.
    assert "/static/" not in (STATIC / "login.html").read_text()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_inline_javascript_parses(tmp_path):
    html = (STATIC / "index.html").read_text()
    script_path = tmp_path / "index.js"
    script_path.write_text("\n".join(re.findall(r"<script>(.*?)</script>", html, re.S)))
    result = subprocess.run(["node", "--check", str(script_path)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr

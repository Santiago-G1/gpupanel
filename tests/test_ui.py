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


def vendor_keys(html: str) -> set[str]:
    """The vendor keys the dashboard's own VENDORS map knows about."""
    block = re.search(r"const VENDORS = \{([^}]*)\}", html)
    assert block, "index.html has no VENDORS map"
    return set(re.findall(r"(\w+)\s*:", block.group(1)))


def test_every_vendor_the_page_names_has_a_colour():
    # NVIDIA green, AMD red, Intel blue, plus the muted one for anything else.
    html = (STATIC / "index.html").read_text()
    css = (STATIC / "panel.css").read_text()
    keys = vendor_keys(html)
    # The three the probe knows about have to be there; a fourth vendor is fine
    # as long as it comes with its .vendor-<key> rule, which the loop below checks.
    assert {"nvidia", "amd", "intel"} <= keys, f"the page lost a vendor: {sorted(keys)}"
    for key in sorted(keys | {"unknown"}):
        rule = re.search(rf"\.vendor-{key}\s*\{{([^}}]*)\}}", css)
        assert rule, f"panel.css has no .vendor-{key} rule"
        assert re.search(r"--vendor\s*:", rule.group(1)), f".vendor-{key} sets no --vendor"


def test_a_gpu_block_wears_the_vendor_it_was_probed_as():
    html = (STATIC / "index.html").read_text()
    css = (STATIC / "panel.css").read_text()
    assert re.search(r'class="gpu-block vendor-\$\{vkey\}"', html), "the per-card block carries no vendor class"
    assert re.search(r'class="pill vendor-tag"', html), "the vendor pill is not the coloured one"
    # Accent and label agree because both read the same --vendor. Anchor on the
    # declaration, not on the first rule with that selector: a media query may
    # add another one ahead of it.
    pills = re.findall(r"\.gpu-block \.pill\.vendor-tag\s*\{([^}]*)\}", css)
    assert any("color:var(--vendor)" in rule for rule in pills), "the vendor pill ignores --vendor"
    blocks = re.findall(r"\.gpu-block\s*\{([^}]*)\}", css)
    assert any("var(--vendor" in rule for rule in blocks), "the card block has no vendor accent"


def test_a_vendor_the_probe_does_not_name_is_the_neutral_one():
    # Absent, empty or unrecognised: vendor-unknown, never a bare "vendor-".
    html = (STATIC / "index.html").read_text()
    assert re.search(r"vkey = VENDORS\[x\.vendor\] \? x\.vendor : 'unknown'", html), \
        "the vendor class does not fall back to 'unknown'"
    assert ".vendor-unknown" in (STATIC / "panel.css").read_text()


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

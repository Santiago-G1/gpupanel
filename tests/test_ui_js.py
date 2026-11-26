"""The dashboard's own decisions, run in node against a stub page.

`index.html` is one self-contained script, so a behaviour the tests only match
by regex is a behaviour nobody checked. These run it: the DOM is replaced by an
element that accepts anything, and the functions the page derives its state
from are called with the shapes `/api/status` really sends.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "static"

# Every element answers and accepts anything: what the page does with the DOM is
# not what these tests look at.
STUBS = """
function el() {
  const target = function () {};
  return new Proxy(target, {
    get(t, k) {
      if (k === Symbol.toPrimitive) return () => '';
      if (!(k in t)) t[k] = el();
      return t[k];
    },
    set(t, k, v) { t[k] = v; return true; },
    apply() { return el(); },
  });
}
const ctx = {
  document: {
    getElementById: () => el(),
    querySelector: () => el(),
    querySelectorAll: () => [],
    addEventListener: () => {},
    createElement: () => el(),
    activeElement: null,
  },
  location: {href: 'http://panel.test/', pathname: '/', origin: 'http://panel.test'},
  fetch: () => { ctx.__fetches += 1; return Promise.reject(new Error('offline')); },
  __fetches: 0,
  localStorage: {getItem: () => null, setItem: () => {}},
  setInterval: () => 0, clearInterval: () => {}, setTimeout: () => 0, clearTimeout: () => {},
  encodeURIComponent, console,
};
ctx.globalThis = ctx;
"""

BASE = {
    "llama_counts": {"total": 0, "loaded": 0, "loading": 0},
    "llama": [],
    "comfyui": {"state": "exited"},
}


def run(tmp_path, body):
    """Run `body` with the page's script loaded, and return its JSON answer."""
    html = (STATIC / "index.html").read_text()
    script = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
    (tmp_path / "index.js").write_text(script)
    # The body runs in the page's own context: its `const` and `function`
    # declarations are script-scoped, not properties of the global object.
    (tmp_path / "run.js").write_text(
        "const fs = require('fs'), vm = require('vm');\n"
        "const src = fs.readFileSync(process.argv[2], 'utf8');\n"
        f"{STUBS}"
        "vm.createContext(ctx);\n"
        f"const out = vm.runInContext(src + '\\n' + {json.dumps('(async () => { ' + body + ' })()')}, ctx);\n"
        "Promise.resolve(out)\n"
        "  .then((value) => console.log(JSON.stringify(value)))\n"
        "  .catch((err) => { console.error(err); process.exit(1); });\n"
    )
    result = subprocess.run(
        ["node", str(tmp_path / "run.js"), str(tmp_path / "index.js")],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def status(configured, **state):
    return {**BASE, "configured": configured, **state}


def modes(tmp_path, cases):
    """`deriveMode` for each (state, expected) pair, as a list of (got, want)."""
    payload = json.dumps([state for state, _ in cases])
    got = run(tmp_path, f"return {payload}.map((s) => deriveMode(s));")
    return list(zip(got, [want for _, want in cases], strict=True))


def test_the_accelerator_buttons_are_not_rebuilt_for_the_same_reading(tmp_path):
    body = """
    const cfg = {jellyfin_accel_types: [{value: 'nvenc', label: 'NVENC'},
      {value: 'vaapi', label: 'VAAPI'}], jellyfin_fallback: 'vaapi'};
    const box = {innerHTML: '', dataset: {}, style: {}, textContent: '', disabled: false,
      classList: {toggle() {}, add() {}, remove() {}}, setAttribute() {}, addEventListener() {}};
    const original = document.getElementById;
    document.getElementById = (id) => (id === 'jf' ? box : original(id));
    renderJellyfinAccel({configured: cfg, jellyfin_accel: 'nvenc'});
    const first = box.innerHTML;
    box.innerHTML = '';
    renderJellyfinAccel({configured: cfg, jellyfin_accel: 'nvenc'});
    const second = box.innerHTML;
    renderJellyfinAccel({configured: cfg, jellyfin_accel: 'vaapi'});
    return [first.includes('data-accel="nvenc"'), second, box.innerHTML.includes('on')];
    """
    assert run(tmp_path, body) == [True, "", True]


def test_a_poll_does_not_overlap_the_one_in_flight(tmp_path):
    body = """
    await refresh();
    const before = __fetches;
    const first = refresh();
    const second = refresh();
    return [__fetches - before, first === second];
    """
    # Two callers, one request: the second reads the answer the first is already
    # waiting for instead of asking again and painting it afterwards.
    assert run(tmp_path, body) == [1, True]


def test_a_reading_is_escaped_before_it_reaches_the_markup(tmp_path):
    body = "return [readingHtml('<b>70</b>', 'W'), readingHtml('N/A', 'W'), readingHtml(70, 'W')];"

    assert run(tmp_path, body) == ["&lt;b&gt;70&lt;/b&gt; W", "—", "70 W"]


def test_a_deployment_with_only_immich_is_not_a_jellyfin_fallback(tmp_path):
    only_immich = {"immich": True, "jellyfin": False, "jellyfin_fallback": "vaapi"}
    cases = [
        (status(only_immich, immich_ml="running", jellyfin_accel="unconfigured"), "media"),
        (status(only_immich, immich_ml="running", jellyfin_accel="unconfigured",
                llama_counts={"total": 1, "loaded": 1, "loading": 0}), "balanced"),
        (status(only_immich, immich_ml="exited", jellyfin_accel="unconfigured"), "idle"),
    ]

    assert modes(tmp_path, cases) == [(want, want) for _, want in cases]


def test_a_deployment_with_only_jellyfin_names_the_card_state(tmp_path):
    only_jellyfin = {"immich": False, "jellyfin": True, "jellyfin_fallback": "vaapi"}
    cases = [
        (status(only_jellyfin, immich_ml="unconfigured", jellyfin_accel="nvenc"), "media"),
        (status(only_jellyfin, immich_ml="unconfigured", jellyfin_accel="vaapi"), "idle"),
        # A model that is still loading already holds the VRAM.
        (status(only_jellyfin, immich_ml="unconfigured", jellyfin_accel="vaapi",
                llama_counts={"total": 1, "loaded": 0, "loading": 1}), "llm"),
    ]

    assert modes(tmp_path, cases) == [(want, want) for _, want in cases]

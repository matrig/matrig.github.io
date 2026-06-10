#!/usr/bin/env python3
"""Build the site as GitHub Pages would, check the output, show it in Firefox.

Usage: python check_build.py [--no-build] [--no-view] [--port N]

Builds with the containerized Jekyll (podman), then verifies _site/:
every YAML entry rendered, no Liquid leftovers or empty links, valid
list markup, static files passed through verbatim. Then serves _site/
(with caching disabled) and opens it in Firefox; Ctrl-C stops the
server. Exits non-zero if any check failed. --no-view skips the
preview, e.g. for scripted use.
"""
import argparse
import functools
import re
import subprocess
import sys
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "_site"
SECTIONS = ["articles", "abstracts", "conferences", "patents"]
PASSTHROUGH = ["CNAME", "robots.txt", "sitemap.xml", "llms.txt"]

failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def build():
    cmd = [
        "podman", "run", "--rm", "--userns=keep-id", "-e", "HOME=/tmp",
        "--entrypoint", "", "-v", f"{ROOT}:/srv/jekyll:z", "-w", "/srv/jekyll",
        "docker.io/jekyll/jekyll:4", "/usr/gem/bin/jekyll", "build",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        sys.exit("jekyll build failed")
    print("ok    jekyll build")


class ListValidator(HTMLParser):
    """Flags non-li children of ul and checks tag balance."""
    VOID = {"br", "img", "meta", "link", "input", "hr", "source", "wbr"}

    def __init__(self):
        super().__init__()
        self.stack, self.bad = [], []

    def handle_starttag(self, tag, attrs):
        if self.stack and self.stack[-1] == "ul" and tag not in ("li", "script", "template"):
            self.bad.append(f"line {self.getpos()[0]}: <{tag}> in <ul>")
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.bad.append(f"line {self.getpos()[0]}: unbalanced </{tag}>")


class NoCacheHandler(SimpleHTTPRequestHandler):
    """Forbid caching so the browser never shows a stale build."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def view(port):
    handler = functools.partial(NoCacheHandler, directory=str(SITE))
    server = ThreadingHTTPServer(("localhost", port), handler)
    url = f"http://localhost:{port}/"
    try:
        subprocess.Popen(["firefox", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print(f"firefox not found, open {url} yourself")
    print(f"serving _site/ at {url} (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true", help="check the existing _site/ without rebuilding")
    ap.add_argument("--no-view", action="store_true", help="skip serving _site/ and opening Firefox")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if not args.no_build:
        build()

    index = SITE / "index.html"
    check("_site/index.html exists", index.exists())
    if not index.exists():
        sys.exit(1)
    html = index.read_text()

    check("front matter stripped", not html.startswith("---"))
    check("no Liquid leftovers", not re.search(r"{[{%]", html))
    check("no empty hrefs", 'href=""' not in html)
    check("footer date rendered", re.search(r"last modified: \d{1,2} \w+ \d{4}", html))

    # every YAML entry must appear as one <li> in its section
    for name in SECTIONS:
        data = yaml.safe_load((ROOT / "_data" / f"{name}.yaml").read_text())
        m = re.search(rf'<div id="show_{name}">(.*?)</div>', html, re.S)
        rendered = len(re.findall(r"<li>", m.group(1))) if m else -1
        check(f"{name}: {len(data)} entries rendered", rendered == len(data),
              f"found {rendered}")

    v = ListValidator()
    v.feed(html)
    check("valid list markup", not v.bad, "; ".join(v.bad[:5]))
    check("tags balanced", not v.stack, f"unclosed: {v.stack[:5]}")

    for f in PASSTHROUGH + [p.relative_to(ROOT).as_posix() for p in (ROOT / "slides").glob("*.html")]:
        same = (SITE / f).exists() and (SITE / f).read_bytes() == (ROOT / f).read_bytes()
        check(f"pass-through: {f}", same)

    pdfs = list(ROOT.glob("publications/**/*.pdf"))
    missing = [p for p in pdfs if not (SITE / p.relative_to(ROOT)).exists()]
    check(f"{len(pdfs)} PDFs copied", not missing, f"missing {len(missing)}")

    internal = re.findall(r'(?:href|src)="(?!https?:|mailto:|#|/)([^"]+)"', html)
    broken = sorted({u for u in internal if not (SITE / u.split("#")[0]).exists()})
    check(f"{len(internal)} internal links resolve", not broken, f"broken: {broken[:5]}")

    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout
    check("no tooling files tracked", not re.search(r"(?im)^claude", tracked))

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    else:
        print("all checks passed")
    if not args.no_view:
        view(args.port)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

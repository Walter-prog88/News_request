#!/usr/bin/env python3
"""Check static lesson links and, when available, JavaScript syntax."""
import argparse
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.handlers = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "id" and value:
                self.ids.append(value)
            if key in ("href", "src") and value:
                self.links.append(value)
            if key.startswith("on") and value:
                self.handlers.append(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default=shutil.which("node"))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    pages = {}
    errors = []
    for path in sorted(root.rglob("*.html")):
        page = Page()
        source = path.read_text(encoding="utf-8")
        page.feed(source)
        pages[path] = (page, source)
        duplicate_ids = [x for x, n in Counter(page.ids).items() if n > 1]
        if duplicate_ids:
            errors.append(f"{path.relative_to(root)}: duplicate ids {duplicate_ids}")
    scripts_checked = 0
    handlers_checked = 0
    for path, (page, source) in pages.items():
        label = str(path.relative_to(root))
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not target.exists():
                errors.append(f"{label}: missing target {link}")
            elif url.fragment and target in pages:
                if unquote(url.fragment) not in pages[target][0].ids:
                    errors.append(f"{label}: missing anchor {link}")
        if not args.node:
            continue
        scripts = [m.group(2) for m in re.finditer(r"<script\b([^>]*)>(.*?)</script>", source, re.I | re.S)
                   if not re.search(r"\bsrc\s*=", m.group(1), re.I)
                   and not re.search(r"application/(?:ld\+)?json", m.group(1), re.I)]
        # Event handlers execute in a function body and may contain return.
        scripts += ["function eventHandler(event){\n" + h + "\n}" for h in page.handlers]
        for index, script in enumerate(scripts):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".js", encoding="utf-8") as tmp:
                tmp.write(script)
                tmp.flush()
                result = subprocess.run([args.node, "--check", tmp.name], text=True, capture_output=True)
            if result.returncode:
                errors.append(f"{label}: script {index + 1}: {result.stderr.strip()}")
        scripts_checked += len(scripts) - len(page.handlers)
        handlers_checked += len(page.handlers)
    print(json.dumps({"html_pages": len(pages), "inline_scripts_checked": scripts_checked,
                      "event_handlers_checked": handlers_checked,
                      "javascript_syntax": "checked" if args.node else "skipped: Node.js unavailable",
                      "errors": errors}, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())

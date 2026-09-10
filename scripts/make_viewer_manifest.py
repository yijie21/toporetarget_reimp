#!/usr/bin/env python3
"""Regenerate a viewer directory's manifest.js from the payloads present.

The viewer loads whatever this lists, so it must describe files that actually
exist — otherwise the page requests a payload that was never committed (dataset
derived ones are git-ignored).

    python scripts/make_viewer_manifest.py [viewer_dir]
"""
import json
import sys
from pathlib import Path


def build(directory) -> list:
    d = Path(directory)
    entries = [dict(key=p.stem[len("data_"):],
                    var="TR_" + p.stem[len("data_"):].upper(),
                    file=p.name,
                    label=p.stem[len("data_"):].replace("_", " "))
               for p in sorted(d.glob("data_*.js"))]
    (d / "manifest.js").write_text("window.TR_MANIFEST = " + json.dumps(entries) + ";")
    return entries


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "viewer"
    found = build(target)
    print(f"{target}/manifest.js: {[e['key'] for e in found] or 'no payloads'}")

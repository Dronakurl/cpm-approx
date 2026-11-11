#!/usr/bin/env python3
"""Render the formulas of README.md to PNG.

The source of every formula is the typst code in an HTML comment of the README,

    <!-- typst:NAME
    $ a^2 + b^2 = c^2 $
    -->
    ![alt text](.../docs/img/NAME.png)

Edit the comment, then run `python tools/formulas.py` (needs the `typst` command line tool). The PNGs
are written to docs/img/ and committed, so that GitHub and PyPI show them (white background, readable in
light and dark mode).
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = """#set page(width: auto, height: auto, margin: 8pt, fill: white)
#set text(size: 12pt, fill: black)
{body}
"""


def main() -> int:
    readme = (ROOT / "README.md").read_text()
    blocks = re.findall(r"<!-- typst:([A-Za-z0-9_-]+)\n(.*?)\n-->", readme, flags=re.S)
    if not blocks:
        print("no typst blocks found", file=sys.stderr)
        return 1
    out_dir = ROOT / "docs" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, body in blocks:
            src = Path(tmp) / f"{name}.typ"
            src.write_text(PAGE.format(body=body))
            target = out_dir / f"{name}.png"
            subprocess.run(["typst", "compile", "--format", "png", "--ppi", "170", str(src), str(target)], check=True)
            print(f"wrote {target.relative_to(ROOT)}")
    missing = [n for n, _ in blocks if f"docs/img/{n}.png" not in readme]
    if missing:
        print(f"warning: no image reference for {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

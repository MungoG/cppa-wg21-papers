#!/usr/bin/env python3
"""Assemble A Young Delegate's Notebook from its chapter files into one manuscript.

Reads the chapters in fixed order, generates a table of contents from the
chapter and section headings, and writes young-delegates-notebook.md. Run from
the book root:

    python build.py

Missing chapters are skipped with a warning, so the build works while the book
is still being written.
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHAPTERS = ROOT / "chapters"
OUTPUT = ROOT / "young-delegates-notebook.md"

TITLE = "A Young Delegate's Notebook"
SUBTITLE = "A Newcomer's Path Into WG21 and the Standardization of C++"
BLURB = (
    "A practical guide for anyone who wants to help shape C++ but has never "
    "set foot in a committee meeting. It accumulates: each chapter stands on "
    "the ones before it, and you can stop at any chapter and still have "
    "something useful to offer. Every paper number is a live link, and every "
    "named resource points somewhere real."
)

ORDER = [
    "intro.md",
    "ch-01.md",
    "ch-02.md",
    "ch-03.md",
    "ch-04.md",
    "ch-05.md",
    "ch-06.md",
    "ch-07.md",
    "ch-08.md",
    "ch-09.md",
    "ch-10.md",
    "ch-11.md",
    "ch-12.md",
    "ch-13.md",
]


def build() -> None:
    parts: list[str] = []
    toc: list[str] = ["## Contents", ""]
    word_count = 0
    found = 0

    for name in ORDER:
        path = CHAPTERS / name
        if not path.exists():
            print(f"warning: missing chapter {name}, skipping")
            continue
        found += 1
        text = path.read_text(encoding="utf-8").strip()
        parts.append(text)
        word_count += len(text.split())
        for line in text.splitlines():
            m = re.match(r"^(#{2,3})\s+(.*)$", line)
            if not m:
                continue
            level = len(m.group(1))
            heading = m.group(2).strip()
            indent = "  " * (level - 2)
            toc.append(f"{indent}- {heading}")

    stamp = datetime.date.today().isoformat()
    header = [
        f"# {TITLE}",
        "",
        f"*{SUBTITLE}*",
        "",
        BLURB,
        "",
        f"*Assembled {stamp}*",
        "",
        "\n".join(toc),
        "",
        "---",
        "",
    ]
    body = "\n\n---\n\n".join(parts)
    OUTPUT.write_text("\n".join(header) + "\n" + body + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.name}: {word_count} words across {found} chapters")


if __name__ == "__main__":
    build()

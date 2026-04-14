"""HTML to Markdown converter for WG21 papers."""

import logging
from pathlib import Path

from .. import ascii_escape
from . import extract as _extract
from . import render as _render

_log = logging.getLogger(__name__)


def convert_html(path: Path) -> tuple[str, str | None]:
    """Convert an HTML file to Markdown.

    Returns (markdown_text, prompts_text_or_none).
    HTML conversion produces a prompts file only when sections
    cannot be converted cleanly.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    soup = _extract.parse_html(text)
    generator = _extract.detect_generator(soup)
    _log.debug("Generator: %s", generator)

    metadata = _extract.extract_metadata(soup, generator)
    problems = _extract.strip_boilerplate(soup, generator)

    body_md = _render.render_body(soup, generator)

    parts = []
    if metadata:
        fm_lines = ["---"]
        order = ["title", "document", "date", "audience", "reply-to"]
        for key in order:
            if key in metadata:
                val = metadata[key]
                if isinstance(val, list):
                    items = [f'  - "{v}"' for v in val]
                    fm_lines.append(f"{key}:\n" + "\n".join(items))
                elif key == "title":
                    fm_lines.append(f'{key}: "{val}"')
                else:
                    fm_lines.append(f"{key}: {val}")
        for key, val in metadata.items():
            if key not in order:
                fm_lines.append(f"{key}: {val}")
        fm_lines.append("---")
        parts.append("\n".join(fm_lines))

    if body_md.strip():
        parts.append(body_md.strip())

    md = "\n\n".join(parts)
    md = md.rstrip() + "\n"

    prompts = None
    if problems:
        prompt_parts = [
            "# tomd - HTML Conversion Issues",
            "",
            "The following issues were encountered during HTML-to-Markdown conversion.",
            "",
        ]
        for i, problem in enumerate(problems, 1):
            prompt_parts.append(f"## Issue {i}")
            prompt_parts.append("")
            prompt_parts.append(problem)
            prompt_parts.append("")
        prompts = "\n".join(prompt_parts)

    return ascii_escape(md), prompts

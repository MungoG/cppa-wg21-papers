"""Scrivener internals. See scrivener.py for the CLI entry point."""

from . import inline_patch  # noqa: F401 - patches ReportLab on import


def escape_xml(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

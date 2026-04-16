"""Shared test helpers for cite tests."""


def _lines(text):
    """Split text into lines with newline terminators."""
    return [line + '\n' for line in text.strip().split('\n')]

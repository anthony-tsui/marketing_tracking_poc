"""Martech GA4/GTM RAG package."""

from __future__ import annotations

import os

__version__ = "0.1.0"


def _clear_unwritable_sslkeylogfile() -> None:
    """Drop SSLKEYLOGFILE when Python cannot write it.

    Cursor and some IDEs set SSLKEYLOGFILE to a virtual volume path.
    ssl.create_default_context then raises PermissionError, which breaks
    Gradio and httpx imports on Windows.
    """
    path = os.environ.get("SSLKEYLOGFILE")
    if not path:
        return
    try:
        with open(path, "a"):
            pass
    except OSError:
        os.environ.pop("SSLKEYLOGFILE", None)


_clear_unwritable_sslkeylogfile()

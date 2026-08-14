"""Launch the Gradio Chat + Eval app."""

from __future__ import annotations

import logging

from martech_rag.apps.gradio_app import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

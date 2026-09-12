"""Notebook logger: every executed block becomes a cell of ``analysis.ipynb`` (STAT 0.1 format
plus markdown handoff cells)."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class NotebookLogger:
    def __init__(self, session_dir: str | Path | None = None, filename: str = "analysis.ipynb") -> None:
        self.session_dir = Path(session_dir) if session_dir else None
        self.notebook_file = self.session_dir / filename if self.session_dir else None
        self.cells: list[dict] = []
        self.initialized = False

    @staticmethod
    def _source(text: str) -> list[str]:
        lines = text.split("\n")
        return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]

    def initialize_notebook(self, dataset_dir: str | None = None, session_name: str = "session") -> None:
        bootstrap = (
            "# STAT analysis notebook - generated automatically\n"
            "from stat_agent.core import SimpleSession\n"
            "import numpy as np\nimport pandas as pd\n\n"
            f"session = SimpleSession(name={session_name!r})\n"
            + (f"session.load_dataset({dataset_dir!r})\n" if dataset_dir else "")
        )
        self.cells = [
            {"cell_type": "markdown", "metadata": {}, "source": self._source(f"# STAT session `{session_name}`\n\nCreated {datetime.now():%Y-%m-%d %H:%M:%S}.")},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": self._source(bootstrap)},
        ]
        self.initialized = True
        self.save()

    def append_markdown(self, text: str) -> None:
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": self._source(text)})
        self.save()

    def append_roi_creation(self, name: str, slice_id: int, definition: dict) -> None:
        code = f"# ROI created in the viewer\nsession.create_roi({name!r}, slice_id={slice_id}, roi_definition={json.dumps(definition)})\n"
        self.cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": self._source(code)})
        self.save()

    def append_code_execution(self, code: str, result: Any, user_query: str | None = None, figures_png: list[bytes] | None = None) -> None:
        header = f"# User: {user_query}\n" if user_query else ""
        if getattr(result, "success", True) is False:
            header += f"# NOTE: this block failed ({getattr(result, 'error_type', 'error')})\n"
        outputs: list[dict] = []
        stdout = getattr(result, "stdout", "") or ""
        if stdout:
            outputs.append({"output_type": "stream", "name": "stdout", "text": self._source(stdout)})
        stderr = getattr(result, "stderr", "") or ""
        if stderr.strip():
            outputs.append({"output_type": "stream", "name": "stderr", "text": self._source(stderr)})
        for png in figures_png or []:
            outputs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(png).decode("ascii")}})
        duration = getattr(result, "duration_s", None)
        if duration is not None:
            outputs.append({"output_type": "stream", "name": "stdout", "text": [f"Execution time: {duration:.2f}s"]})
        self.cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": outputs, "source": self._source(header + code)})
        self.save()

    def save(self) -> None:
        if not self.notebook_file:
            return
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "cells": self.cells,
        }
        self.notebook_file.parent.mkdir(parents=True, exist_ok=True)
        self.notebook_file.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


__all__ = ["NotebookLogger"]

"""Lightweight experiment tracking helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JSONLTracker:
    """Append-only JSONL logger for training/evaluation metrics."""

    output_path: Path
    _buffer: list[dict[str, Any]] = field(default_factory=list)

    def log(self, payload: dict[str, Any]) -> None:
        self._buffer.append(payload)

    def flush(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            for row in self._buffer:
                handle.write(json.dumps(row) + "\n")
        self._buffer.clear()


def save_artifact_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write artifact metadata as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

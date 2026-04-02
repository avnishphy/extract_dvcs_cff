"""
Path resolution helpers for extract_dvcs_cff config system.
"""
from pathlib import Path
from typing import Optional

def resolve_path(path: Path, base: Optional[Path] = None) -> Path:
    """Resolve a path relative to a base directory if not absolute."""
    if path.is_absolute():
        return path
    if base is not None:
        return (base / path).resolve()
    return path.resolve()

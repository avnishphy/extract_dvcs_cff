"""
Path helpers for extract_dvcs_cff.
"""
from pathlib import Path
from typing import Optional

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# Add more helpers as needed

"""
Serialization helpers for extract_dvcs_cff.
"""
import json
import yaml
from typing import Any

def save_json(obj: Any, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_yaml(obj: Any, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

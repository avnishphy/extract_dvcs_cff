"""
extract_dvcs_cff.config package init.
"""
# Explicit imports for config submodules
from .defaults import MainConfig, get_default_config
from .paths import resolve_path

__all__ = ["MainConfig", "get_default_config", "resolve_path"]

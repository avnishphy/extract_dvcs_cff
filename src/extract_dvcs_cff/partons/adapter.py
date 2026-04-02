"""
Adapter for PARTONS external physics code.
"""
from typing import Any, Dict
import logging

class PartonsAdapter:
    """Adapter for PARTONS backend."""
    def __init__(self, partons_path: str = "partons"):
        self.partons_path = partons_path
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        # Placeholder: check if PARTONS is available
        return False

    def compute_cffs(self, kinematics: Dict[str, Any], model: str) -> Dict[str, float]:
        if not self.available:
            logging.warning("PARTONS backend unavailable.")
            raise RuntimeError("PARTONS not available.")
        # Placeholder: call PARTONS and return CFFs
        raise NotImplementedError

    def compute_observables(self, kinematics: Dict[str, Any], model: str) -> Dict[str, float]:
        if not self.available:
            logging.warning("PARTONS backend unavailable.")
            raise RuntimeError("PARTONS not available.")
        # Placeholder: call PARTONS and return observables
        raise NotImplementedError

    def generate_pseudodata(self, kinematics_list, model: str, seed: int = 42):
        if not self.available:
            logging.warning("PARTONS backend unavailable.")
            raise RuntimeError("PARTONS not available.")
        # Placeholder: generate pseudodata
        raise NotImplementedError

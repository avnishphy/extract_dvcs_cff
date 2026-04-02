"""
Adapter for APFEL external physics code.
"""
from typing import Any, Dict
import logging

class ApfelAdapter:
    """Adapter for APFEL backend."""
    def __init__(self, apfel_path: str = "apfel"):
        self.apfel_path = apfel_path
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        # Placeholder: check if APFEL is available
        return False

    def evolution_check(self, gpd: Dict[str, float], Q2: float) -> bool:
        if not self.available:
            logging.warning("APFEL backend unavailable.")
            return False
        # Placeholder: implement evolution check
        raise NotImplementedError

    def forward_limit_check(self, gpd: Dict[str, float], pdf: Dict[str, float]) -> bool:
        if not self.available:
            logging.warning("APFEL backend unavailable.")
            return False
        # Placeholder: implement forward limit check
        raise NotImplementedError

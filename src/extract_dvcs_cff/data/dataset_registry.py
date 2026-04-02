"""
Registry for known datasets and metadata.
"""
from typing import Dict
from .schemas import DatasetRecord

class DatasetRegistry:
    """Registry for tracking loaded datasets and metadata."""
    def __init__(self):
        self._datasets: Dict[str, DatasetRecord] = {}

    def register(self, record: DatasetRecord):
        self._datasets[record.dataset_id] = record

    def get(self, dataset_id: str) -> DatasetRecord:
        return self._datasets[dataset_id]

    def all(self):
        return list(self._datasets.values())

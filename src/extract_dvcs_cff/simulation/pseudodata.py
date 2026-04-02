"""
Pseudodata generation for closure tests and benchmarking.
"""
from typing import List, Dict, Any
import numpy as np
import logging

from extract_dvcs_cff.partons.adapter import PartonsAdapter


def generate_pseudodata(config):
    """Generate pseudodata for all kinematic points using a benchmark model."""
    # Placeholder: load kinematics from config or dataset
    kinematics_list = []  # Should be populated from dataset
    model = config.benchmark.model_name
    seed = config.benchmark.seed
    np.random.seed(seed)
    partons = PartonsAdapter()
    if not partons.available:
        logging.error("PARTONS backend unavailable. Cannot generate pseudodata.")
        return
    # Placeholder: call PARTONS to generate pseudodata
    # Save manifest with model, seed, etc.
    raise NotImplementedError

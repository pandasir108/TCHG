from __future__ import annotations

from copy import deepcopy
from typing import Dict


def ablation_configs(base_cfg: Dict) -> Dict[str, Dict]:
    variants: Dict[str, Dict] = {}

    def add(name: str, model_updates: dict, training_updates: dict | None = None) -> None:
        cfg = deepcopy(base_cfg)
        cfg["model"].update(model_updates)
        if training_updates:
            cfg["training"].update(training_updates)
        variants[name] = cfg

    add("Full TCHG", {"name": "tchgt"})
    add("w/o Entity Admission Gate", {"name": "tchgt", "use_entity_gate": False})
    add("w/o Behavior Reliability Modulation", {"name": "tchgt", "use_behavior_modulation": False})
    add("w/o Context Operator Selection", {"name": "tchgt", "use_context_operator": False})
    add("w/o Component-wise Memory", {"name": "tchgt", "component_wise_memory": False})
    add("w/o Masked Memory", {"name": "tchgt", "use_masked_memory": False})
    add(
        "w/o Uncertainty Calibration",
        {"name": "tchgt", "use_uncertainty_calibration": False},
        {"lambda_brier": 0.0, "lambda_unc": 0.0, "lambda_uerr": 0.0},
    )
    add(
        "Uniform Decay Memory",
        {
            "name": "tchgt",
            "component_wise_memory": True,
            "memory_lambdas": [0.045, 0.045, 0.045],
            "uncertainty_etas": [0.060, 0.060, 0.060],
        },
    )
    return variants

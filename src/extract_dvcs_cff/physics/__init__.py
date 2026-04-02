"""Physics package exports with lazy loading."""

from __future__ import annotations

import importlib

__all__ = [
    "ConvolutionResult",
    "DifferentiableCFFConvolution",
    "DipoleFormFactorProvider",
    "NullPDFProvider",
    "PhysicsConstraintEvaluator",
    "TabulatedFormFactorProvider",
    "mellin_moment",
    "polynomiality_penalty",
    "positivity_penalty",
    "Q2EvolutionLayer",
    "GaussianLikelihood",
    "compute_likelihood",
    "TorchDVCSObservableLayer",
    "ToyDVCSObservableCalculator",
    "map_observable_label",
    "observable_name_to_index",
]


_MODULE_FOR_SYMBOL = {
    "ConvolutionResult": ".cff_convolution",
    "DifferentiableCFFConvolution": ".cff_convolution",
    "DipoleFormFactorProvider": ".constraints",
    "NullPDFProvider": ".constraints",
    "PhysicsConstraintEvaluator": ".constraints",
    "TabulatedFormFactorProvider": ".constraints",
    "mellin_moment": ".constraints",
    "polynomiality_penalty": ".constraints",
    "positivity_penalty": ".constraints",
    "Q2EvolutionLayer": ".evolution",
    "GaussianLikelihood": ".likelihood",
    "compute_likelihood": ".likelihood",
    "TorchDVCSObservableLayer": ".observables",
    "ToyDVCSObservableCalculator": ".observables",
    "map_observable_label": ".observables",
    "observable_name_to_index": ".observables",
}


def __getattr__(name: str):
    module_name = _MODULE_FOR_SYMBOL.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)

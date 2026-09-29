"""HyperReservoir kernel (arXiv:2609.34847)."""

from hyperreservoir.conceptor import apply_conceptor, conceptor_matrix, state_correlation
from hyperreservoir.esn import (
    ContextInputESN,
    init_input_weights,
    init_recurrent,
    leaky_esn_step,
    run_leaky_esn,
    scale_spectral_radius,
)
from hyperreservoir.hyper import (
    HyperReservoir,
    effective_readout,
    predict_augmented,
    predict_block,
    psi_aug,
)
from hyperreservoir.ridge import ridge_fit, ridge_predict
from hyperreservoir.systems import integrate, lorenz_rhs, rk4_step, rossler_rhs

__all__ = [
    "ContextInputESN",
    "HyperReservoir",
    "apply_conceptor",
    "conceptor_matrix",
    "effective_readout",
    "init_input_weights",
    "init_recurrent",
    "integrate",
    "leaky_esn_step",
    "lorenz_rhs",
    "predict_augmented",
    "predict_block",
    "psi_aug",
    "ridge_fit",
    "ridge_predict",
    "rk4_step",
    "rossler_rhs",
    "run_leaky_esn",
    "scale_spectral_radius",
    "state_correlation",
]

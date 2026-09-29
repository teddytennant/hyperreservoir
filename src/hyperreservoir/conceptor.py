"""Full-matrix Conceptor (paper eqs. 4–7 / S9–S10)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from hyperreservoir.esn import leaky_esn_step


def state_correlation(X: Array) -> Array:
    """Uncentered empirical correlation, eqs. (4) and (S9):

    R_k = (1/n_k) sum_{i in I_k} x_i x_i^T
    """
    n_k = X.shape[0]
    return (X.T @ X) / n_k


def conceptor_matrix(R: Array, gamma: float) -> Array:
    """Full-matrix Conceptor, eqs. (5) and (S10):

    C_k = R_k (R_k + γ^{-2} I)^{-1}
    """
    n = R.shape[0]
    eye = jnp.eye(n, dtype=R.dtype)
    return R @ jnp.linalg.inv(R + (gamma ** -2) * eye)


def apply_conceptor(C: Array, x_provisional: Array) -> Array:
    """Filter the provisional state, eq. (6): x_{t+1} = C_k x_{t+1}^{provisional}."""
    return C @ x_provisional


def conceptor_step(
    x: Array,
    u: Array,
    J: Array,
    W_in: Array,
    alpha: float,
    C: Array,
) -> Array:
    """Provisional leaky-ESN step (eq. 2) then Conceptor filter (eq. 6)."""
    x_prov = leaky_esn_step(x, u, J, W_in, alpha)
    return apply_conceptor(C, x_prov)

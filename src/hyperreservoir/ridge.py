"""Ridge regression with unregularized bias (paper eqs. S7–S8)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def ridge_fit(X: Array, Y: Array, lam: float) -> tuple[Array, Array]:
    """Fit ŷ = W x + b by ridge regression.

    X: (T, F) readout features, Y: (T, D) or (T,) targets.
    An unregularized constant column is appended (S7). Coefficients (S8):

        W_aug = (X_aug^T X_aug + λ R)^{-1} X_aug^T Y

    R is identity except the bias entry is zero, so the output bias is not
    regularized. Returns W of shape (D, F) and b of shape (D,) matching eq. (3):
    ŷ_t = W_out x_t + b_out.
    """
    Y = jnp.atleast_2d(Y)
    if Y.shape[0] == 1 and X.shape[0] != 1:
        Y = Y.T
    t, f = X.shape
    ones = jnp.ones((t, 1), dtype=X.dtype)
    X_aug = jnp.concatenate([X, ones], axis=1)
    xtx = X_aug.T @ X_aug
    r = jnp.eye(f + 1, dtype=X.dtype).at[-1, -1].set(0.0)
    w_aug = jnp.linalg.solve(xtx + lam * r, X_aug.T @ Y)
    W = w_aug[:-1, :].T
    b = w_aug[-1, :]
    return W, b


def ridge_predict(X: Array, W: Array, b: Array) -> Array:
    """ŷ = X W^T + b, i.e. ŷ_t = W x_t + b_out."""
    return X @ W.T + b

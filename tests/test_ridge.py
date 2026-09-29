"""Ridge regression recovers a known linear map (eqs. S7–S8 / eq. 3)."""

import jax.numpy as jnp

from hyperreservoir.ridge import ridge_fit, ridge_predict


def test_ridge_recovers_known_weights_at_lambda_zero():
    X = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 0.5],
        ]
    )
    W_true = jnp.array(
        [
            [2.0, -3.0],
            [0.5, 4.0],
        ]
    )
    b_true = jnp.array([1.25, -0.75])
    Y = X @ W_true.T + b_true
    W, b = ridge_fit(X, Y, lam=0.0)
    assert jnp.allclose(W, W_true, atol=1e-4)
    assert jnp.allclose(b, b_true, atol=1e-4)
    pred = ridge_predict(X, W, b)
    assert jnp.allclose(pred, Y, atol=1e-4)

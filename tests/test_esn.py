"""Tests for leaky ESN algebra (eq. 2) and the context-input baseline."""

import jax
import jax.numpy as jnp

from hyperreservoir.esn import ContextInputESN, leaky_esn_step, phi, scale_spectral_radius


def test_leaky_esn_step_matches_eq2():
    x = jnp.array([0.2, -0.4, 0.1, 0.5])
    u = jnp.array([0.3, -0.7])
    J = jnp.array(
        [
            [0.2, -0.1, 0.0, 0.3],
            [0.1, 0.4, -0.2, 0.0],
            [0.0, 0.2, -0.3, 0.1],
            [-0.1, 0.0, 0.2, 0.25],
        ]
    )
    W_in = jnp.array(
        [
            [0.5, -0.2],
            [0.1, 0.4],
            [-0.3, 0.2],
            [0.0, 0.6],
        ]
    )
    alpha = 0.3
    got = leaky_esn_step(x, u, J, W_in, alpha)
    expected = x + alpha * (-x + J @ phi(x) + W_in @ u)
    assert jnp.allclose(got, expected, atol=1e-6)


def test_echo_state_contracts_toward_zero():
    n = 6
    key = jax.random.PRNGKey(0)
    J = jax.random.normal(key, (n, n))
    J = scale_spectral_radius(J, rho=0.7)
    W_in = jnp.zeros((n, 1))
    u = jnp.zeros((1,))
    alpha = 1.0
    x = jax.random.normal(jax.random.PRNGKey(1), (n,))
    n0 = jnp.linalg.norm(x)
    for _ in range(40):
        x = leaky_esn_step(x, u, J, W_in, alpha)
    assert jnp.linalg.norm(x) < 0.05 * n0
    assert jnp.linalg.norm(x) < 1e-3


def test_context_input_esn_input_dim():
    obs_dim, context_dim, n = 3, 2, 8
    esn = ContextInputESN.init(
        jax.random.PRNGKey(2), n=n, obs_dim=obs_dim, context_dim=context_dim
    )
    assert esn.input_dim == obs_dim + context_dim
    assert esn.W_in.shape == (n, obs_dim + context_dim)
    u = jnp.ones((obs_dim,))
    c = jnp.array([1.0, 0.0])
    x = jnp.zeros((n,))
    x1 = esn.step(x, u, c)
    assert x1.shape == (n,)
    concat = esn.concat_input(u, c)
    assert concat.shape == (obs_dim + context_dim,)
    assert jnp.allclose(concat[:obs_dim], u)
    assert jnp.allclose(concat[obs_dim:], c)

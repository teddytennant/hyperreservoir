"""Tests for HyperReservoir updates and bilinear readout (eqs. 8–14)."""

import jax
import jax.numpy as jnp

from hyperreservoir.esn import leaky_esn_step
from hyperreservoir.hyper import (
    HyperReservoir,
    effective_readout,
    predict_augmented,
    predict_block,
    psi_aug,
)


def test_hyperreservoir_steps_match_eqs_8_9_independently():
    key = jax.random.PRNGKey(3)
    n, m, obs_dim, ctx_dim = 5, 3, 2, 2
    hr = HyperReservoir.init(key, n=n, m=m, obs_dim=obs_dim, context_dim=ctx_dim)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(4), 4)
    h_r = jax.random.normal(k1, (n,))
    h_h = jax.random.normal(k2, (m,))
    u = jax.random.normal(k3, (obs_dim,))
    c = jax.random.normal(k4, (ctx_dim,))

    h_r_next, h_h_next = hr.step(h_r, h_h, u, c)
    h_r_eq8 = leaky_esn_step(h_r, u, hr.J_r, hr.W_in_r, hr.alpha_r)
    h_h_eq9 = leaky_esn_step(h_h, c, hr.J_h, hr.W_in_h, hr.alpha_h)
    assert jnp.allclose(h_r_next, h_r_eq8, atol=1e-6)
    assert jnp.allclose(h_h_next, h_h_eq9, atol=1e-6)

    u2 = u + 10.0
    c2 = c + 10.0
    h_r_u2, h_h_u2 = hr.step(h_r, h_h, u2, c)
    h_r_c2, h_h_c2 = hr.step(h_r, h_h, u, c2)
    assert jnp.allclose(h_h_u2, h_h_next, atol=1e-6)
    assert jnp.allclose(h_r_c2, h_r_next, atol=1e-6)
    assert not jnp.allclose(h_r_u2, h_r_next, atol=1e-5)
    assert not jnp.allclose(h_h_c2, h_h_next, atol=1e-5)


def test_psi_aug_concat_and_kronecker():
    n, m = 4, 3
    h_r = jnp.arange(n, dtype=jnp.float32) + 0.5
    h_h = jnp.arange(m, dtype=jnp.float32) + 1.25
    psi = psi_aug(h_r, h_h)
    assert psi.shape == (n + m + m * n,)
    assert jnp.allclose(psi[:n], h_r)
    assert jnp.allclose(psi[n : n + m], h_h)
    kron = jnp.kron(h_h, h_r)
    outer_flat = jnp.outer(h_h, h_r).reshape(-1)
    assert jnp.allclose(psi[n + m :], kron)
    assert jnp.allclose(kron, outer_flat)


def test_eq11_equals_eq13():
    key = jax.random.PRNGKey(5)
    n, m, d_out = 6, 4, 3
    k = jax.random.split(key, 6)
    h_r = jax.random.normal(k[0], (n,))
    h_h = jax.random.normal(k[1], (m,))
    W_r = jax.random.normal(k[2], (d_out, n))
    W_h = jax.random.normal(k[3], (d_out, m))
    W_rh = jax.random.normal(k[4], (d_out, m * n))
    b = jax.random.normal(k[5], (d_out,))
    y11 = predict_augmented(h_r, h_h, W_r, W_h, W_rh, b)
    y13 = predict_block(h_r, h_h, W_r, W_h, W_rh, b)
    assert jnp.allclose(y11, y13, atol=1e-5)


def test_weff_formula_matches_eq11():
    key = jax.random.PRNGKey(6)
    n, m, d_out = 5, 3, 2
    k = jax.random.split(key, 6)
    h_r = jax.random.normal(k[0], (n,))
    h_h = jax.random.normal(k[1], (m,))
    W_r = jax.random.normal(k[2], (d_out, n))
    W_h = jax.random.normal(k[3], (d_out, m))
    W_rh = jax.random.normal(k[4], (d_out, m * n))
    b = jax.random.normal(k[5], (d_out,))
    y11 = predict_augmented(h_r, h_h, W_r, W_h, W_rh, b)
    W_eff = effective_readout(W_r, W_rh, h_h)
    y14 = W_eff @ h_r + W_h @ h_h + b
    assert jnp.allclose(y11, y14, atol=1e-5)

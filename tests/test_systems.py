"""Lorenz / Rössler integrators and temporal-scale factor ν (eqs. 15–25)."""

import jax.numpy as jnp

from hyperreservoir.systems import NU_FAST, NU_SLOW, integrate, lorenz_rhs, rossler_rhs


def _finite_and_speed(rhs, x0):
    dt = 0.01
    n_steps = 40
    slow = integrate(rhs, x0, dt, n_steps, nu=NU_SLOW)
    fast = integrate(rhs, x0, dt, n_steps, nu=NU_FAST)
    assert jnp.all(jnp.isfinite(slow))
    assert jnp.all(jnp.isfinite(fast))
    disp_slow = jnp.linalg.norm(slow[-1] - x0)
    disp_fast = jnp.linalg.norm(fast[-1] - x0)
    assert disp_fast > disp_slow


def test_lorenz_trajectory_finite_and_nu_changes_speed():
    x0 = jnp.array([1.0, 1.0, 1.0])
    _finite_and_speed(lorenz_rhs, x0)


def test_rossler_trajectory_finite_and_nu_changes_speed():
    x0 = jnp.array([1.0, 1.0, 1.0])
    _finite_and_speed(rossler_rhs, x0)

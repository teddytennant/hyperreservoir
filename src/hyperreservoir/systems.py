"""Lorenz and Rössler vector fields (Sec. III) and a tiny RK4 integrator."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

# Lorenz parameters, eq. (18)
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

# Rössler parameters, eq. (22)
A_R = 0.2
B_R = 0.2
C_R = 5.7

# Temporal scales, eq. (25)
NU_SLOW = 0.5
NU_FAST = 1.5


def lorenz_rhs(
    state: Array,
    sigma: float = SIGMA,
    rho: float = RHO,
    beta: float = BETA,
    nu: float = 1.0,
) -> Array:
    """Lorenz vector field, eqs. (15)–(17), optionally scaled by ν as in eq. (24)."""
    chi, psi, omega = state
    dchi = sigma * (psi - chi)
    dpsi = chi * (rho - omega) - psi
    domega = chi * psi - beta * omega
    return nu * jnp.stack([dchi, dpsi, domega])


def rossler_rhs(
    state: Array,
    a: float = A_R,
    b: float = B_R,
    c: float = C_R,
    nu: float = 1.0,
) -> Array:
    """Rössler vector field, eqs. (19)–(21), scaled by ν in eq. (24): φ̇ = ν_k f_R(φ)."""
    chi, psi, omega = state
    dchi = -psi - omega
    dpsi = chi + a * psi
    domega = b + omega * (chi - c)
    return nu * jnp.stack([dchi, dpsi, domega])


def rk4_step(f: Callable[[Array], Array], state: Array, dt: float) -> Array:
    """Classical fourth-order Runge–Kutta step (paper uses RK4)."""
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(
    rhs: Callable[..., Array],
    x0: Array,
    dt: float,
    n_steps: int,
    nu: float = 1.0,
    **rhs_kwargs,
) -> Array:
    """Integrate ``n_steps`` RK4 steps. Returns (n_steps + 1, dim) including x0."""

    def f(state: Array) -> Array:
        return rhs(state, nu=nu, **rhs_kwargs)

    def body(state, _):
        nxt = rk4_step(f, state, dt)
        return nxt, nxt

    _, traj = jax.lax.scan(body, x0, jnp.arange(n_steps))
    return jnp.concatenate([x0[None, :], traj], axis=0)

"""Leaky echo-state network (paper eqs. 2–3) and the context-input baseline."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from hyperreservoir.ridge import ridge_fit, ridge_predict


def phi(x: Array) -> Array:
    """Elementwise nonlinearity. Table S1: tanh."""
    return jnp.tanh(x)


def scale_spectral_radius(J: Array, rho: float) -> Array:
    """Rescale recurrent matrix so its spectral radius equals ``rho``."""
    sr = jnp.max(jnp.abs(jnp.linalg.eigvals(J)))
    return jnp.where(sr > 0, J * (rho / sr), J)


def init_recurrent(key: Array, n: int, rho: float = 0.9) -> Array:
    """Gaussian entries with std N^{-1/2}, then rescale to spectral radius ``rho`` (S1.2)."""
    J = jax.random.normal(key, (n, n)) / jnp.sqrt(n)
    return scale_spectral_radius(J, rho)


def init_input_weights(key: Array, n: int, u_dim: int, sigma_in: float) -> Array:
    """Zero-mean Gaussian with std σ_in / sqrt(U) (S1.2)."""
    return jax.random.normal(key, (n, u_dim)) * (sigma_in / jnp.sqrt(u_dim))


def leaky_esn_step(
    x: Array,
    u: Array,
    J: Array,
    W_in: Array,
    alpha: float,
) -> Array:
    """One leaky-ESN update, eq. (2):

    x_{t+1} = x_t + α [-x_t + J φ(x_t) + W_in u_t]
    """
    return x + alpha * (-x + J @ phi(x) + W_in @ u)


def run_leaky_esn(
    u_seq: Array,
    J: Array,
    W_in: Array,
    alpha: float,
    x0: Array | None = None,
) -> Array:
    """Drive the reservoir with ``u_seq`` of shape (T, U). Returns states (T, N) after each input."""
    n = J.shape[0]
    if x0 is None:
        x0 = jnp.zeros((n,), dtype=u_seq.dtype)

    def body(x, u):
        x_next = leaky_esn_step(x, u, J, W_in, alpha)
        return x_next, x_next

    _, xs = jax.lax.scan(body, x0, u_seq)
    return xs


@dataclass
class ContextInputESN:
    """Context-input ESN: u_t concatenates observed state and explicit context (Sec. II.A)."""

    J: Array
    W_in: Array
    alpha: float
    W_out: Array | None = None
    b_out: Array | None = None

    @property
    def n(self) -> int:
        return int(self.J.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.W_in.shape[1])

    @classmethod
    def init(
        cls,
        key: Array,
        n: int,
        obs_dim: int,
        context_dim: int,
        rho: float = 0.9,
        sigma_in: float = 0.3,
        alpha: float = 0.1,
    ) -> "ContextInputESN":
        k_j, k_in = jax.random.split(key)
        u_dim = obs_dim + context_dim
        return cls(
            J=init_recurrent(k_j, n, rho),
            W_in=init_input_weights(k_in, n, u_dim, sigma_in),
            alpha=alpha,
        )

    def concat_input(self, u_obs: Array, c: Array) -> Array:
        return jnp.concatenate([u_obs, c], axis=-1)

    def step(self, x: Array, u_obs: Array, c: Array) -> Array:
        return leaky_esn_step(x, self.concat_input(u_obs, c), self.J, self.W_in, self.alpha)

    def run(self, u_obs_seq: Array, c_seq: Array, x0: Array | None = None) -> Array:
        u_seq = jnp.concatenate([u_obs_seq, c_seq], axis=-1)
        return run_leaky_esn(u_seq, self.J, self.W_in, self.alpha, x0)

    def fit(self, states: Array, targets: Array, washout: int, lam: float) -> tuple[Array, Array]:
        W, b = ridge_fit(states[washout:], targets[washout:], lam)
        self.W_out, self.b_out = W, b
        return W, b

    def predict(self, states: Array) -> Array:
        if self.W_out is None or self.b_out is None:
            raise RuntimeError("readout not fitted")
        return ridge_predict(states, self.W_out, self.b_out)

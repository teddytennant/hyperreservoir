"""HyperReservoir: two ESNs and bilinear/Kronecker readout (eqs. 8–14)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from hyperreservoir.esn import init_input_weights, init_recurrent, leaky_esn_step, run_leaky_esn
from hyperreservoir.ridge import ridge_fit, ridge_predict


def psi_aug(h_r: Array, h_h: Array) -> Array:
    """Augmented readout features, eq. (10):

    ψ_t^{aug} = [ h^R_t ; h^H_t ; h^H_t ⊗ h^R_t ]
    """
    return jnp.concatenate([h_r, h_h, jnp.kron(h_h, h_r)], axis=-1)


def partition_wrh(W_rh: Array, n: int, m: int) -> Array:
    """Partition W^{RH} into M blocks B_m each D_out × N, eq. (12).

    Returns B with shape (D_out, M, N).
    """
    d_out = W_rh.shape[0]
    return W_rh.reshape(d_out, m, n)


def effective_readout(W_r: Array, W_rh: Array, h_h: Array) -> Array:
    """Context-dependent main-reservoir map, eq. (14):

    W_eff(h^H_t) = W^R + sum_m h^H_{t,m} B_m
    """
    n = W_r.shape[1]
    m = h_h.shape[-1]
    B = partition_wrh(W_rh, n, m)
    return W_r + jnp.einsum("m,dmn->dn", h_h, B)


def predict_augmented(
    h_r: Array,
    h_h: Array,
    W_r: Array,
    W_h: Array,
    W_rh: Array,
    b_out: Array,
) -> Array:
    """Linear readout on augmented features, eq. (11):

    ŷ_t = W^R h^R_t + W^H h^H_t + W^{RH} (h^H_t ⊗ h^R_t) + b_out
    """
    return W_r @ h_r + W_h @ h_h + W_rh @ jnp.kron(h_h, h_r) + b_out


def predict_block(
    h_r: Array,
    h_h: Array,
    W_r: Array,
    W_h: Array,
    W_rh: Array,
    b_out: Array,
) -> Array:
    """Block form of the same readout, eq. (13):

    ŷ_t = (W^R + sum_m h^H_{t,m} B_m) h^R_t + W^H h^H_t + b_out
    """
    W_eff = effective_readout(W_r, W_rh, h_h)
    return W_eff @ h_r + W_h @ h_h + b_out


def split_readout(W: Array, n: int, m: int) -> tuple[Array, Array, Array, Array]:
    """Split ridge weights (D, N+M+M N) plus bias into W^R, W^H, W^{RH}, b."""
    W_r = W[:, :n]
    W_h = W[:, n : n + m]
    W_rh = W[:, n + m :]
    return W_r, W_h, W_rh


@dataclass
class HyperReservoir:
    """Main reservoir driven by u_t only; context reservoir driven by c_t only."""

    J_r: Array
    W_in_r: Array
    alpha_r: float
    J_h: Array
    W_in_h: Array
    alpha_h: float
    W_r: Array | None = None
    W_h: Array | None = None
    W_rh: Array | None = None
    b_out: Array | None = None

    @property
    def n(self) -> int:
        return int(self.J_r.shape[0])

    @property
    def m(self) -> int:
        return int(self.J_h.shape[0])

    @classmethod
    def init(
        cls,
        key: Array,
        n: int,
        m: int,
        obs_dim: int,
        context_dim: int,
        rho_r: float = 0.9,
        rho_h: float = 0.9,
        sigma_in_r: float = 0.3,
        sigma_in_h: float = 0.1,
        alpha_r: float = 0.1,
        alpha_h: float = 1.0 / 60.0,
    ) -> "HyperReservoir":
        k_jr, k_inr, k_jh, k_inh = jax.random.split(key, 4)
        return cls(
            J_r=init_recurrent(k_jr, n, rho_r),
            W_in_r=init_input_weights(k_inr, n, obs_dim, sigma_in_r),
            alpha_r=alpha_r,
            J_h=init_recurrent(k_jh, m, rho_h),
            W_in_h=init_input_weights(k_inh, m, context_dim, sigma_in_h),
            alpha_h=alpha_h,
        )

    def step(
        self, h_r: Array, h_h: Array, u: Array, c: Array
    ) -> tuple[Array, Array]:
        """Independent updates, eqs. (8)–(9). Main ignores c; context ignores u."""
        h_r_next = leaky_esn_step(h_r, u, self.J_r, self.W_in_r, self.alpha_r)
        h_h_next = leaky_esn_step(h_h, c, self.J_h, self.W_in_h, self.alpha_h)
        return h_r_next, h_h_next

    def run(
        self,
        u_seq: Array,
        c_seq: Array,
        h_r0: Array | None = None,
        h_h0: Array | None = None,
    ) -> tuple[Array, Array]:
        h_r_seq = run_leaky_esn(u_seq, self.J_r, self.W_in_r, self.alpha_r, h_r0)
        h_h_seq = run_leaky_esn(c_seq, self.J_h, self.W_in_h, self.alpha_h, h_h0)
        return h_r_seq, h_h_seq

    def features(self, h_r_seq: Array, h_h_seq: Array) -> Array:
        return jax.vmap(psi_aug)(h_r_seq, h_h_seq)

    def fit(
        self,
        h_r_seq: Array,
        h_h_seq: Array,
        targets: Array,
        washout: int,
        lam: float,
    ) -> tuple[Array, Array]:
        feats = self.features(h_r_seq, h_h_seq)
        W, b = ridge_fit(feats[washout:], targets[washout:], lam)
        W_r, W_h, W_rh = split_readout(W, self.n, self.m)
        self.W_r, self.W_h, self.W_rh, self.b_out = W_r, W_h, W_rh, b
        return W, b

    def predict_from_states(self, h_r_seq: Array, h_h_seq: Array) -> Array:
        if self.W_r is None:
            raise RuntimeError("readout not fitted")
        feats = self.features(h_r_seq, h_h_seq)
        W = jnp.concatenate([self.W_r, self.W_h, self.W_rh], axis=1)
        return ridge_predict(feats, W, self.b_out)

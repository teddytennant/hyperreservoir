"""Tiny CPU smoke: HyperReservoir vs context-input ESN on two Lorenz contexts."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from hyperreservoir.esn import ContextInputESN
from hyperreservoir.hyper import HyperReservoir
from hyperreservoir.systems import NU_FAST, NU_SLOW, integrate, lorenz_rhs

OBS_DIM = 3
CTX_DIM = 2
N_MAIN = 30
M_CTX = 8
N_ESN = 30
STEPS = 200
WASHOUT = 50
DT = 0.02
LAM = 1e-4
SEED = 0


def onehot(k: int, dim: int = CTX_DIM) -> jnp.ndarray:
    return jnp.eye(dim)[k]


def make_traj(x0: jnp.ndarray, nu: float) -> jnp.ndarray:
    traj = integrate(lorenz_rhs, x0, DT, STEPS, nu=nu)
    return traj[:-1]


def mse(a: jnp.ndarray, b: jnp.ndarray) -> float:
    return float(jnp.mean((a - b) ** 2))


def collect_after_washout(seqs):
    return jnp.concatenate([s[WASHOUT:] for s in seqs], axis=0)


def main() -> None:
    key = jax.random.PRNGKey(SEED)
    k_ic, k_hr, k_esn = jax.random.split(key, 3)

    ics = jax.random.normal(k_ic, (6, 3)) * jnp.array([1.0, 1.0, 2.0]) + jnp.array(
        [0.0, 0.0, 20.0]
    )
    nus = (NU_SLOW, NU_FAST)
    train_obs, train_ctx, train_y = [], [], []
    test_obs, test_ctx, test_y = [], [], []
    for i, x0 in enumerate(ics):
        k = i % 2
        obs = make_traj(x0, nus[k])
        ctx = jnp.broadcast_to(onehot(k), (STEPS, CTX_DIM))
        y = jnp.concatenate([obs[1:], obs[-1:]], axis=0)
        if i < 4:
            train_obs.append(obs)
            train_ctx.append(ctx)
            train_y.append(y)
        else:
            test_obs.append(obs)
            test_ctx.append(ctx)
            test_y.append(y)

    hr = HyperReservoir.init(
        k_hr, n=N_MAIN, m=M_CTX, obs_dim=OBS_DIM, context_dim=CTX_DIM
    )
    hr_train_r, hr_train_h = [], []
    for obs, ctx in zip(train_obs, train_ctx):
        h_r, h_h = hr.run(obs, ctx)
        hr_train_r.append(h_r)
        hr_train_h.append(h_h)
    hr.fit(
        collect_after_washout(hr_train_r),
        collect_after_washout(hr_train_h),
        collect_after_washout(train_y),
        washout=0,
        lam=LAM,
    )
    yhat_tr_hr = hr.predict_from_states(
        collect_after_washout(hr_train_r), collect_after_washout(hr_train_h)
    )
    hr_test_r, hr_test_h = [], []
    for obs, ctx in zip(test_obs, test_ctx):
        h_r, h_h = hr.run(obs, ctx)
        hr_test_r.append(h_r)
        hr_test_h.append(h_h)
    yhat_te_hr = hr.predict_from_states(
        collect_after_washout(hr_test_r), collect_after_washout(hr_test_h)
    )

    esn = ContextInputESN.init(
        k_esn, n=N_ESN, obs_dim=OBS_DIM, context_dim=CTX_DIM
    )
    esn_train_x = [esn.run(obs, ctx) for obs, ctx in zip(train_obs, train_ctx)]
    esn.fit(
        collect_after_washout(esn_train_x),
        collect_after_washout(train_y),
        washout=0,
        lam=LAM,
    )
    yhat_tr_esn = esn.predict(collect_after_washout(esn_train_x))
    esn_test_x = [esn.run(obs, ctx) for obs, ctx in zip(test_obs, test_ctx)]
    yhat_te_esn = esn.predict(collect_after_washout(esn_test_x))

    y_tr = collect_after_washout(train_y)
    y_te = collect_after_washout(test_y)

    results = {
        "hyperreservoir": {
            "train_mse": mse(yhat_tr_hr, y_tr),
            "test_mse": mse(yhat_te_hr, y_te),
            "N": N_MAIN,
            "M": M_CTX,
        },
        "context_input_esn": {
            "train_mse": mse(yhat_tr_esn, y_tr),
            "test_mse": mse(yhat_te_esn, y_te),
            "N": N_ESN,
        },
        "steps": STEPS,
        "washout": WASHOUT,
        "lambda": LAM,
        "contexts": {"nu_slow": NU_SLOW, "nu_fast": NU_FAST},
    }
    out = Path("results.json")
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

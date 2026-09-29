"""Full-matrix Conceptor eigenvalues lie in [0, 1] (eqs. 4–5)."""

import jax
import jax.numpy as jnp

from hyperreservoir.conceptor import conceptor_matrix, state_correlation


def test_conceptor_eigenvalues_in_unit_interval():
    key = jax.random.PRNGKey(7)
    t, n = 20, 5
    X = jax.random.normal(key, (t, n))
    R = state_correlation(X)
    evals_r = jnp.linalg.eigvalsh(R)
    assert jnp.all(evals_r >= -1e-5)

    gamma = 3.0
    C = conceptor_matrix(R, gamma)
    evals = jnp.real(jnp.linalg.eigvals(C))
    assert jnp.all(evals >= -1e-5)
    assert jnp.all(evals <= 1.0 + 1e-5)

    A = jax.random.normal(jax.random.PRNGKey(8), (n, n))
    R_psd = A @ A.T
    C2 = conceptor_matrix(R_psd, gamma=1.5)
    evals2 = jnp.sort(jnp.real(jnp.linalg.eigvals(C2)))
    lam = jnp.sort(jnp.linalg.eigvalsh(R_psd))
    expected = lam / (lam + 1.5 ** -2)
    assert jnp.allclose(evals2, expected, atol=1e-5)
    assert jnp.all(evals2 >= -1e-5)
    assert jnp.all(evals2 <= 1.0 + 1e-5)

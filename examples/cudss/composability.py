"""Example: the token API composes with jit and vmap.

A batch under vmap is ONE block-diagonal cuDSS system: vmap(analyze) mints a
single registry entry and every downstream phase is one batched call.
(Nested vmap is not supported — flatten nested batches into one batch axis.)
"""
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax import cudss


def test_composability():

    M1 = jnp.array([
        [4., 0., 1., 0., 0.],
        [0., 3., 2., 0., 0.],
        [0., 0., 5., 0., 1.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 2.],
    ])
    M2 = M1 * 0.9

    b1 = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0])
    b2 = b1 * 1.1

    m1 = M1 + M1.T - jnp.diag(M1) * jnp.eye(M1.shape[0])
    m2 = M2 + M2.T - jnp.diag(M2) * jnp.eye(M2.shape[0])
    true_x1 = jnp.linalg.solve(m1, b1)
    true_x2 = jnp.linalg.solve(m2, b2)

    LHS1 = jsparse.BCSR.fromdense(M1)
    csr_offsets, csr_columns, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data
    csr_values2 = jsparse.BCSR.fromdense(M2).data

    csr_values = jnp.vstack([csr_values1, csr_values2])
    b = jnp.vstack([b1, b2])

    def token_solve(values, b):
        token = cudss.analyze(values, csr_offsets, csr_columns, mtype_id=1, mview_id=1)
        token = cudss.factorize(token, values)
        return cudss.solve(token, b)

    # single solve, eager
    x1 = token_solve(csr_values[0], b[0])

    # jit + vmap: one block-diagonal factorization + one block solve
    x = jax.jit(jax.vmap(token_solve))(csr_values, b)

    # see difference between dense solves and cuDSS
    print(f"difference between cudss and cusolver in single solve: {jnp.linalg.norm(x1 - true_x1)}")
    print(f"difference between cudss and cusolver in vmap solve: {jnp.linalg.norm(x - jnp.stack([true_x1, true_x2]))}")

    assert jnp.allclose(x1, true_x1, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(x, jnp.stack([true_x1, true_x2]), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_composability()
    print("composability example completed.")

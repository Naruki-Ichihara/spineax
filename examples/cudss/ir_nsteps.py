"""
Example: per-call iterative refinement via the ir_nsteps argument.

With the token API, IR is a plain argument on factorize/refactorize/solve —
no signal plumbing. The default is 0 (OFF): IR can corrupt LDL^T re-solves
on indefinite systems (see safe_ir.py for a selection strategy).

Uses an ill-conditioned SPD system (mtype_id=3) where IR converges and
visibly improves accuracy, on the single and block-batch paths.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax import cudss


def make_spd_system(n=200, cond=1e12, seed=0):
    """Create an ill-conditioned SPD system where IR has room to help.

    Dense SPD matrix with prescribed condition number via eigenvalue control.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    Q, _ = jnp.linalg.qr(jax.random.normal(k1, (n, n), dtype=jnp.float64))
    eigs = jnp.logspace(0, jnp.log10(cond), n)  # eigenvalues from 1 to cond
    A = Q @ jnp.diag(eigs) @ Q.T
    A = 0.5 * (A + A.T)  # ensure exact symmetry

    # Upper triangle for CSR (dense, but fine for testing)
    A_upper = jnp.triu(A)
    LHS = jsparse.BCSR.fromdense(A_upper)

    rhs = jax.random.normal(k2, (n,), dtype=jnp.float64)
    x_true = jnp.linalg.solve(A, rhs)

    actual_cond = float(jnp.linalg.cond(A))
    print(f"  n={n}, nnz={LHS.data.shape[0]}, cond(A)={actual_cond:.2e}")
    print(f"  ||x_true|| = {float(jnp.linalg.norm(x_true)):.2e}")

    return LHS.indptr, LHS.indices, LHS.data, rhs, x_true


def test_single_vs_batch():
    """Accuracy vs ir_nsteps, single path and block-batch path."""
    print("=" * 70)
    print("TEST 1: accuracy vs ir_nsteps, single vs block batch")
    print("=" * 70)

    offsets, columns, values, b, x_true = make_spd_system()

    # single entry
    token = cudss.analyze(values, offsets, columns, mtype_id=3, mview_id=1)
    token = cudss.factorize(token, values)

    # one block-diagonal entry for a batch of 4 copies
    batch_size = 4
    vals_batch = jnp.stack([values] * batch_size)
    b_batch = jnp.stack([b] * batch_size)
    btoken = cudss.analyze(vals_batch, offsets, columns, mtype_id=3, mview_id=1)
    btoken = cudss.factorize(btoken, vals_batch)

    # Note: with ir_nsteps > 0 the batch elements are NOT bitwise identical
    # even though the blocks are copies — IR runs one global refinement over
    # the block-diagonal system and is itself not run-to-run deterministic.
    # The per-block errors agree in magnitude; we report the spread.
    print(f"\n  {'nsteps':>6}  |  {'single err':>12}  |  {'batch[0] err':>12}  |  {'block spread':>12}")
    print(f"  {'-'*6}  |  {'-'*12}  |  {'-'*12}  |  {'-'*12}")

    for nsteps in [0, 1, 2, 5, 10, 20]:
        x_single = cudss.solve(token, b, ir_nsteps=nsteps)
        err_single = float(jnp.linalg.norm(x_single - x_true) / jnp.linalg.norm(x_true))

        x_batch = cudss.solve(btoken, b_batch, ir_nsteps=nsteps)
        err_batch = float(jnp.linalg.norm(x_batch[0] - x_true) / jnp.linalg.norm(x_true))

        spread = max(
            float(jnp.linalg.norm(x_batch[i] - x_batch[0]))
            for i in range(1, batch_size)
        )
        print(f"  {nsteps:>6}  |  {err_single:>12.2e}  |  {err_batch:>12.2e}  |  {spread:>12.2e}")

    cudss.release(token)
    cudss.release(btoken)
    print()


def test_ir_respected_in_jit():
    """Two solves in one jit with different ir_nsteps must differ."""
    print("=" * 70)
    print("TEST 2: two solves in one jit, different ir_nsteps")
    print("=" * 70)

    offsets, columns, values, b, x_true = make_spd_system()

    @jax.jit
    def solve_both(values, b):
        token = cudss.analyze(values, offsets, columns, mtype_id=3, mview_id=1)
        token = cudss.factorize(token, values)
        x_0 = cudss.solve(token, b, ir_nsteps=0)
        x_20 = cudss.solve(token, b, ir_nsteps=20)
        return x_0, x_20

    x_0, x_20 = solve_both(values, b)
    err_0 = float(jnp.linalg.norm(x_0 - x_true) / jnp.linalg.norm(x_true))
    err_20 = float(jnp.linalg.norm(x_20 - x_true) / jnp.linalg.norm(x_true))
    diff = float(jnp.linalg.norm(x_0 - x_20))

    print(f"  ir_nsteps=0   |  relative error = {err_0:.2e}")
    print(f"  ir_nsteps=20  |  relative error = {err_20:.2e}")
    print(f"  ||x(ir=0) - x(ir=20)|| = {diff:.2e}")
    if diff > 1e-15:
        print("  PASS: different ir_nsteps produced different results")
    else:
        print("  WARN: results identical — ir_nsteps may not be reaching cuDSS")
    print()


if __name__ == "__main__":
    test_single_vs_batch()
    test_ir_respected_in_jit()
    print("All ir_nsteps examples completed.")

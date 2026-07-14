"""
Re-solve with a new RHS + iterative refinement, WITHOUT refactorization.

History (signal-era bug, see git history of ir_divergence_on_resolve.py):

    factorize -> solve(+IR)                       => correct
    factorize -> solve(+IR) -> new RHS solve(+IR) => IR diverged

Root cause candidate: the old handlers left A's values pointer aimed at a
previous call's (transient) XLA input buffer, so IR's residual computation
could read stale/reused memory on later solves. The token carries the
last-factorized values as a pytree leaf and hands them to EVERY solve call
as a live buffer (zero-copy, design doc step 10) — so re-solves always
compute residuals against the matrix that was actually factorized.

This example re-runs the original reproducer through the token API on a
well-conditioned SPD system and a KKT (symmetric indefinite) system, and
reports whether the second solve stays accurate. It also demonstrates WHY
ir_nsteps defaults to 0: on indefinite LDL^T systems IR can still be
numerically harmful (a cuDSS-level property, not a spineax bug) — see
safe_ir.py for a selection strategy.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np
from spineax import cudss


def make_spd_system(n, seed=0):
    """Build a well-conditioned SPD matrix and two distinct RHS vectors."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.linspace(1.0, 10.0, n)  # condition number = 10
    A_full = Q @ np.diag(eigs) @ Q.T
    A_full = (A_full + A_full.T) / 2

    x_true_1 = rng.standard_normal(n)
    x_true_2 = rng.standard_normal(n) * 5.0
    b1 = A_full @ x_true_1
    b2 = A_full @ x_true_2
    return A_full, b1, b2, x_true_1, x_true_2


def make_kkt_system(n_x, n_c, seed=0):
    """Build a KKT (symmetric indefinite) system and two distinct RHS vectors."""
    rng = np.random.default_rng(seed)
    n = n_x + n_c

    Q, _ = np.linalg.qr(rng.standard_normal((n_x, n_x)))
    eigs = np.linspace(1.0, 5.0, n_x)
    H = Q @ np.diag(eigs) @ Q.T
    H = (H + H.T) / 2

    J = rng.standard_normal((n_c, n_x))

    K = np.zeros((n, n))
    K[:n_x, :n_x] = H
    K[n_x:, :n_x] = J
    K[:n_x, n_x:] = J.T

    x_true_1 = rng.standard_normal(n)
    x_true_2 = rng.standard_normal(n) * 5.0
    b1 = K @ x_true_1
    b2 = K @ x_true_2
    return K, b1, b2, x_true_1, x_true_2


def run_resolve(name, A_full, b1, b2, x_true_1, x_true_2, mtype_id):
    print("=" * 70)
    print(f"{name}: factorize -> solve(b1) -> solve(b2), no refactorization")
    print("=" * 70)

    LHS = jsparse.BCSR.fromdense(jnp.triu(jnp.asarray(A_full)))
    offsets, columns, values = LHS.indptr, LHS.indices, LHS.data
    b1 = jnp.asarray(b1)
    b2 = jnp.asarray(b2)

    print(f"  {'ir':>4}  {'err(b1)':>10}  {'err(b2, re-solve)':>18}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*18}")
    for ir in [0, 5, 20]:
        token = cudss.analyze(values, offsets, columns, mtype_id=mtype_id, mview_id=1)
        token = cudss.factorize(token, values)
        x1 = cudss.solve(token, b1, ir_nsteps=ir)
        x2 = cudss.solve(token, b2, ir_nsteps=ir)  # the historically-divergent call
        err1 = float(jnp.linalg.norm(x1 - x_true_1) / jnp.linalg.norm(x_true_1))
        err2 = float(jnp.linalg.norm(x2 - x_true_2) / jnp.linalg.norm(x_true_2))
        # divergence = the RE-solve being wildly worse than the first solve
        # (the signal-era bug); comparable errors are just this matrix's
        # baseline accuracy at this ir setting
        flag = "  <-- re-solve diverged!" if err2 > 1e3 * max(err1, 1e-14) else ""
        print(f"  {ir:>4}  {err1:>10.2e}  {err2:>18.2e}{flag}")
        cudss.release(token)
    print()


if __name__ == "__main__":
    n = 300
    A, b1, b2, xt1, xt2 = make_spd_system(n)
    run_resolve("SPD (mtype=3)", A, b1, b2, xt1, xt2, mtype_id=3)

    K, b1, b2, xt1, xt2 = make_kkt_system(200, 100)
    run_resolve("KKT / LDL^T (mtype=1)", K, b1, b2, xt1, xt2, mtype_id=1)

    print("ir_divergence example completed.")

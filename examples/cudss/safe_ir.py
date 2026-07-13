"""
Example: Safe iterative refinement for batched indefinite systems.

Strategy: factorize once, solve twice (with and without IR), then
per-element select whichever has the smaller residual norm.

This avoids IR divergence on badly-conditioned indefinite systems
while still benefiting from IR on well-conditioned ones.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax.cudss.solver import CuDSSSolver


def make_kkt_batch(n_x=100, n_c=40, batch_size=8, seed=0):
    """Create a batch of KKT systems — some well-conditioned, some near-singular.

    Returns systems sharing the same sparsity pattern but with
    different numerical values. Half the batch has well-conditioned H
    (IR should help), the other half has near-singular H with tiny
    eigenvalues (static pivoting kicks in hard, IR may diverge).
    """
    key = jax.random.PRNGKey(seed)
    n = n_x + n_c

    all_upper_values = []   # upper triangle CSR values (for cuDSS)
    all_full_values = []    # full symmetric CSR values (for residuals)
    all_rhs = []
    all_x_true = []
    all_conds = []
    all_labels = []

    for i in range(batch_size):
        k1, k2, k3, key = jax.random.split(key, 4)

        Q, _ = jnp.linalg.qr(jax.random.normal(k1, (n_x, n_x), dtype=jnp.float64))

        if i < batch_size // 2:
            # Well-conditioned H: eigenvalues from 1 to 1e3
            eigs = jnp.logspace(0, 3, n_x)
            label = "good H"
        else:
            # Near-singular H: half the eigenvalues are tiny (1e-14),
            # forcing heavy static pivoting in cuDSS
            n_tiny = n_x // 2
            eigs_tiny = jnp.full(n_tiny, 1e-14)
            eigs_ok = jnp.logspace(0, 3, n_x - n_tiny)
            eigs = jnp.concatenate([eigs_tiny, eigs_ok])
            label = "bad H"

        H = Q @ jnp.diag(eigs) @ Q.T

        # Constraint Jacobian
        J = jax.random.normal(k2, (n_c, n_x), dtype=jnp.float64)

        # Assemble KKT
        K = jnp.zeros((n, n), dtype=jnp.float64)
        K = K.at[:n_x, :n_x].set(H)
        K = K.at[n_x:, :n_x].set(J)
        K = K.at[:n_x, n_x:].set(J.T)

        rhs = jax.random.normal(k3, (n,), dtype=jnp.float64)
        x_true = jnp.linalg.solve(K, rhs)

        # Upper triangle CSR (for cuDSS solver)
        K_upper = jnp.triu(K)
        LHS_upper = jsparse.BCSR.fromdense(K_upper)

        # Full symmetric CSR (for sparse residual computation)
        LHS_full = jsparse.BCSR.fromdense(K)

        all_upper_values.append(LHS_upper.data)
        all_full_values.append(LHS_full.data)
        all_rhs.append(rhs)
        all_x_true.append(x_true)
        all_conds.append(float(jnp.linalg.cond(K)))
        all_labels.append(label)

        if i == 0:
            upper_offsets = LHS_upper.indptr
            upper_columns = LHS_upper.indices
            full_offsets = LHS_full.indptr
            full_columns = LHS_full.indices

    upper_vals_batch = jnp.stack(all_upper_values)
    full_vals_batch = jnp.stack(all_full_values)
    b_batch = jnp.stack(all_rhs)
    x_true_batch = jnp.stack(all_x_true)

    return (upper_offsets, upper_columns, upper_vals_batch,
            full_offsets, full_columns, full_vals_batch,
            b_batch, x_true_batch, all_conds, all_labels)


def sparse_residual_norm(full_values, full_offsets, full_columns, n, x, b):
    """Compute ||b - A @ x|| using sparse matvec (no dense materialization)."""
    A = jsparse.BCSR((full_values, full_columns, full_offsets), shape=(n, n))
    return jnp.linalg.norm(b - A @ x)


def test_safe_ir():
    """Factorize once, solve twice, select per-element best."""
    print("=" * 70)
    print("Safe IR: factorize once, solve with ir=0 and ir=20, pick best")
    print("=" * 70)

    batch_size = 8
    (upper_offsets, upper_columns, upper_vals_batch,
     full_offsets, full_columns, full_vals_batch,
     b_batch, x_true_batch, conds, labels) = make_kkt_batch(batch_size=batch_size)

    n = full_offsets.shape[0] - 1  # matrix dimension

    solver = CuDSSSolver(upper_offsets, upper_columns, device_id=0, mtype_id=1, mview_id=1)

    zero = jnp.array([0], dtype=jnp.int32)
    one = jnp.array([1], dtype=jnp.int32)
    ir0 = jnp.array([0], dtype=jnp.int32)
    ir_n = jnp.array([20], dtype=jnp.int32)

    vmap_solver = jax.vmap(solver, in_axes=(0, 0, None, None, None))

    @jax.jit
    def safe_ir_solve(b_batch, upper_vals_batch, full_vals_batch):
        # 1. Factorize only (no solve)
        vmap_solver(b_batch, upper_vals_batch, one, zero, ir0)

        # 2. Solve without IR
        x0, inertia = vmap_solver(b_batch, upper_vals_batch, zero, one, ir0)

        # 3. Solve with IR=20 (reuses same factorization)
        x_ir, _ = vmap_solver(b_batch, upper_vals_batch, zero, one, ir_n)

        # 4. Per-element residual norms via sparse matvec (no dense materialization)
        resid_fn = lambda vals, x, b: sparse_residual_norm(vals, full_offsets, full_columns, n, x, b)
        resid_0 = jax.vmap(resid_fn)(full_vals_batch, x0, b_batch)
        resid_ir = jax.vmap(resid_fn)(full_vals_batch, x_ir, b_batch)

        # 5. Per-element selection: pick whichever has smaller residual
        use_ir = resid_ir < resid_0  # shape [batch_size]
        x_best = jnp.where(use_ir[:, None], x_ir, x0)

        return x0, x_ir, x_best, inertia, resid_0, resid_ir, use_ir

    x0, x_ir, x_best, inertia, resid_0, resid_ir, use_ir = \
        safe_ir_solve(b_batch, upper_vals_batch, full_vals_batch)

    # Report per-element results
    print(f"\n  {'elem':>4}  {'type':>6}  {'cond(K)':>10}  {'err(ir=0)':>10}  {'err(ir=20)':>11}  "
          f"{'err(best)':>10}  {'resid(0)':>10}  {'resid(20)':>11}  {'chose':>6}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*6}")

    for i in range(batch_size):
        err_0 = float(jnp.linalg.norm(x0[i] - x_true_batch[i]) / jnp.linalg.norm(x_true_batch[i]))
        err_ir = float(jnp.linalg.norm(x_ir[i] - x_true_batch[i]) / jnp.linalg.norm(x_true_batch[i]))
        err_best = float(jnp.linalg.norm(x_best[i] - x_true_batch[i]) / jnp.linalg.norm(x_true_batch[i]))
        r0 = float(resid_0[i])
        r_ir = float(resid_ir[i])
        chose = "ir=20" if bool(use_ir[i]) else "ir=0"

        print(f"  {i:>4}  {labels[i]:>6}  {conds[i]:>10.2e}  {err_0:>10.2e}  {err_ir:>11.2e}  "
              f"{err_best:>10.2e}  {r0:>10.2e}  {r_ir:>11.2e}  {chose:>6}")

    # Summary
    n_ir = int(use_ir.sum())
    print(f"\n  Selected ir=20 for {n_ir}/{batch_size} elements, ir=0 for {batch_size - n_ir}/{batch_size}")

    err_all_0 = float(jnp.linalg.norm(x0 - x_true_batch) / jnp.linalg.norm(x_true_batch))
    err_all_ir = float(jnp.linalg.norm(x_ir - x_true_batch) / jnp.linalg.norm(x_true_batch))
    err_all_best = float(jnp.linalg.norm(x_best - x_true_batch) / jnp.linalg.norm(x_true_batch))
    print(f"  Batch relative error (ir=0):   {err_all_0:.2e}")
    print(f"  Batch relative error (ir=20):  {err_all_ir:.2e}")
    print(f"  Batch relative error (best):   {err_all_best:.2e}")
    print()


if __name__ == "__main__":
    test_safe_ir()
    print("Safe IR example completed.")

"""
Example: Runtime-configurable iterative refinement steps.

Demonstrates passing ir_nsteps_signal as a GPU integer to control
the number of iterative refinement steps cuDSS performs per solve.

Uses a KKT system where H is moderately conditioned so the LDL^T
factorization is decent but has rounding error that IR can correct.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax.cudss.solver import CuDSSSolver


def make_kkt_system(n_x=100, n_c=40, cond_H=1e4, seed=0):
    """Create a KKT system with controlled conditioning.

    KKT structure:
        [H   J^T] [x]   [g]
        [J   0  ] [y] = [h]

    H is symmetric positive definite with prescribed condition number.
    The overall KKT is symmetric indefinite (saddle point).
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    n = n_x + n_c

    # H with controlled condition number
    Q, _ = jnp.linalg.qr(jax.random.normal(k1, (n_x, n_x), dtype=jnp.float64))
    eigs = jnp.logspace(0, jnp.log10(cond_H), n_x)  # eigenvalues from 1 to cond_H
    H = Q @ jnp.diag(eigs) @ Q.T

    # Constraint Jacobian — scale so J contributions are comparable to H
    J = jax.random.normal(k2, (n_c, n_x), dtype=jnp.float64)

    # Assemble KKT (symmetric indefinite)
    K = jnp.zeros((n, n), dtype=jnp.float64)
    K = K.at[:n_x, :n_x].set(H)
    K = K.at[n_x:, :n_x].set(J)
    K = K.at[:n_x, n_x:].set(J.T)

    rhs = jax.random.normal(k3, (n,), dtype=jnp.float64)
    x_true = jnp.linalg.solve(K, rhs)

    # Upper triangle CSR for cuDSS (symmetric, upper view)
    K_upper = jnp.triu(K)
    LHS = jsparse.BCSR.fromdense(K_upper)

    kkt_cond = float(jnp.linalg.cond(K))
    print(f"  KKT: n_x={n_x}, n_c={n_c}, total={n}, nnz={LHS.data.shape[0]}")
    print(f"  cond(H)={cond_H:.0e}, cond(K)={kkt_cond:.2e}")
    print(f"  ||x_true|| = {float(jnp.linalg.norm(x_true)):.2e}")

    return LHS.indptr, LHS.indices, LHS.data, rhs, x_true


def test_ir_nsteps_single():
    """Test different IR step counts on KKT system."""
    print("=" * 60)
    print("Single solve: varying ir_nsteps on KKT system")
    print("=" * 60)

    offsets, columns, values, b, x_true = make_kkt_system()
    solver = CuDSSSolver(offsets, columns, device_id=0, mtype_id=1, mview_id=1)

    refac = jnp.array([1], dtype=jnp.int32)
    do_solve = jnp.array([1], dtype=jnp.int32)

    for nsteps in [0, 1, 2, 5, 10, 20, 50]:
        ir_signal = jnp.array([nsteps], dtype=jnp.int32)
        x, _ = solver(b, values, refac, do_solve, ir_nsteps_signal=ir_signal)
        err = float(jnp.linalg.norm(x - x_true) / jnp.linalg.norm(x_true))
        print(f"  ir_nsteps={nsteps:3d}  |  relative error = {err:.2e}")

    print()


def test_ir_nsteps_in_jit():
    """Test that ir_nsteps can vary within a single jit scope."""
    print("=" * 60)
    print("Within-jit: two solves with different ir_nsteps")
    print("=" * 60)

    offsets, columns, values, b, x_true = make_kkt_system()
    solver = CuDSSSolver(offsets, columns, device_id=0, mtype_id=1, mview_id=1)

    @jax.jit
    def solve_twice(b, vals):
        refac = jnp.array([1], dtype=jnp.int32)
        do_solve = jnp.array([1], dtype=jnp.int32)
        ir_low = jnp.array([0], dtype=jnp.int32)
        ir_high = jnp.array([50], dtype=jnp.int32)
        x_low, _ = solver(b, vals, refac, do_solve, ir_nsteps_signal=ir_low)
        x_high, _ = solver(b, vals, refac, do_solve, ir_nsteps_signal=ir_high)
        return x_low, x_high

    x_low, x_high = solve_twice(b, values)
    err_low = float(jnp.linalg.norm(x_low - x_true) / jnp.linalg.norm(x_true))
    err_high = float(jnp.linalg.norm(x_high - x_true) / jnp.linalg.norm(x_true))
    print(f"  ir_nsteps=0   |  relative error = {err_low:.2e}")
    print(f"  ir_nsteps=50  |  relative error = {err_high:.2e}")
    print()


def test_ir_nsteps_vmap():
    """Test ir_nsteps flows correctly through vmap."""
    print("=" * 60)
    print("vmap solve: ir_nsteps_signal with batched KKT inputs")
    print("=" * 60)

    offsets, columns, values, b, x_true = make_kkt_system()
    solver = CuDSSSolver(offsets, columns, device_id=0, mtype_id=1, mview_id=1)

    batch_size = 4
    b_batch = jnp.stack([b] * batch_size)
    vals_batch = jnp.stack([values] * batch_size)

    refac = jnp.array([1], dtype=jnp.int32)
    do_solve = jnp.array([1], dtype=jnp.int32)

    for nsteps in [0, 5, 50]:
        ir_signal = jnp.array([nsteps], dtype=jnp.int32)
        x_batch, _ = jax.jit(jax.vmap(
            solver, in_axes=(0, 0, None, None, None)
        ))(b_batch, vals_batch, refac, do_solve, ir_signal)
        err = float(jnp.linalg.norm(x_batch[0] - x_true) / jnp.linalg.norm(x_true))
        print(f"  ir_nsteps={nsteps:3d}  |  relative error (first batch element) = {err:.2e}")

    print()


def test_ir_nsteps_default():
    """Test that omitting ir_nsteps_signal uses the default (5 steps)."""
    print("=" * 60)
    print("Default ir_nsteps (should be 5)")
    print("=" * 60)

    offsets, columns, values, b, x_true = make_kkt_system()
    solver = CuDSSSolver(offsets, columns, device_id=0, mtype_id=1, mview_id=1)

    refac = jnp.array([1], dtype=jnp.int32)
    do_solve = jnp.array([1], dtype=jnp.int32)

    # Without ir_nsteps_signal (default = 5)
    x_default, _ = solver(b, values, refac, do_solve)

    # Explicit ir_nsteps=5
    ir_5 = jnp.array([5], dtype=jnp.int32)
    x_explicit, _ = solver(b, values, refac, do_solve, ir_nsteps_signal=ir_5)

    err_default = float(jnp.linalg.norm(x_default - x_true) / jnp.linalg.norm(x_true))
    err_explicit = float(jnp.linalg.norm(x_explicit - x_true) / jnp.linalg.norm(x_true))
    print(f"  default (None) |  relative error = {err_default:.2e}")
    print(f"  explicit (5)   |  relative error = {err_explicit:.2e}")
    print()


if __name__ == "__main__":
    test_ir_nsteps_single()
    test_ir_nsteps_in_jit()
    test_ir_nsteps_vmap()
    test_ir_nsteps_default()
    print("All ir_nsteps tests completed.")

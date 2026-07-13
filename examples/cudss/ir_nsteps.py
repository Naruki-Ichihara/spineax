"""
Test: Runtime-configurable iterative refinement steps via ir_nsteps_signal.

Uses an ill-conditioned SPD system (mtype_id=3) where IR converges and
visibly improves accuracy. Tests single, jit, and vmapped pseudo-batch paths
to verify ir_nsteps_signal plumbing end-to-end.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax.cudss.solver import CuDSSSolver


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


def test_single_vs_vmap():
    """Core test: verify vmapped pseudo-batch path matches single solve for each ir_nsteps."""
    print("=" * 70)
    print("TEST 1: Single vs vmap consistency for varying ir_nsteps")
    print("=" * 70)

    offsets, columns, values, b, x_true = make_spd_system()

    # Single solver
    solver_single = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)

    # Separate solver for vmap (needs its own solver_id / state)
    solver_vmap = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)

    refac = jnp.array([1], dtype=jnp.int32)
    do_solve = jnp.array([1], dtype=jnp.int32)
    batch_size = 4
    b_batch = jnp.stack([b] * batch_size)
    vals_batch = jnp.stack([values] * batch_size)

    vmap_fn = jax.jit(jax.vmap(solver_vmap, in_axes=(0, 0, None, None, None)))

    print(f"\n  {'nsteps':>6}  |  {'single err':>12}  |  {'vmap[0] err':>12}  |  {'match':>8}")
    print(f"  {'-'*6}  |  {'-'*12}  |  {'-'*12}  |  {'-'*8}")

    for nsteps in [0, 1, 2, 5, 10, 20]:
        ir_signal = jnp.array([nsteps], dtype=jnp.int32)

        # Single solve
        x_single, _ = solver_single(b, values, refac, do_solve, ir_nsteps_signal=ir_signal)
        err_single = float(jnp.linalg.norm(x_single - x_true) / jnp.linalg.norm(x_true))

        # Vmapped pseudo-batch solve
        x_vmap, _ = vmap_fn(b_batch, vals_batch, refac, do_solve, ir_signal)
        err_vmap = float(jnp.linalg.norm(x_vmap[0] - x_true) / jnp.linalg.norm(x_true))

        # Check all batch elements agree
        all_same = all(
            float(jnp.linalg.norm(x_vmap[i] - x_vmap[0])) < 1e-15
            for i in range(1, batch_size)
        )

        print(f"  {nsteps:>6}  |  {err_single:>12.2e}  |  {err_vmap:>12.2e}  |  {'OK' if all_same else 'DIFF!'}")

    print()


def test_vmap_ir_signal_respected():
    """Verify that changing ir_nsteps_signal actually changes vmap output."""
    print("=" * 70)
    print("TEST 2: vmap output changes with ir_nsteps (signal is respected)")
    print("=" * 70)

    offsets, columns, values, b, x_true = make_spd_system()
    solver = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)

    batch_size = 4
    b_batch = jnp.stack([b] * batch_size)
    vals_batch = jnp.stack([values] * batch_size)
    refac = jnp.array([1], dtype=jnp.int32)
    do_solve = jnp.array([1], dtype=jnp.int32)

    vmap_fn = jax.jit(jax.vmap(solver, in_axes=(0, 0, None, None, None)))

    ir_0 = jnp.array([0], dtype=jnp.int32)
    ir_10 = jnp.array([10], dtype=jnp.int32)

    x_0, _ = vmap_fn(b_batch, vals_batch, refac, do_solve, ir_0)
    x_10, _ = vmap_fn(b_batch, vals_batch, refac, do_solve, ir_10)

    diff = float(jnp.linalg.norm(x_0[0] - x_10[0]))
    print(f"  ||x(ir=0) - x(ir=10)|| = {diff:.2e}")
    if diff > 1e-15:
        print(f"  PASS: different ir_nsteps produced different results")
    else:
        print(f"  WARN: results identical — ir_nsteps_signal may not be reaching cuDSS")
    print()


def test_within_jit_vmap():
    """Two vmapped solves in the same jit with different ir_nsteps — shared state registry test."""
    print("=" * 70)
    print("TEST 3: Two vmapped solves in one jit, different ir_nsteps (shared state)")
    print("=" * 70)

    offsets, columns, values, b, x_true = make_spd_system()
    solver = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)

    batch_size = 4
    b_batch = jnp.stack([b] * batch_size)
    vals_batch = jnp.stack([values] * batch_size)

    @jax.jit
    def two_vmap_solves(b_batch, vals_batch):
        refac = jnp.array([1], dtype=jnp.int32)
        do_solve = jnp.array([1], dtype=jnp.int32)
        ir_low = jnp.array([0], dtype=jnp.int32)
        ir_high = jnp.array([20], dtype=jnp.int32)

        fn = jax.vmap(solver, in_axes=(0, 0, None, None, None))
        x_low, _ = fn(b_batch, vals_batch, refac, do_solve, ir_low)
        x_high, _ = fn(b_batch, vals_batch, refac, do_solve, ir_high)
        return x_low, x_high

    x_low, x_high = two_vmap_solves(b_batch, vals_batch)
    err_low = float(jnp.linalg.norm(x_low[0] - x_true) / jnp.linalg.norm(x_true))
    err_high = float(jnp.linalg.norm(x_high[0] - x_true) / jnp.linalg.norm(x_true))

    print(f"  vmap ir_nsteps=0   |  relative error = {err_low:.2e}")
    print(f"  vmap ir_nsteps=20  |  relative error = {err_high:.2e}")
    print(f"  Shared state: both calls used solver_id={solver.solver_id}")
    print()


def test_vmap_default():
    """Test that omitting ir_nsteps_signal uses default (5) in the vmap path."""
    print("=" * 70)
    print("TEST 4: vmap with default ir_nsteps (omitted = 5)")
    print("=" * 70)

    offsets, columns, values, b, x_true = make_spd_system()
    solver = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)

    batch_size = 4
    b_batch = jnp.stack([b] * batch_size)
    vals_batch = jnp.stack([values] * batch_size)
    refac = jnp.array([1], dtype=jnp.int32)
    do_solve = jnp.array([1], dtype=jnp.int32)

    # Default (no ir_nsteps_signal — __call__ fills in default=5)
    x_default, _ = jax.jit(jax.vmap(
        solver, in_axes=(0, 0, None, None)
    ))(b_batch, vals_batch, refac, do_solve)

    # Explicit ir_nsteps=5
    ir_5 = jnp.array([5], dtype=jnp.int32)
    x_explicit, _ = jax.jit(jax.vmap(
        solver, in_axes=(0, 0, None, None, None)
    ))(b_batch, vals_batch, refac, do_solve, ir_5)

    diff = float(jnp.linalg.norm(x_default[0] - x_explicit[0]))
    err_default = float(jnp.linalg.norm(x_default[0] - x_true) / jnp.linalg.norm(x_true))
    err_explicit = float(jnp.linalg.norm(x_explicit[0] - x_true) / jnp.linalg.norm(x_true))

    print(f"  default (None) |  relative error = {err_default:.2e}")
    print(f"  explicit (5)   |  relative error = {err_explicit:.2e}")
    print(f"  ||default - explicit|| = {diff:.2e}")
    if diff < 1e-15:
        print(f"  PASS: default matches explicit ir_nsteps=5")
    else:
        print(f"  WARN: default differs from explicit ir_nsteps=5")
    print()


if __name__ == "__main__":
    test_single_vs_vmap()
    test_vmap_ir_signal_respected()
    test_within_jit_vmap()
    test_vmap_default()
    print("All ir_nsteps vmap tests completed.")

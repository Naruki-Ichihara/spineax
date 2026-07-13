"""
Test: Does an ir=0 solve on a shared cuDSS state corrupt a subsequent ir=5 solve?
Is the issue two solves, or separating refactorize from solve into different calls?

Tests (each uses its own solver instance / cuDSS state):
  A) refac+solve(ir=5) in ONE call                    (baseline)
  B) refac(no solve) + solve(ir=0) + solve(ir=5)       (two solves)
  C) refac(no solve) + solve(ir=5)                     (split refac/solve, single solve)
  D) refac+solve(ir=0) in ONE call                     (baseline ir=0)

If C matches A: the issue is two solves corrupting each other.
If C differs from A: the issue is splitting refac and solve into separate calls.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax.cudss.solver import CuDSSSolver


def make_spd_system(n=200, cond=1e10, seed=0):
    """SPD system where IR visibly helps."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    Q, _ = jnp.linalg.qr(jax.random.normal(k1, (n, n), dtype=jnp.float64))
    eigs = jnp.logspace(0, jnp.log10(cond), n)
    A = Q @ jnp.diag(eigs) @ Q.T
    A = 0.5 * (A + A.T)
    A_upper = jnp.triu(A)
    LHS = jsparse.BCSR.fromdense(A_upper)
    rhs = jax.random.normal(k2, (n,), dtype=jnp.float64)
    x_true = jnp.linalg.solve(A, rhs)
    return LHS.indptr, LHS.indices, LHS.data, rhs, x_true


def test_cold_path():
    """Test corruption on cold path (first JIT call)."""
    print("=" * 70)
    print("COLD PATH: first JIT call")
    print("=" * 70)

    offsets, columns, values, b, x_true = make_spd_system()

    refac = jnp.array([1], dtype=jnp.int32)
    no_refac = jnp.array([0], dtype=jnp.int32)
    do_solve = jnp.array([1], dtype=jnp.int32)
    no_solve = jnp.array([0], dtype=jnp.int32)
    ir_0 = jnp.array([0], dtype=jnp.int32)
    ir_5 = jnp.array([5], dtype=jnp.int32)

    # A) Baseline: refac+solve(ir=5) in one call
    solver_a = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)

    @jax.jit
    def test_a(b, vals):
        x, _ = solver_a(b, vals, refac, do_solve, ir_nsteps_signal=ir_5)
        return x

    # B) Two solves: refac(no solve) + solve(ir=0) + solve(ir=5)
    solver_b = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)

    @jax.jit
    def test_b(b, vals):
        solver_b(b, vals, refac, no_solve, ir_nsteps_signal=ir_0)
        x0, _ = solver_b(b, vals, no_refac, do_solve, ir_nsteps_signal=ir_0)
        x5, _ = solver_b(b, vals, no_refac, do_solve, ir_nsteps_signal=ir_5)
        return x0, x5

    # C) Split refac/solve (single solve): refac(no solve) + solve(ir=5)
    solver_c = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)

    @jax.jit
    def test_c(b, vals):
        solver_c(b, vals, refac, no_solve, ir_nsteps_signal=ir_0)
        x5, _ = solver_c(b, vals, no_refac, do_solve, ir_nsteps_signal=ir_5)
        return x5

    # D) Baseline: refac+solve(ir=0) in one call
    solver_d = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)

    @jax.jit
    def test_d(b, vals):
        x, _ = solver_d(b, vals, refac, do_solve, ir_nsteps_signal=ir_0)
        return x

    x_a = test_a(b, values)
    x0_b, x5_b = test_b(b, values)
    x_c = test_c(b, values)
    x_d = test_d(b, values)

    err_a = float(jnp.linalg.norm(x_a - x_true) / jnp.linalg.norm(x_true))
    err_b0 = float(jnp.linalg.norm(x0_b - x_true) / jnp.linalg.norm(x_true))
    err_b5 = float(jnp.linalg.norm(x5_b - x_true) / jnp.linalg.norm(x_true))
    err_c = float(jnp.linalg.norm(x_c - x_true) / jnp.linalg.norm(x_true))
    err_d = float(jnp.linalg.norm(x_d - x_true) / jnp.linalg.norm(x_true))

    print(f"\n  A) refac+solve(ir=5) combined:    err = {err_a:.2e}")
    print(f"  B) refac + solve(ir=0) + solve(ir=5):")
    print(f"       ir=0:                        err = {err_b0:.2e}")
    print(f"       ir=5:                        err = {err_b5:.2e}")
    print(f"  C) refac + solve(ir=5) split:     err = {err_c:.2e}")
    print(f"  D) refac+solve(ir=0) combined:    err = {err_d:.2e}")

    diff_ac = float(jnp.linalg.norm(x_a - x_c))
    diff_ab5 = float(jnp.linalg.norm(x_a - x5_b))
    print(f"\n  ||A - C|| = {diff_ac:.2e}  (split refac/solve issue?  {'YES' if diff_ac > 1e-14 else 'no'})")
    print(f"  ||A - B.ir5|| = {diff_ab5:.2e}  (two-solve corruption?  {'YES' if diff_ab5 > 1e-14 else 'no'})")
    if diff_ac < 1e-14 and diff_ab5 > 1e-14:
        print(f"  --> Splitting refac/solve is fine; two solves corrupt each other")
    elif diff_ac > 1e-14:
        print(f"  --> Splitting refac/solve into separate calls causes differences")
    print()

    return values


def test_warm_path():
    """Test corruption on warm path (second+ JIT call, with refactorization)."""
    print("=" * 70)
    print("WARM PATH: second JIT call (perturbed values, refactorize)")
    print("=" * 70)

    offsets, columns, values, b, x_true = make_spd_system()

    refac = jnp.array([1], dtype=jnp.int32)
    no_refac = jnp.array([0], dtype=jnp.int32)
    do_solve = jnp.array([1], dtype=jnp.int32)
    no_solve = jnp.array([0], dtype=jnp.int32)
    ir_0 = jnp.array([0], dtype=jnp.int32)
    ir_5 = jnp.array([5], dtype=jnp.int32)

    solver_a5 = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)
    solver_a0 = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)
    solver_c5 = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)
    solver_c0 = CuDSSSolver(offsets, columns, device_id=0, mtype_id=3, mview_id=1)

    @jax.jit
    def combined_ir5(b, vals):
        x, _ = solver_a5(b, vals, refac, do_solve, ir_nsteps_signal=ir_5)
        return x

    @jax.jit
    def combined_ir0(b, vals):
        x, _ = solver_a0(b, vals, refac, do_solve, ir_nsteps_signal=ir_0)
        return x

    @jax.jit
    def split_ir5(b, vals):
        solver_c5(b, vals, refac, no_solve, ir_nsteps_signal=ir_0)
        x, _ = solver_c5(b, vals, no_refac, do_solve, ir_nsteps_signal=ir_5)
        return x

    @jax.jit
    def split_ir0(b, vals):
        solver_c0(b, vals, refac, no_solve, ir_nsteps_signal=ir_0)
        x, _ = solver_c0(b, vals, no_refac, do_solve, ir_nsteps_signal=ir_0)
        return x

    # Cold path (prime the JIT)
    _ = combined_ir5(b, values)
    _ = combined_ir0(b, values)
    _ = split_ir5(b, values)
    _ = split_ir0(b, values)

    # Warm path iterations with perturbed values
    for i, scale_factor in enumerate([1.001, 1.002, 1.003], start=2):
        vals_i = values * scale_factor
        x_comb5 = combined_ir5(b, vals_i)
        x_comb0 = combined_ir0(b, vals_i)
        x_split5 = split_ir5(b, vals_i)
        x_split0 = split_ir0(b, vals_i)

        scale = float(jnp.linalg.norm(x_comb5))
        diff_ir5 = float(jnp.linalg.norm(x_comb5 - x_split5)) / scale
        diff_ir0 = float(jnp.linalg.norm(x_comb0 - x_split0)) / scale

        print(f"  Call {i}: ||combined - split|| / ||x||:")
        print(f"    ir=5: {diff_ir5:.2e}  {'DIFFERS' if diff_ir5 > 1e-14 else 'OK'}")
        print(f"    ir=0: {diff_ir0:.2e}  {'DIFFERS' if diff_ir0 > 1e-14 else 'OK'}")

    print()
    print("  If ir=0 is OK but ir=5 DIFFERS: IR uses stale A pointer (cuDSS issue)")
    print("  If both DIFFER: our refac/solve split is fundamentally broken")
    print()


if __name__ == "__main__":
    test_cold_path()
    test_warm_path()
    print("Diagnostic tests completed.")

"""
Reproducer for IR divergence when re-solving with a new RHS without refactorization.

Bug description:
  factorize -> solve(+IR)                      => correct solution
  factorize -> solve(+IR) -> new RHS solve(+IR) => IR diverges (without new refactorization)

Tests three calling patterns:
  A) Passing signals explicitly as dynamic args (baseline)
  B) ft.partial with signals captured outside jit, called inside jit scope
  C) ft.partial partials called as separate jit invocations (no shared scope)
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import functools as ft

import jax
jax.config.update("jax_enable_x64", True)
jax.devices()  # force CUDA init

import equinox as eqx
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np
from spineax.cudss.solver import CuDSSSolver


def make_spd_system(n, seed=0):
    """Build a well-conditioned SPD matrix and two distinct RHS vectors."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.linspace(1.0, 10.0, n)  # condition number = 10
    A_full = Q @ np.diag(eigs) @ Q.T
    A_full = (A_full + A_full.T) / 2

    A_upper = np.triu(A_full)

    x_true_1 = rng.standard_normal(n)
    x_true_2 = rng.standard_normal(n) * 5.0
    b1 = A_full @ x_true_1
    b2 = A_full @ x_true_2

    return A_full, A_upper, b1, b2


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
    K_upper = np.triu(K)

    x_true_1 = rng.standard_normal(n)
    x_true_2 = rng.standard_normal(n) * 3.0
    b1 = K @ x_true_1
    b2 = K @ x_true_2

    return K, K_upper, b1, b2


def to_csr(A_upper):
    """Convert dense upper-triangular matrix to JAX CSR components."""
    M = jnp.array(A_upper, dtype=jnp.float64)
    LHS = jsparse.BCSR.fromdense(M)
    return LHS.indptr, LHS.indices, LHS.data


def residual_norm(A_full, x, b):
    return float(jnp.linalg.norm(jnp.array(A_full) @ x - jnp.array(b)))


def report(label, res):
    diverged = res > 1.0 or not np.isfinite(res)
    tag = ">>> BUG CONFIRMED" if diverged else "OK"
    print(f"  {label}:  residual = {res:.2e}  [{tag}]")
    return diverged


# =========================================================================
# Test A: explicit dynamic signal args (baseline — known to work)
# =========================================================================
def test_explicit_signals(name, A_full, A_upper, b1, b2, mtype_id):
    print(f"\n{'='*70}")
    print(f"[A] Explicit signals (single): {name}")
    print(f"{'='*70}")

    csr_offsets, csr_columns, csr_values = to_csr(A_upper)
    solver = CuDSSSolver(csr_offsets, csr_columns, 0, mtype_id, 1)
    jit_solver = eqx.filter_jit(solver)

    refact_on  = jnp.array([1], dtype=jnp.int32)
    refact_off = jnp.array([0], dtype=jnp.int32)
    solve_on   = jnp.array([1], dtype=jnp.int32)

    b1j = jnp.array(b1, dtype=jnp.float64)
    b2j = jnp.array(b2, dtype=jnp.float64)

    x1, _ = jit_solver(b1j, csr_values, refact_on, solve_on)
    jax.block_until_ready(x1)
    report("Call 1  factorize+solve  b1", residual_norm(A_full, x1, b1))

    x2, _ = jit_solver(b2j, csr_values, refact_off, solve_on)
    jax.block_until_ready(x2)
    report("Call 2  solve-only       b2", residual_norm(A_full, x2, b2))

    x3, _ = jit_solver(b2j, csr_values, refact_on, solve_on)
    jax.block_until_ready(x3)
    report("Call 3  refact+solve     b2", residual_norm(A_full, x3, b2))


# =========================================================================
# Test B: ft.partial defined outside, both called within ONE jit scope
#   — mirrors the user's actual pattern
# =========================================================================
def test_partial_same_jit_scope(name, A_full, A_upper, b1, b2, mtype_id):
    print(f"\n{'='*70}")
    print(f"[B] ft.partial inside same jit scope (single): {name}")
    print(f"{'='*70}")

    csr_offsets, csr_columns, csr_values = to_csr(A_upper)
    _solver = CuDSSSolver(csr_offsets, csr_columns, 0, mtype_id, 1)

    # --- partials defined OUTSIDE jit, exactly as user does ---
    refactorize_and_solve = ft.partial(
        _solver,
        refactorize_signal=jnp.array([1], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    )
    solve_only = ft.partial(
        _solver,
        refactorize_signal=jnp.array([0], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    )

    b1j = jnp.array(b1, dtype=jnp.float64)
    b2j = jnp.array(b2, dtype=jnp.float64)

    # Both partials called inside a single jit scope
    @eqx.filter_jit
    def factorize_then_resolve(b_first, b_second, csr_vals):
        x1, in1 = refactorize_and_solve(b_first, csr_vals)
        x2, in2 = solve_only(b_second, csr_vals)
        return x1, x2

    x1, x2 = factorize_then_resolve(b1j, b2j, csr_values)
    jax.block_until_ready(x2)
    report("Call 1  refact+solve  b1", residual_norm(A_full, x1, b1))
    report("Call 2  solve-only    b2", residual_norm(A_full, x2, b2))

    # Second invocation — same compiled XLA, call_count increments
    x1b, x2b = factorize_then_resolve(b1j, b2j, csr_values)
    jax.block_until_ready(x2b)
    report("Invoke2 Call 1  refact+solve  b1", residual_norm(A_full, x1b, b1))
    report("Invoke2 Call 2  solve-only    b2", residual_norm(A_full, x2b, b2))


# =========================================================================
# Test C: ft.partial, each wrapped in its own filter_jit (separate scopes)
# =========================================================================
def test_partial_separate_jit(name, A_full, A_upper, b1, b2, mtype_id):
    print(f"\n{'='*70}")
    print(f"[C] ft.partial separate jit scopes (single): {name}")
    print(f"{'='*70}")

    csr_offsets, csr_columns, csr_values = to_csr(A_upper)
    _solver = CuDSSSolver(csr_offsets, csr_columns, 0, mtype_id, 1)

    refactorize_and_solve = eqx.filter_jit(ft.partial(
        _solver,
        refactorize_signal=jnp.array([1], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    ))
    solve_only = eqx.filter_jit(ft.partial(
        _solver,
        refactorize_signal=jnp.array([0], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    ))

    b1j = jnp.array(b1, dtype=jnp.float64)
    b2j = jnp.array(b2, dtype=jnp.float64)

    x1, _ = refactorize_and_solve(b1j, csr_values)
    jax.block_until_ready(x1)
    report("Call 1  refact+solve  b1", residual_norm(A_full, x1, b1))

    x2, _ = solve_only(b2j, csr_values)
    jax.block_until_ready(x2)
    report("Call 2  solve-only    b2", residual_norm(A_full, x2, b2))

    x3, _ = refactorize_and_solve(b2j, csr_values)
    jax.block_until_ready(x3)
    report("Call 3  refact+solve  b2", residual_norm(A_full, x3, b2))


# =========================================================================
# Test D: ft.partial inside same jit, BATCHED (vmap)
# =========================================================================
def test_partial_same_jit_scope_batched(name, A_full, A_upper, b1, b2, mtype_id):
    print(f"\n{'='*70}")
    print(f"[D] ft.partial inside same jit scope (batched): {name}")
    print(f"{'='*70}")

    csr_offsets, csr_columns, csr_values = to_csr(A_upper)
    batch_size = 4
    _solver = CuDSSSolver(csr_offsets, csr_columns, 0, mtype_id, 1)

    refactorize_and_solve = ft.partial(
        _solver,
        refactorize_signal=jnp.array([1], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    )
    solve_only = ft.partial(
        _solver,
        refactorize_signal=jnp.array([0], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    )

    b1_batch = jnp.tile(jnp.array(b1, dtype=jnp.float64)[None, :], (batch_size, 1))
    b2_batch = jnp.tile(jnp.array(b2, dtype=jnp.float64)[None, :], (batch_size, 1))
    csr_batch = jnp.tile(csr_values[None, :], (batch_size, 1))

    @eqx.filter_jit
    def factorize_then_resolve(b_first, b_second, csr_vals):
        x1, in1 = jax.vmap(refactorize_and_solve)(b_first, csr_vals)
        x2, in2 = jax.vmap(solve_only)(b_second, csr_vals)
        return x1, x2

    x1, x2 = factorize_then_resolve(b1_batch, b2_batch, csr_batch)
    jax.block_until_ready(x2)
    report("Call 1  refact+solve  b1", residual_norm(A_full, x1[0], b1))
    report("Call 2  solve-only    b2", residual_norm(A_full, x2[0], b2))


# =========================================================================
# Test E: factorize-only then solve-only (explicit signals, separate calls)
# =========================================================================
def test_factorize_then_solve_explicit(name, A_full, A_upper, b1, b2, mtype_id):
    print(f"\n{'='*70}")
    print(f"[E] Factorize-only then solve-only, explicit signals: {name}")
    print(f"{'='*70}")

    csr_offsets, csr_columns, csr_values = to_csr(A_upper)
    solver = CuDSSSolver(csr_offsets, csr_columns, 0, mtype_id, 1)
    jit_solver = eqx.filter_jit(solver)

    refact_on  = jnp.array([1], dtype=jnp.int32)
    refact_off = jnp.array([0], dtype=jnp.int32)
    solve_on   = jnp.array([1], dtype=jnp.int32)
    solve_off  = jnp.array([0], dtype=jnp.int32)

    b1j = jnp.array(b1, dtype=jnp.float64)
    b2j = jnp.array(b2, dtype=jnp.float64)

    # Factorize only (no solve) with b1 as dummy RHS
    _, _ = jit_solver(b1j, csr_values, refact_on, solve_off)

    # Solve only with b1
    x1, _ = jit_solver(b1j, csr_values, refact_off, solve_on)
    jax.block_until_ready(x1)
    report("factorize-only -> solve b1", residual_norm(A_full, x1, b1))

    # Solve only with b2 (no refactorization)
    x2, _ = jit_solver(b2j, csr_values, refact_off, solve_on)
    jax.block_until_ready(x2)
    report("solve-only           -> b2", residual_norm(A_full, x2, b2))

    # Control: refactorize + solve with b2
    x3, _ = jit_solver(b2j, csr_values, refact_on, solve_on)
    jax.block_until_ready(x3)
    report("refact+solve         -> b2", residual_norm(A_full, x3, b2))


# =========================================================================
# Test F: ft.partial with all 3 partials (user's exact pattern), same jit
#   refactorize (factorize-only) -> linear_solve (solve-only) -> new RHS
# =========================================================================
def test_partial_factorize_then_solve_same_jit(name, A_full, A_upper, b1, b2, mtype_id):
    print(f"\n{'='*70}")
    print(f"[F] ft.partial 3-way (refact/solve/both) same jit: {name}")
    print(f"{'='*70}")

    csr_offsets, csr_columns, csr_values = to_csr(A_upper)
    _solver = CuDSSSolver(csr_offsets, csr_columns, 0, mtype_id, 1)

    # User's exact partial definitions
    linear_solve = ft.partial(
        _solver,
        refactorize_signal=jnp.array([0], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    )
    refactorize_and_linear_solve = ft.partial(
        _solver,
        refactorize_signal=jnp.array([1], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    )
    refactorize = ft.partial(
        _solver,
        refactorize_signal=jnp.array([1], dtype=jnp.int32),
        solve_signal=jnp.array([0], dtype=jnp.int32),
    )

    b1j = jnp.array(b1, dtype=jnp.float64)
    b2j = jnp.array(b2, dtype=jnp.float64)

    # Pattern: factorize-only, then solve b1, then solve b2
    @eqx.filter_jit
    def factorize_then_two_solves(b_first, b_second, csr_vals):
        _, in0 = refactorize(b_first, csr_vals)         # factorize only
        x1, in1 = linear_solve(b_first, csr_vals)       # solve only b1
        x2, in2 = linear_solve(b_second, csr_vals)      # solve only b2
        return x1, x2

    x1, x2 = factorize_then_two_solves(b1j, b2j, csr_values)
    jax.block_until_ready(x2)
    report("refact-only -> solve b1", residual_norm(A_full, x1, b1))
    report("solve-only       -> b2", residual_norm(A_full, x2, b2))

    # Second invocation
    x1b, x2b = factorize_then_two_solves(b1j, b2j, csr_values)
    jax.block_until_ready(x2b)
    report("Invoke2: refact-only -> solve b1", residual_norm(A_full, x1b, b1))
    report("Invoke2: solve-only       -> b2", residual_norm(A_full, x2b, b2))

    # Also test: refactorize_and_linear_solve first, then solve-only
    @eqx.filter_jit
    def combined_then_resolve(b_first, b_second, csr_vals):
        x1, in1 = refactorize_and_linear_solve(b_first, csr_vals)
        x2, in2 = linear_solve(b_second, csr_vals)
        return x1, x2

    x1c, x2c = combined_then_resolve(b1j, b2j, csr_values)
    jax.block_until_ready(x2c)
    report("refact+solve b1 -> solve-only b2", residual_norm(A_full, x2c, b2))


# =========================================================================
# Test G: ft.partial 3-way, separate jit scopes (sequential calls)
# =========================================================================
def test_partial_factorize_then_solve_separate_jit(name, A_full, A_upper, b1, b2, mtype_id):
    print(f"\n{'='*70}")
    print(f"[G] ft.partial 3-way separate jit scopes: {name}")
    print(f"{'='*70}")

    csr_offsets, csr_columns, csr_values = to_csr(A_upper)
    _solver = CuDSSSolver(csr_offsets, csr_columns, 0, mtype_id, 1)

    linear_solve = eqx.filter_jit(ft.partial(
        _solver,
        refactorize_signal=jnp.array([0], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    ))
    refactorize_and_linear_solve = eqx.filter_jit(ft.partial(
        _solver,
        refactorize_signal=jnp.array([1], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    ))
    refactorize = eqx.filter_jit(ft.partial(
        _solver,
        refactorize_signal=jnp.array([1], dtype=jnp.int32),
        solve_signal=jnp.array([0], dtype=jnp.int32),
    ))

    b1j = jnp.array(b1, dtype=jnp.float64)
    b2j = jnp.array(b2, dtype=jnp.float64)

    # Factorize only
    _, _ = refactorize(b1j, csr_values)

    # Solve b1
    x1, _ = linear_solve(b1j, csr_values)
    jax.block_until_ready(x1)
    report("refact-only -> solve b1", residual_norm(A_full, x1, b1))

    # Solve b2 (no refactorization)
    x2, _ = linear_solve(b2j, csr_values)
    jax.block_until_ready(x2)
    report("solve-only       -> b2", residual_norm(A_full, x2, b2))

    # Control: refactorize+solve b2
    x3, _ = refactorize_and_linear_solve(b2j, csr_values)
    jax.block_until_ready(x3)
    report("refact+solve     -> b2", residual_norm(A_full, x3, b2))


# =========================================================================
# Test H: Single jit scope, CHANGING CSR values each iteration
#   Mimics real optimization: KKT matrix changes each outer iteration,
#   but within one iteration, refactorize_and_solve and linear_solve
#   receive the SAME perturbed_data.
# =========================================================================
def test_changing_matrix_single_jit(name, A_full, A_upper, mtype_id, n_iters=20):
    print(f"\n{'='*70}")
    print(f"[H] Changing matrix, single jit scope: {name}")
    print(f"{'='*70}")

    csr_offsets, csr_columns, csr_values = to_csr(A_upper)
    n = A_full.shape[0]
    _solver = CuDSSSolver(csr_offsets, csr_columns, 0, mtype_id, 1)

    refactorize_and_solve = ft.partial(
        _solver,
        refactorize_signal=jnp.array([1], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    )
    solve_only = ft.partial(
        _solver,
        refactorize_signal=jnp.array([0], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    )

    # All calls in same jit scope — they share FFI state
    @eqx.filter_jit
    def one_iteration(b_aff, b_cen, b_reg, csr_vals):
        x_aff, _ = refactorize_and_solve(b_aff, csr_vals)
        x_cen, _ = solve_only(b_cen, csr_vals)
        x_reg, _ = solve_only(b_reg, csr_vals)
        return x_aff, x_cen, x_reg

    rng = np.random.default_rng(123)

    # Find diagonal indices in CSR for perturbation
    offsets_np = np.array(csr_offsets)
    columns_np = np.array(csr_columns)
    diag_indices = []
    diag_rows = []
    for i in range(n):
        start, end = offsets_np[i], offsets_np[i + 1]
        for k in range(start, end):
            if columns_np[k] == i:
                diag_indices.append(k)
                diag_rows.append(i)
                break
    diag_indices = jnp.array(diag_indices)

    any_diverged = False
    for it in range(n_iters):
        # Perturb diagonal slightly to simulate changing KKT matrix
        n_diag = len(diag_indices)
        perturbation = jnp.array(rng.standard_normal(n_diag) * 0.1, dtype=jnp.float64)
        perturbed_csr = csr_values.at[diag_indices].add(perturbation)

        # Reconstruct full matrix for residual checking
        A_perturbed = np.array(A_full).copy()
        for idx, di in enumerate(diag_indices):
            row = diag_rows[idx]
            A_perturbed[row, row] += float(perturbation[idx])

        b_aff = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
        b_cen = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
        b_reg = jnp.array(rng.standard_normal(n), dtype=jnp.float64)

        x_aff, x_cen, x_reg = one_iteration(b_aff, b_cen, b_reg, perturbed_csr)
        jax.block_until_ready(x_reg)

        res_aff = residual_norm(A_perturbed, x_aff, b_aff)
        res_cen = residual_norm(A_perturbed, x_cen, b_cen)
        res_reg = residual_norm(A_perturbed, x_reg, b_reg)

        diverged = any(not np.isfinite(r) or r > 1.0
                       for r in [res_aff, res_cen, res_reg])
        tag = ">>> DIVERGED" if diverged else "ok"
        print(f"  iter {it:2d}: affine={res_aff:.2e}  "
              f"central={res_cen:.2e}  regular={res_reg:.2e}  [{tag}]")
        if diverged:
            any_diverged = True

    if any_diverged:
        print(f"  >>> BUG CONFIRMED across {n_iters} iterations")
    else:
        print(f"  All {n_iters} iterations OK")


# =========================================================================
# Test I: Same as H but with fixed matrix (control — should always pass)
# =========================================================================
def test_fixed_matrix_single_jit(name, A_full, A_upper, mtype_id, n_iters=20):
    print(f"\n{'='*70}")
    print(f"[I] Fixed matrix, single jit scope (control): {name}")
    print(f"{'='*70}")

    csr_offsets, csr_columns, csr_values = to_csr(A_upper)
    n = A_full.shape[0]
    _solver = CuDSSSolver(csr_offsets, csr_columns, 0, mtype_id, 1)

    refactorize_and_solve = ft.partial(
        _solver,
        refactorize_signal=jnp.array([1], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    )
    solve_only = ft.partial(
        _solver,
        refactorize_signal=jnp.array([0], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
    )

    @eqx.filter_jit
    def one_iteration(b_aff, b_cen, b_reg, csr_vals):
        x_aff, _ = refactorize_and_solve(b_aff, csr_vals)
        x_cen, _ = solve_only(b_cen, csr_vals)
        x_reg, _ = solve_only(b_reg, csr_vals)
        return x_aff, x_cen, x_reg

    rng = np.random.default_rng(123)

    any_diverged = False
    for it in range(n_iters):
        b_aff = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
        b_cen = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
        b_reg = jnp.array(rng.standard_normal(n), dtype=jnp.float64)

        # Same csr_values every iteration
        x_aff, x_cen, x_reg = one_iteration(b_aff, b_cen, b_reg, csr_values)
        jax.block_until_ready(x_reg)

        res_aff = residual_norm(A_full, x_aff, b_aff)
        res_cen = residual_norm(A_full, x_cen, b_cen)
        res_reg = residual_norm(A_full, x_reg, b_reg)

        diverged = any(not np.isfinite(r) or r > 1.0
                       for r in [res_aff, res_cen, res_reg])
        tag = ">>> DIVERGED" if diverged else "ok"
        print(f"  iter {it:2d}: affine={res_aff:.2e}  "
              f"central={res_cen:.2e}  regular={res_reg:.2e}  [{tag}]")
        if diverged:
            any_diverged = True

    if any_diverged:
        print(f"  >>> BUG CONFIRMED across {n_iters} iterations")
    else:
        print(f"  All {n_iters} iterations OK")


# =========================================================================
# Test J: lax.scan approach — single call site, changing matrix
#   Same setup as test H (which diverges), but using lax.scan so all
#   solver invocations go through ONE custom_call → ONE shared FFI state.
# =========================================================================
def test_changing_matrix_scan(name, A_full, A_upper, mtype_id, n_iters=20):
    print(f"\n{'='*70}")
    print(f"[J] Changing matrix, lax.scan (single call site): {name}")
    print(f"{'='*70}")

    csr_offsets, csr_columns, csr_values = to_csr(A_upper)
    n = A_full.shape[0]
    _solver = CuDSSSolver(csr_offsets, csr_columns, 0, mtype_id, 1)

    solve_on = jnp.array([1], dtype=jnp.int32)

    @eqx.filter_jit
    def one_iteration(b_aff, b_cen, b_reg, csr_vals):
        rhs_stack = jnp.stack([b_aff, b_cen, b_reg])
        signals = jnp.stack([
            jnp.array([1], dtype=jnp.int32),  # refactorize first
            jnp.array([0], dtype=jnp.int32),  # solve only
            jnp.array([0], dtype=jnp.int32),  # solve only
        ])

        def body(carry, inputs):
            rhs, refact_sig = inputs
            x, inertia = _solver(rhs, csr_vals,
                                 refactorize_signal=refact_sig,
                                 solve_signal=solve_on)
            return carry, (x, inertia)

        _, (xs, inertias) = jax.lax.scan(body, None, (rhs_stack, signals))
        return xs[0], xs[1], xs[2]

    # Find diagonal indices for perturbation (same as test H)
    offsets_np = np.array(csr_offsets)
    columns_np = np.array(csr_columns)
    diag_indices = []
    diag_rows = []
    for i in range(n):
        start, end = offsets_np[i], offsets_np[i + 1]
        for k in range(start, end):
            if columns_np[k] == i:
                diag_indices.append(k)
                diag_rows.append(i)
                break
    diag_indices = jnp.array(diag_indices)

    rng = np.random.default_rng(123)  # same seed as test H

    any_diverged = False
    for it in range(n_iters):
        n_diag = len(diag_indices)
        perturbation = jnp.array(rng.standard_normal(n_diag) * 0.1, dtype=jnp.float64)
        perturbed_csr = csr_values.at[diag_indices].add(perturbation)

        A_perturbed = np.array(A_full).copy()
        for idx in range(n_diag):
            row = diag_rows[idx]
            A_perturbed[row, row] += float(perturbation[idx])

        b_aff = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
        b_cen = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
        b_reg = jnp.array(rng.standard_normal(n), dtype=jnp.float64)

        x_aff, x_cen, x_reg = one_iteration(b_aff, b_cen, b_reg, perturbed_csr)
        jax.block_until_ready(x_reg)

        res_aff = residual_norm(A_perturbed, x_aff, b_aff)
        res_cen = residual_norm(A_perturbed, x_cen, b_cen)
        res_reg = residual_norm(A_perturbed, x_reg, b_reg)

        diverged = any(not np.isfinite(r) or r > 1.0
                       for r in [res_aff, res_cen, res_reg])
        tag = ">>> DIVERGED" if diverged else "ok"
        print(f"  iter {it:2d}: affine={res_aff:.2e}  "
              f"central={res_cen:.2e}  regular={res_reg:.2e}  [{tag}]")
        if diverged:
            any_diverged = True

    if any_diverged:
        print(f"  >>> BUG CONFIRMED across {n_iters} iterations")
    else:
        print(f"  All {n_iters} iterations OK")


# =========================================================================
# Test K: Single function called multiple times (NOT scan) within jit
#   Tests whether calling the same Python function with different args
#   produces one or multiple custom_calls in the HLO graph.
# =========================================================================
def test_single_function_multiple_calls(name, A_full, A_upper, mtype_id, n_iters=20):
    print(f"\n{'='*70}")
    print(f"[K] Single function called 3x (no scan), changing matrix: {name}")
    print(f"{'='*70}")

    csr_offsets, csr_columns, csr_values = to_csr(A_upper)
    n = A_full.shape[0]
    _solver = CuDSSSolver(csr_offsets, csr_columns, 0, mtype_id, 1)

    # One function that wraps the solver — called 3 times
    def do_solve(b, csr_vals, refact_sig):
        return _solver(b, csr_vals,
                       refactorize_signal=refact_sig,
                       solve_signal=jnp.array([1], dtype=jnp.int32))

    refact_on  = jnp.array([1], dtype=jnp.int32)
    refact_off = jnp.array([0], dtype=jnp.int32)

    @eqx.filter_jit
    def one_iteration(b_aff, b_cen, b_reg, csr_vals):
        x_aff, _ = do_solve(b_aff, csr_vals, refact_on)
        x_cen, _ = do_solve(b_cen, csr_vals, refact_off)
        x_reg, _ = do_solve(b_reg, csr_vals, refact_off)
        return x_aff, x_cen, x_reg

    # Find diagonal indices for perturbation
    offsets_np = np.array(csr_offsets)
    columns_np = np.array(csr_columns)
    diag_indices = []
    diag_rows = []
    for i in range(n):
        start, end = offsets_np[i], offsets_np[i + 1]
        for k in range(start, end):
            if columns_np[k] == i:
                diag_indices.append(k)
                diag_rows.append(i)
                break
    diag_indices = jnp.array(diag_indices)

    rng = np.random.default_rng(123)

    any_diverged = False
    for it in range(n_iters):
        n_diag = len(diag_indices)
        perturbation = jnp.array(rng.standard_normal(n_diag) * 0.1, dtype=jnp.float64)
        perturbed_csr = csr_values.at[diag_indices].add(perturbation)

        A_perturbed = np.array(A_full).copy()
        for idx in range(n_diag):
            row = diag_rows[idx]
            A_perturbed[row, row] += float(perturbation[idx])

        b_aff = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
        b_cen = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
        b_reg = jnp.array(rng.standard_normal(n), dtype=jnp.float64)

        x_aff, x_cen, x_reg = one_iteration(b_aff, b_cen, b_reg, perturbed_csr)
        jax.block_until_ready(x_reg)

        res_aff = residual_norm(A_perturbed, x_aff, b_aff)
        res_cen = residual_norm(A_perturbed, x_cen, b_cen)
        res_reg = residual_norm(A_perturbed, x_reg, b_reg)

        diverged = any(not np.isfinite(r) or r > 1.0
                       for r in [res_aff, res_cen, res_reg])
        tag = ">>> DIVERGED" if diverged else "ok"
        print(f"  iter {it:2d}: affine={res_aff:.2e}  "
              f"central={res_cen:.2e}  regular={res_reg:.2e}  [{tag}]")
        if diverged:
            any_diverged = True

    if any_diverged:
        print(f"  >>> BUG CONFIRMED across {n_iters} iterations")
    else:
        print(f"  All {n_iters} iterations OK")


if __name__ == "__main__":
    print("="*70)
    print("Reproducer: IR divergence on re-solve without refactorization")
    print("="*70)

    # KKT (indefinite) systems only
    systems = []
    for n_x, n_c in [(6, 3), (20, 10)]:
        K, Ku, b1, b2 = make_kkt_system(n_x, n_c, seed=42)
        systems.append((f"KKT n_x={n_x} n_c={n_c}", K, Ku, b1, b2, 1))

    # Test K: single function called 3x (no scan)
    for name, A, Au, b1, b2, mt in systems:
        test_single_function_multiple_calls(name, A, Au, mt, n_iters=20)

    # Test J: changing matrix, lax.scan single call site (FIX)
    for name, A, Au, b1, b2, mt in systems:
        test_changing_matrix_scan(name, A, Au, mt, n_iters=20)

    print("\n" + "="*70)
    print("Done. If 'BUG CONFIRMED' appears above, IR diverged on re-solve.")
    print("="*70)

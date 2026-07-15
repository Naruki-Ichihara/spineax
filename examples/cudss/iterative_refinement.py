"""
Example: JAX-native iterative refinement via solve(token, b, ir_nsteps=...).

cuDSS's internal IR is permanently OFF in spineax: its refinement SpMV reads
CSR pointers captured at earlier phase calls, which under the zero-copy
design are dead XLA temporaries by solve time for batched systems (see
_ir_off in solver.py). Instead, ir_nsteps is a STATIC argument on solve()
that unrolls same-precision Richardson refinement into the jaxpr:

    x = A^-1 b;   then ir_nsteps times:   x += A^-1 (b - A x)

with the residual computed by the token's own matvec and every A^-1 an
extra SOLVE against the existing factors — no refactorization. This is the
same iteration cuDSS runs internally (accuracy parity verified on these
systems), but it is batch-safe and composes with jit/vmap/grad like every
other spineax op.

WHERE IR HELPS: refinement repairs FACTORIZATION error, not conditioning.
On a backward-stable factorization the residual already sits at the
same-precision floor and extra steps do nothing. The system below is the
interesting case — a symmetric-indefinite KKT matrix whose badly scaled H
block makes the static-pivoted LDL^T factors inaccurate, so one refinement
step buys ~10 digits in both residual and solution.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from spineax import cudss

rng = np.random.default_rng(3)


def make_kkt(n_x=150, n_c=50, h_min=1e-12):
    """KKT [[H, J^T], [J, 0]] with H eigenvalues in [h_min, 1]: the tiny
    pivots make the LDL^T factors inaccurate — refinement has real work."""
    Q, _ = np.linalg.qr(rng.standard_normal((n_x, n_x)))
    H = (Q * np.logspace(np.log10(h_min), 0, n_x)) @ Q.T
    J = rng.standard_normal((n_c, n_x))
    n = n_x + n_c
    K = np.zeros((n, n))
    K[:n_x, :n_x] = (H + H.T) / 2
    K[n_x:, :n_x] = J
    K[:n_x, n_x:] = J.T
    return K


def upper_csr(A):
    mask = np.triu(np.ones(A.shape, bool))
    offs = jnp.asarray(np.concatenate([[0], np.cumsum(mask.sum(1))]), jnp.int32)
    cols = jnp.asarray(np.nonzero(mask)[1], jnp.int32)
    return jnp.asarray(A[mask]), offs, cols


def test_refinement_recovers_digits():
    """Forward error and residual vs ir_nsteps, single and block-batch
    paths. The old cuDSS-side IR hard-crashed on the batch path; the
    JAX-side refinement is just arithmetic on the one block-diagonal
    system, so the batch refines exactly like the single system."""
    print("=" * 74)
    print("TEST 1: error vs ir_nsteps on an ill-conditioned KKT (LDL^T, mtype 1)")
    print("=" * 74)

    K = make_kkt()
    x_true = rng.standard_normal(K.shape[0])
    b = jnp.asarray(K @ x_true)
    vals, offs, cols = upper_csr(K)
    print(f"  n={K.shape[0]}, cond(K)={np.linalg.cond(K):.1e}")

    token = cudss.analyze(vals, offs, cols, mtype_id=1, mview_id=1)
    token = cudss.factorize(token, vals)

    # one block-diagonal entry holding 4 copies of the same system
    batch = 4
    vals_b, b_b = jnp.stack([vals] * batch), jnp.stack([b] * batch)
    btoken = cudss.analyze(vals_b, offs, cols, mtype_id=1, mview_id=1)
    btoken = cudss.factorize(btoken, vals_b)

    def fwd(x):
        return np.linalg.norm(np.asarray(x) - x_true) / np.linalg.norm(x_true)

    def res(x):
        return np.linalg.norm(K @ np.asarray(x) - np.asarray(b)) / np.linalg.norm(np.asarray(b))

    print(f"\n  {'nsteps':>6}  {'fwd err':>11}  {'residual':>11}"
          f"  {'batch fwd':>11}  {'batch resid':>11}")
    resids = {}
    for nsteps in range(4):
        x = cudss.solve(token, b, ir_nsteps=nsteps)
        xb = cudss.solve(btoken, b_b, ir_nsteps=nsteps)
        resids[nsteps] = res(x)
        print(f"  {nsteps:>6}  {fwd(x):>11.2e}  {res(x):>11.2e}"
              f"  {max(fwd(xb[i]) for i in range(batch)):>11.2e}"
              f"  {max(res(xb[i]) for i in range(batch)):>11.2e}")

    assert resids[1] < resids[0] * 1e-6, "one IR step should recover ~10 digits"
    cudss.release(token)
    cudss.release(btoken)
    print()


def test_static_unrolled_in_jit():
    """ir_nsteps is static: each value unrolls its own refinement chain in
    the jaxpr, so two solves in ONE jit with different ir_nsteps differ."""
    print("=" * 74)
    print("TEST 2: two solves in one jit, different ir_nsteps")
    print("=" * 74)

    K = make_kkt()
    b = jnp.asarray(K @ rng.standard_normal(K.shape[0]))
    vals, offs, cols = upper_csr(K)

    @jax.jit
    def solve_both(vals, b):
        token = cudss.analyze(vals, offs, cols, mtype_id=1, mview_id=1)
        token = cudss.factorize(token, vals)
        return cudss.solve(token, b, ir_nsteps=0), cudss.solve(token, b, ir_nsteps=2)

    x_0, x_2 = solve_both(vals, b)

    def res(x):
        return float(jnp.linalg.norm(K @ x - b) / jnp.linalg.norm(b))

    print(f"  ir_nsteps=0  |  relative residual = {res(x_0):.2e}")
    print(f"  ir_nsteps=2  |  relative residual = {res(x_2):.2e}")
    assert res(x_2) < res(x_0) * 1e-6
    print("  PASS: refinement unrolled and effective inside jit")
    print()


def test_ir_composes_with_grad():
    """Refined solves differentiate like unrefined ones: the refinement sits
    inside the non-differentiated solver callables, gradients come from the
    implicit function theorem, and the adjoint solves are themselves REFINED
    — so on this system the ir=2 gradient is the trustworthy one."""
    print("=" * 74)
    print("TEST 3: grad through a refined solve")
    print("=" * 74)

    K = make_kkt()
    b = jnp.asarray(K @ rng.standard_normal(K.shape[0]))
    vals, offs, cols = upper_csr(K)

    def loss(vals, nsteps):
        token = cudss.analyze(vals, offs, cols, mtype_id=1, mview_id=1)
        token = cudss.factorize(token, vals)
        return jnp.sum(cudss.solve(token, b, ir_nsteps=nsteps))

    g_0 = jax.grad(loss)(vals, 0)
    g_2 = jax.grad(loss)(vals, 2)
    rel = float(jnp.linalg.norm(g_2 - g_0) / jnp.linalg.norm(g_2))
    print(f"  ||grad(ir=2) - grad(ir=0)|| / ||grad(ir=2)|| = {rel:.2e}")
    print("  (the difference IS the refinement of the forward/adjoint solves)")
    print()


if __name__ == "__main__":
    test_refinement_recovers_digits()
    test_static_unrolled_in_jit()
    test_ir_composes_with_grad()
    print("All iterative refinement examples completed.")

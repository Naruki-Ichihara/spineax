"""Tests for the token-based factorization API (docs/token_design.md)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from spineax.cudss import tokens as tk


def _require_gpu():
    if not jax.devices("gpu"):
        pytest.skip("CUDA device required for cuDSS tests")


def _sym_system(n=50, dtype=jnp.float64, seed=0, shift=None):
    """Random symmetric matrix (upper-triangle CSR) + dense reference.

    With shift=None the matrix is diagonally-dominant PD; a scalar shift
    controls the spectrum for indefinite tests.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A = A + A.T + (n if shift is None else shift) * np.eye(n)
    upper = np.triu(A)
    # CSR of the dense upper triangle: offsets/columns of explicit entries
    mask = np.triu(np.ones((n, n), dtype=bool))
    columns = np.nonzero(mask)[1].astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(mask.sum(axis=1))]).astype(np.int32)
    values = upper[mask]
    return (
        jnp.asarray(values, dtype=dtype),
        jnp.asarray(offsets),
        jnp.asarray(columns),
        jnp.asarray(A, dtype=dtype),
    )


def _general_complex_system(n=30, dtype=jnp.complex128, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = A + n * np.eye(n)
    columns = np.tile(np.arange(n, dtype=np.int32), n)
    offsets = (np.arange(n + 1, dtype=np.int32) * n)
    values = A.reshape(-1)
    return (
        jnp.asarray(values, dtype=dtype),
        jnp.asarray(offsets),
        jnp.asarray(columns),
        jnp.asarray(A, dtype=dtype),
    )


_TOL = {
    jnp.float32: 1e-4,
    jnp.float64: 1e-10,
    jnp.complex64: 1e-3,
    jnp.complex128: 1e-10,
}


def _rel_err(A, x, b):
    A, x, b = np.asarray(A), np.asarray(x), np.asarray(b)
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)


# correctness ==================================================================
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_analyze_factorize_solve(dtype):
    _require_gpu()
    values, offsets, columns, A = _sym_system(dtype=dtype)
    b = jnp.asarray(np.random.default_rng(2).standard_normal(A.shape[0]), dtype=dtype)

    token = tk.analyze(values, offsets, columns, mtype_id=1, mview_id=1)
    assert token.kind == "single"
    assert token.n == A.shape[0]

    token = tk.factorize(token, values)
    inertia = tk.inertia(tk.query(token))
    np.testing.assert_array_equal(np.asarray(inertia), [A.shape[0], 0])  # PD

    x = tk.solve(token, b)
    assert _rel_err(A, x, b) < _TOL[dtype]


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
def test_complex_general(dtype):
    _require_gpu()
    values, offsets, columns, A = _general_complex_system(dtype=dtype)
    b = jnp.asarray(
        np.random.default_rng(3).standard_normal(A.shape[0])
        + 1j * np.random.default_rng(4).standard_normal(A.shape[0]),
        dtype=dtype,
    )
    token = tk.analyze(values, offsets, columns, mtype_id=0, mview_id=0)
    token = tk.factorize(token, values)
    x = tk.solve(token, b)
    assert _rel_err(A, x, b) < _TOL[dtype]


def test_jit_full_chain():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(5).standard_normal(A.shape[0]))

    @jax.jit
    def full(values, b):
        t = tk.analyze(values, offsets, columns, mtype_id=1, mview_id=1)
        t = tk.factorize(t, values)
        return tk.solve(t, b), tk.inertia(tk.query(t))

    x, inertia = full(values, b)
    assert _rel_err(A, x, b) < 1e-10
    np.testing.assert_array_equal(np.asarray(inertia), [A.shape[0], 0])


def test_refactorize():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(6).standard_normal(A.shape[0]))

    token = tk.analyze(values, offsets, columns)
    token = tk.factorize(token, values)
    token = tk.refactorize(token, values * 2.0)
    x = tk.solve(token, b)
    assert _rel_err(2 * np.asarray(A), x, b) < 1e-10
    inertia = tk.inertia(tk.query(token))
    np.testing.assert_array_equal(np.asarray(inertia), [A.shape[0], 0])


# multi-RHS and vmap ===========================================================
def test_multirhs_stack():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    B = jnp.asarray(np.random.default_rng(7).standard_normal((6, A.shape[0])))

    token = tk.analyze(values, offsets, columns)
    token = tk.factorize(token, values)
    X = tk.solve(token, B)  # one multi-RHS SOLVE
    for i in range(B.shape[0]):
        assert _rel_err(A, X[i], B[i]) < 1e-10


def test_vmap_multirhs_fast_path():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    B = jnp.asarray(np.random.default_rng(8).standard_normal((6, A.shape[0])))

    token = tk.analyze(values, offsets, columns)
    token = tk.factorize(token, values)
    X_vmap = jax.vmap(lambda b: tk.solve(token, b))(B)
    X_direct = tk.solve(token, B)
    np.testing.assert_allclose(np.asarray(X_vmap), np.asarray(X_direct), rtol=1e-14)


def test_vmap_batch_is_block_diagonal(monkeypatch):
    _require_gpu()
    # capacity is re-read from the env each call; raise it so LRU eviction
    # cannot mask the entry count this test asserts on
    monkeypatch.setenv("SPINEAX_FACTOR_CACHE", "64")
    values, offsets, columns, A = _sym_system()
    vals_batch = jnp.stack([values, values * 3.0])
    b = jnp.asarray(np.random.default_rng(9).standard_normal(A.shape[0]))
    b_batch = jnp.stack([b, b])

    size_before = tk.registry_size()
    tokens = jax.vmap(lambda v: tk.analyze(v, offsets, columns))(vals_batch)
    # ONE block-diagonal entry, its id broadcast across the batch
    assert tokens.id.shape == (2, 1)
    ids = np.asarray(tokens.id)
    assert ids[0, 0] == ids[1, 0]
    assert tk.registry_size() - size_before == 1

    tokens = jax.vmap(tk.factorize)(tokens, vals_batch)
    inertias = tk.inertia(tk.query(tokens), batch_size=2)
    np.testing.assert_array_equal(np.asarray(inertias), [[A.shape[0], 0]] * 2)
    xs = jax.vmap(tk.solve)(tokens, b_batch)
    assert _rel_err(A, xs[0], b) < 1e-10
    assert _rel_err(3 * np.asarray(A), xs[1], b) < 1e-10

    # vmap-minted tokens work eagerly (outside vmap) too: batch-shaped args
    tokens2 = tk.factorize(tokens, vals_batch * 2.0)
    xs2 = tk.solve(tokens2, b_batch)
    assert _rel_err(2 * np.asarray(A), xs2[0], b) < 1e-10
    assert _rel_err(6 * np.asarray(A), xs2[1], b) < 1e-10


def test_explicit_batch_door():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    n = A.shape[0]
    vals = jnp.stack([values, values * 2.0, values * 5.0])
    token = tk.analyze(vals, offsets, columns)
    assert token.kind == "pbatch"
    assert token.batch_size == 3
    token = tk.factorize(token, vals)
    inertia = tk.inertia(tk.query(token), batch_size=3)
    np.testing.assert_array_equal(np.asarray(inertia), [[n, 0]] * 3)

    B = jnp.asarray(np.random.default_rng(20).standard_normal((3, n)))
    X = tk.solve(token, B)
    for i, s in enumerate([1.0, 2.0, 5.0]):
        assert _rel_err(s * np.asarray(A), X[i], B[i]) < 1e-10

    token = tk.refactorize(token, vals * 3.0)
    X = tk.solve(token, B)
    for i, s in enumerate([3.0, 6.0, 15.0]):
        assert _rel_err(s * np.asarray(A), X[i], B[i]) < 1e-10


def test_vmap_batched_patterns():
    _require_gpu()
    # two systems, SAME shapes but DIFFERENT sparsity patterns (the "general
    # case": the whole block is analyzed, no shared structure)
    n = 4
    offs = jnp.asarray([0, 2, 3, 4, 5], dtype=jnp.int32)  # 5 nnz, upper view
    cols_a = jnp.asarray([0, 1, 1, 2, 3], dtype=jnp.int32)  # diag + (0,1)
    cols_b = jnp.asarray([0, 2, 1, 2, 3], dtype=jnp.int32)  # diag + (0,2)
    vals = jnp.asarray([4.0, 1.0, 3.0, 5.0, 2.0], dtype=jnp.float64)

    def dense(cols):
        A = np.zeros((n, n))
        v = np.asarray(vals)
        k = 0
        for i in range(n):
            for j in range(int(offs[i]), int(offs[i + 1])):
                A[i, int(cols[j])] = v[k]
                k += 1
        return A + A.T - np.diag(np.diag(A))

    cols_batch = jnp.stack([cols_a, cols_b])
    tokens = jax.vmap(lambda c: tk.analyze(vals, offs, c))(cols_batch)
    tokens = jax.vmap(lambda t: tk.factorize(t, vals))(tokens)
    b = jnp.asarray(np.random.default_rng(21).standard_normal(n))
    xs = jax.vmap(lambda t: tk.solve(t, b))(tokens)
    assert _rel_err(dense(cols_a), xs[0], b) < 1e-10
    assert _rel_err(dense(cols_b), xs[1], b) < 1e-10


def test_vmap_factorize_unbatched_token_raises():
    _require_gpu()
    values, offsets, columns, _ = _sym_system()
    token = tk.analyze(values, offsets, columns)
    vals_batch = jnp.stack([values, values * 2.0])
    with pytest.raises(ValueError, match="vmap\\(analyze\\)"):
        jax.vmap(lambda v: tk.factorize(token, v))(vals_batch)


# control flow and autodiff ====================================================
def test_lax_cond_refactorize():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(10).standard_normal(A.shape[0]))

    @jax.jit
    def step(token, vals, b, do_refactor):
        token = jax.lax.cond(
            do_refactor,
            lambda t: tk.refactorize(t, vals),
            lambda t: t,
            token,
        )
        return tk.solve(token, b), token

    token = tk.analyze(values, offsets, columns)
    token = tk.factorize(token, values)

    x, token = step(token, values * 4.0, b, False)
    assert _rel_err(A, x, b) < 1e-10  # skip branch: old factors
    x, token = step(token, values * 4.0, b, True)
    assert _rel_err(4 * np.asarray(A), x, b) < 1e-10  # take branch: new factors


def test_custom_vjp_adjoint_reuse(monkeypatch):
    _require_gpu()
    # capacity is re-read from the env each call; raise it so LRU eviction
    # cannot mask the entry count this test asserts on
    monkeypatch.setenv("SPINEAX_FACTOR_CACHE", "64")
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(11).standard_normal(A.shape[0]))

    @jax.custom_vjp
    def token_solve(vals, b):
        t = tk.analyze(vals, offsets, columns)
        t = tk.factorize(t, vals)
        return tk.solve(t, b)

    def fwd(vals, b):
        t = tk.analyze(vals, offsets, columns)
        t = tk.factorize(t, vals)
        return tk.solve(t, b), t  # token threads through residuals

    def bwd(t, v):
        # symmetric: lambda = A^-T v = A^-1 v, reusing the forward factors
        return (None, tk.solve(t, v))

    token_solve.defvjp(fwd, bwd)

    size_before = tk.registry_size()
    grad_b = jax.jit(jax.grad(lambda v, b: token_solve(v, b).sum(), argnums=1))(values, b)
    expected = np.linalg.solve(np.asarray(A), np.ones(A.shape[0]))
    np.testing.assert_allclose(np.asarray(grad_b), expected, rtol=1e-10)
    # forward+backward share ONE factorization
    assert tk.registry_size() - size_before == 1


# inertia ======================================================================
def test_inertia_indefinite():
    _require_gpu()
    # shift=0: random symmetric, genuinely indefinite
    values, offsets, columns, A = _sym_system(n=40, shift=0.0, seed=12)
    eigs = np.linalg.eigvalsh(np.asarray(A))
    expected = [int((eigs > 0).sum()), int((eigs < 0).sum())]

    token = tk.analyze(values, offsets, columns)
    token = tk.factorize(token, values)
    inertia = tk.inertia(tk.query(token))
    np.testing.assert_array_equal(np.asarray(inertia), expected)


# error handling ===============================================================
def test_solve_before_factorize_raises():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(15).standard_normal(A.shape[0]))
    token = tk.analyze(values, offsets, columns)
    with pytest.raises(Exception, match="requires a factorized token"):
        tk.solve(token, b).block_until_ready()


def test_refactorize_before_factorize_raises():
    _require_gpu()
    values, offsets, columns, _ = _sym_system()
    token = tk.analyze(values, offsets, columns)
    with pytest.raises(Exception, match="requires a factorized token"):
        tk.refactorize(token, values).id.block_until_ready()


def test_values_size_mismatch_raises():
    _require_gpu()
    values, offsets, columns, _ = _sym_system()
    token = tk.analyze(values, offsets, columns)
    with pytest.raises(ValueError, match="nnz"):
        tk.factorize(token, values[:-1])


def test_dtype_mismatch_raises():
    _require_gpu()
    values, offsets, columns, _ = _sym_system(dtype=jnp.float64)
    token = tk.analyze(values, offsets, columns)
    with pytest.raises(ValueError, match="dtype"):
        tk.factorize(token, values.astype(jnp.float32))


# ported from the legacy CuDSSSolver/CuDSSSolverRE suite ======================
def _legacy_base_system(dtype=jnp.float32):
    M1 = jnp.array(
        [
            [4.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
        ],
        dtype=dtype,
    )
    b1 = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0], dtype=dtype)
    m1 = M1 + M1.T - jnp.diag(M1) * jnp.eye(M1.shape[0], dtype=dtype)
    true_x1 = jnp.linalg.solve(m1, b1)
    return M1, b1, m1, true_x1


def _csr_of(M):
    import jax.experimental.sparse as jsparse

    bcsr = jsparse.BCSR.fromdense(M)
    return bcsr.indptr, bcsr.indices, bcsr.data


def test_legacy_composability():
    _require_gpu()
    M1, b1, m1, true_x1 = _legacy_base_system(jnp.float32)
    M2 = M1 * 0.9
    b2 = b1 * 1.1
    m2 = M2 + M2.T - jnp.diag(M2) * jnp.eye(M2.shape[0], dtype=M2.dtype)
    true_x2 = jnp.linalg.solve(m2, b2)

    offsets, columns, values1 = _csr_of(M1)
    _, _, values2 = _csr_of(M2)
    values = jnp.vstack([values1, values2])
    b = jnp.vstack([b1, b2])

    def token_solve(values, b):
        t = tk.analyze(values, offsets, columns, mtype_id=1, mview_id=1)
        t = tk.factorize(t, values)
        return tk.solve(t, b)

    x1 = token_solve(values[0], b[0])
    x2 = jax.jit(jax.vmap(token_solve))(values, b)

    assert jnp.allclose(x1, true_x1, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(x2, jnp.stack([true_x1, true_x2]), rtol=1e-5, atol=1e-5)


@pytest.mark.xfail(
    reason="nested vmap of token ops is not supported: the custom_vmap rules "
    "emit plain ffi_calls which have no batching rule for an outer vmap. "
    "Flatten nested batches into one block-diagonal batch instead.",
    strict=True,
)
def test_legacy_nested_vmap():
    _require_gpu()
    M1, b1, _, _ = _legacy_base_system(jnp.float32)
    offsets, columns, values1 = _csr_of(M1)
    values = jnp.stack([jnp.stack([values1, values1])] * 2)
    b = jnp.stack([jnp.stack([b1, b1])] * 2)

    def token_solve(values, b):
        t = tk.analyze(values, offsets, columns, mtype_id=1, mview_id=1)
        t = tk.factorize(t, values)
        return tk.solve(t, b)

    jax.jit(jax.vmap(jax.vmap(token_solve)))(values, b)


@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
)
def test_legacy_datatypes(dtype):
    _require_gpu()
    _, b1, m1, true_x1 = _legacy_base_system(dtype)
    offsets, columns, values = _csr_of(m1)  # full symmetric matrix, mview full

    token = tk.analyze(values, offsets, columns, mtype_id=1, mview_id=0)
    token = tk.factorize(token, values)
    x = tk.solve(token, b1)

    assert x.shape == b1.shape
    assert jnp.allclose(x, true_x1, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mtype_id", list(range(5)))
def test_legacy_solver_types(mtype_id):
    _require_gpu()
    _, b1, m1, true_x1 = _legacy_base_system(jnp.float32)
    offsets, columns, values = _csr_of(m1)

    token = tk.analyze(values, offsets, columns, mtype_id=mtype_id, mview_id=0)
    token = tk.factorize(token, values)
    x = tk.solve(token, b1)

    assert x.shape == b1.shape
    assert jnp.allclose(x, true_x1, rtol=1e-5, atol=1e-5)


def test_query():
    _require_gpu()
    # port of the CuDSSSolverRE "return everything" test
    M1, b1, _, true_x1 = _legacy_base_system(jnp.float32)
    offsets, columns, values = _csr_of(M1)  # upper triangle, mview upper
    n = b1.shape[0]

    token = tk.analyze(values, offsets, columns, mtype_id=1, mview_id=1)
    token = tk.factorize(token, values)
    x = tk.solve(token, b1)
    assert jnp.allclose(x, true_x1, rtol=1e-5, atol=1e-5)

    out = tk.query(token)
    assert out["lu_nnz"][0] > 0
    assert out["npivots"][0] >= 0
    assert out["inertia"].shape == (2,)
    for key in ("perm_reorder_row", "perm_reorder_col", "perm_row",
                "perm_col", "perm_matching", "scale_row", "scale_col"):
        assert out[key].shape == (n,), key
    assert out["diag"].shape == (n,)
    assert out["diag"].dtype == jnp.float32
    assert out["nd_partition_tree"].shape[0] > 0
    assert out["nsuperpanels"].shape == (1,)
    assert out["schur_shape"].shape == (2,)

    # inertia() over the same data agrees with a fresh query after refactorize
    inr1 = tk.inertia(out)
    token = tk.factorize(token, values)
    inr2 = tk.inertia(tk.query(token))
    np.testing.assert_array_equal(np.asarray(inr1), np.asarray(inr2))


def test_query_before_factorize_raises():
    _require_gpu()
    values, offsets, columns, _ = _sym_system()
    token = tk.analyze(values, offsets, columns)
    with pytest.raises(Exception, match="requires a factorized token"):
        jax.block_until_ready(tk.query(token))


# lifetime =====================================================================
def test_release():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(16).standard_normal(A.shape[0]))
    token = tk.analyze(values, offsets, columns)
    token = tk.factorize(token, values)
    tk.solve(token, b).block_until_ready()

    assert tk.release(token) is True
    assert tk.release(token) is False  # second release is a no-op
    with pytest.raises(Exception, match="unknown or evicted"):
        tk.solve(token, b).block_until_ready()


def test_lru_eviction():
    _require_gpu()
    values, offsets, columns, _ = _sym_system()
    b = jnp.asarray(np.random.default_rng(17).standard_normal(offsets.shape[0] - 1))
    cap = tk.cache_capacity()

    victim = tk.analyze(values, offsets, columns)
    victim = tk.factorize(victim, values)
    # cap fresh entries push the untouched victim out
    for _ in range(cap):
        tk.analyze(values, offsets, columns).id.block_until_ready()
    assert tk.registry_size() <= cap
    with pytest.raises(Exception, match="unknown or evicted"):
        tk.solve(victim, b).block_until_ready()

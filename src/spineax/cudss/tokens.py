"""Token-based persistent factorization (docs/token_design.md).

Four free functions mapping 1:1 onto cuDSS phases, plus one data door:

    token = analyze(csr_values, csr_offsets, csr_columns, ...)   # ANALYSIS
    token = factorize(token, csr_values, ...)                    # FACTORIZATION
    token = refactorize(token, csr_values, ...)                  # REFACTORIZATION
    x     = solve(token, b, ...)                                 # SOLVE
    data  = query(token)          # every cuDSS data item, unconditionally
    inr   = inertia(data, batch_size)  # per-block [pos, neg] from that data

The ``FactorToken`` is a pytree with one traced ``int32`` leaf (the registry
id) plus static metadata. The traced id gives XLA a data dependency ordering
every call after the one that produced its token, threads through
``custom_vjp`` residuals so an adjoint can reuse the forward factorization, and
passes unchanged through ``jit``/``grad``/``vmap``/``lax.scan``/``lax.cond``.

Batching is always block-diagonal: a batch of B systems IS one bigger sparse
system, and a single system is the B=1 special case, all served by the
``pbatch_solve`` native module. Two doors to a batch:

- ``jax.vmap`` over ``analyze`` (batched values, shared or batched pattern):
  ONE block-diagonal registry entry; the token id is broadcast across the
  batch, and vmapped ``factorize``/``refactorize``/``solve`` collapse to one
  block-diagonal cuDSS call each.
- explicit shapes: ``analyze`` with 2-D ``(B, nnz)`` values mints the same
  kind of entry eagerly, with ``batch_size=B`` recorded in the token statics.

Factorizations live in a process-global LRU cache on the C++ side (capacity
from ``SPINEAX_FACTOR_CACHE``, default 8); using an evicted token raises at
runtime. ``release(token)`` frees an entry eagerly (outside jit only).

Inertia (per-block positive/negative eigenvalue counts from the LDL^T
diagonal) is a property of the factorization, so it is returned by
``factorize`` / ``refactorize`` — an IPM can inspect it before paying for a
solve.
"""

import dataclasses
from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

# Force JAX to initialize the CUDA context before importing the native module.
jax.devices()

try:
    from spineax import pbatch_solve as _ps  # native nanobind module
except ImportError as e:
    raise ImportError(
        "spineax.cudss.tokens requires the pbatch_solve native module "
        "(built when BUILD_PBATCH_SOLVE is enabled): the token API treats a "
        "single solve as the batch_size=1 case of the block-diagonal batch "
        "construction, which lives there."
    ) from e

# registrations ================================================================
_SUFFIXES = ("f32", "f64", "c64", "c128")
_DTYPE_SUFFIX = {
    jnp.dtype(jnp.float32): "f32",
    jnp.dtype(jnp.float64): "f64",
    jnp.dtype(jnp.complex64): "c64",
    jnp.dtype(jnp.complex128): "c128",
}

# Plain FFI handlers (no instantiate/State): register_ffi_target only.
for _s in _SUFFIXES:
    _handlers = getattr(_ps, f"token_handlers_{_s}")()
    for _op in ("analyze", "factorize", "refactorize", "solve", "query"):
        jax.ffi.register_ffi_target(
            f"spineax_token_{_op}_{_s}", _handlers[_op], platform="CUDA"
        )


def _suffix(dtype) -> str:
    try:
        return _DTYPE_SUFFIX[jnp.dtype(dtype)]
    except KeyError:
        raise ValueError(
            f"spineax tokens: unsupported dtype {dtype} "
            f"(supported: f32, f64, c64, c128)"
        ) from None


def compute_inertia_from_diag(diag, batch_size, matrix_dim):
    """Per-block [positive, negative] counts from the LDL^T diagonal.

    cuDSS >= 0.8 returns CUDSS_DATA_DIAG **in input order** ("the original
    matrix order, taking into account all permutations" — cuDSS docs; changed
    in the 0.8 release notes from the internal order of <= 0.7). Input order
    is block-grouped, so the reshape splits cleanly per block and NO
    permutation may be applied. The legacy ``diag[argsort(perm)]`` reorder
    was correct for cuDSS <= 0.7 and silently misattributes blocks in
    heterogeneous batches on 0.8 (found via the newton_ipm_loop example;
    verified against the docs with a distinct-diagonal experiment).
    """
    out = diag.reshape([batch_size, matrix_dim])

    # cuDSS pivoting threshold seems to be 1e-13. everything above this on
    # plus or minus side seems to reliably indicate that particular inertia value.
    threshold = 1e-13
    positive = jnp.sum(out >= threshold, axis=1)
    negative = jnp.sum(out <= -threshold, axis=1)

    return jnp.stack([positive, negative], axis=1, dtype=jnp.int32)


# the token ====================================================================
class FactorToken(eqx.Module):
    """Handle to a cached cuDSS (block-diagonal) factorization.

    ``id`` is the only traced leaf: dataflow ordering, vjp residuals and vmap
    batching all operate on it. It is ``int32[1]`` for a token minted outside
    vmap, or ``int32[B, 1]`` (B equal copies of one entry id) for a token
    minted by ``vmap(analyze)``. The static fields resolve dispatch at trace
    time and make the token self-describing — including for a future lineax
    adapter, where a FactorToken is the solver state.
    """

    id: Array                                # int32[1] or int32[B, 1] — traced
    kind: str = eqx.field(static=True)       # "single" | "pbatch" (descriptive)
    dtype: jnp.dtype = eqx.field(static=True)
    n: int = eqx.field(static=True)          # BLOCK dimension (one system)
    nnz: int = eqx.field(static=True)        # BLOCK nnz (one system)
    batch_size: int = eqx.field(static=True) # 1 unless minted by the explicit door
    mtype_id: int = eqx.field(static=True)
    mview_id: int = eqx.field(static=True)
    device_id: int = eqx.field(static=True)


def _with_id(token: FactorToken, new_id: Array) -> FactorToken:
    return dataclasses.replace(token, id=new_id)


def _as_ir(ir_nsteps) -> Array:
    """Normalize the optional iterative-refinement argument to int32[1].

    Default is 0: IR is numerically wrong for LDL^T (dev-branch finding), so it
    is strictly opt-in.
    """
    if ir_nsteps is None:
        return jnp.zeros((1,), dtype=jnp.int32)
    return jnp.asarray(ir_nsteps, dtype=jnp.int32).reshape((1,))


def _batch_of(token: FactorToken) -> int:
    """Effective batch size: explicit-door static, or vmap-minted id shape."""
    if token.id.ndim == 2:
        return token.id.shape[0]
    return token.batch_size


# raw FFI calls ================================================================
def _ffi_analyze(values, offsets, columns, *, batch_size, pattern_batched,
                 device_id, mtype_id, mview_id):
    fn = jax.ffi.ffi_call(
        f"spineax_token_analyze_{_suffix(values.dtype)}",
        jax.ShapeDtypeStruct((1,), jnp.int32),
        has_side_effect=True,
    )
    return fn(
        values,
        offsets.astype(jnp.int32),
        columns.astype(jnp.int32),
        batch_size=batch_size,
        pattern_batched=pattern_batched,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
    )


def _ffi_numeric(token_id, values, ir, *, op):
    fn = jax.ffi.ffi_call(
        f"spineax_token_{op}_{_suffix(values.dtype)}",
        jax.ShapeDtypeStruct(token_id.shape, jnp.int32),  # token (same ids)
        has_side_effect=True,
    )
    return fn(token_id, values, ir)


def _ffi_solve(token_id, b, ir):
    fn = jax.ffi.ffi_call(
        f"spineax_token_solve_{_suffix(b.dtype)}",
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        has_side_effect=True,
    )
    return fn(token_id, b, ir)


# vmap-aware wrappers over the single (B=1) view ===============================
# Static config is baked via lru_cache closures so custom_vmap only ever sees
# array arguments. The rules are where vmap becomes block-diagonal batching.

@lru_cache(maxsize=None)
def _make_analyze(suffix, device_id, mtype_id, mview_id):
    del suffix  # dispatch is by values dtype; suffix only keys the cache

    @jax.custom_batching.custom_vmap
    def analyze_id(csr_values, csr_offsets, csr_columns):
        return _ffi_analyze(csr_values, csr_offsets, csr_columns,
                            batch_size=1, pattern_batched=0,
                            device_id=device_id, mtype_id=mtype_id,
                            mview_id=mview_id)

    @analyze_id.def_vmap
    def _(axis_size, in_batched, csr_values, csr_offsets, csr_columns):
        # A batch of systems is ONE block-diagonal system: mint one entry and
        # broadcast its id across the batch. (Unbatched inputs never reach
        # this rule — JAX hoists them out of vmap.)
        vb, ob, cb = in_batched
        vals = csr_values if vb else jnp.broadcast_to(
            csr_values, (axis_size,) + csr_values.shape)
        pattern_batched = ob or cb
        offs, cols = csr_offsets, csr_columns
        if pattern_batched:  # per-block patterns: materialize both batched
            if not ob:
                offs = jnp.broadcast_to(offs, (axis_size,) + offs.shape)
            if not cb:
                cols = jnp.broadcast_to(cols, (axis_size,) + cols.shape)
        token_id = _ffi_analyze(vals, offs, cols,
                                batch_size=axis_size,
                                pattern_batched=int(pattern_batched),
                                device_id=device_id, mtype_id=mtype_id,
                                mview_id=mview_id)
        return jnp.broadcast_to(token_id, (axis_size, 1)), True

    return analyze_id


@lru_cache(maxsize=None)
def _make_numeric(suffix, refactor):
    op = "refactorize" if refactor else "factorize"
    del suffix

    @jax.custom_batching.custom_vmap
    def numeric_id(token_id, csr_values, ir):
        return _ffi_numeric(token_id, csr_values, ir, op=op)

    @numeric_id.def_vmap
    def _(axis_size, in_batched, token_id, csr_values, ir):
        tb, vb, ib = in_batched
        if ib:
            raise ValueError(
                f"spineax tokens: ir_nsteps cannot vary across a batched "
                f"{op} — one block-diagonal phase has one IR setting")
        if not tb:
            raise ValueError(
                f"spineax tokens: vmap({op}) with an unbatched token and "
                "batched values — one entry cannot hold a batch of "
                "factorizations. vmap(analyze) over the batch first.")
        vals = csr_values if vb else jnp.broadcast_to(
            csr_values, (axis_size,) + csr_values.shape)
        return _ffi_numeric(token_id, vals, ir, op=op), True

    return numeric_id


@lru_cache(maxsize=None)
def _make_solve(suffix):
    del suffix

    @jax.custom_batching.custom_vmap
    def solve_id(token_id, b_values, ir):
        return _ffi_solve(token_id, b_values, ir)

    @solve_id.def_vmap
    def _(axis_size, in_batched, token_id, b_values, ir):
        _, bb, ib = in_batched
        if ib:
            raise ValueError(
                "spineax tokens: ir_nsteps cannot vary across a batched solve")
        b = b_values if bb else jnp.broadcast_to(
            b_values, (axis_size,) + b_values.shape)
        # One call either way, courtesy of the layout identities:
        # - unbatched token (one entry, N=n) + (B, n) rhs -> one multi-RHS SOLVE;
        # - batched ids (one block entry, N=B*n) + (B, n) rhs -> one block SOLVE.
        return _ffi_solve(token_id, b, ir), True

    return solve_id


# free functions ===============================================================
def analyze(csr_values, csr_offsets, csr_columns, *,
            mtype_id=1, mview_id=1, device_id=0) -> FactorToken:
    """Run cuDSS ANALYSIS (reordering, elimination tree) on a CSR system.

    Creates a registry entry owning private device copies of the (expanded)
    CSR structure and returns an *analyzed* ``FactorToken``. Call ``factorize``
    before ``solve``.

    Parameters
    ----------
    csr_values : ``(nnz,)`` for one system, or ``(B, nnz)`` for an explicit
        batch of B same-pattern systems, held as one block-diagonal entry.
        Batched values dtype is one of f32/f64/c64/c128.
    csr_offsets, csr_columns : the pattern (cast to int32). For an explicit
        batch: shared ``(n+1,)``/``(nnz,)``, or per-block ``(B, n+1)`` /
        ``(B, nnz)``.
    mtype_id : 0 general, 1 symmetric, 2 hermitian, 3 spd, 4 hpd.
    mview_id : 0 full, 1 upper, 2 lower.

    Under ``jax.vmap`` a batched system likewise becomes ONE block-diagonal
    entry (the token id is broadcast across the batch); an unbatched system is
    hoisted out and analyzed once.
    """
    dtype = jnp.dtype(csr_values.dtype)
    if csr_values.ndim == 2:
        # explicit block-diagonal batch door
        batch_size, nnz = csr_values.shape
        n = csr_offsets.shape[-1] - 1
        pattern_batched = int(csr_offsets.ndim == 2 or csr_columns.ndim == 2)
        token_id = _ffi_analyze(
            csr_values, csr_offsets, csr_columns,
            batch_size=int(batch_size), pattern_batched=pattern_batched,
            device_id=int(device_id), mtype_id=int(mtype_id),
            mview_id=int(mview_id))
        return FactorToken(
            id=token_id, kind="pbatch", dtype=dtype, n=int(n), nnz=int(nnz),
            batch_size=int(batch_size), mtype_id=int(mtype_id),
            mview_id=int(mview_id), device_id=int(device_id))
    n = csr_offsets.shape[-1] - 1
    nnz = csr_columns.shape[-1]
    token_id = _make_analyze(
        _suffix(dtype), int(device_id), int(mtype_id), int(mview_id)
    )(csr_values, csr_offsets, csr_columns)
    return FactorToken(
        id=token_id, kind="single", dtype=dtype, n=int(n), nnz=int(nnz),
        batch_size=1, mtype_id=int(mtype_id), mview_id=int(mview_id),
        device_id=int(device_id))


def _numeric(token, csr_values, ir_nsteps, refactor):
    op = "refactorize" if refactor else "factorize"
    if jnp.dtype(csr_values.dtype) != token.dtype:
        raise ValueError(
            f"spineax tokens: {op} values dtype {csr_values.dtype} != "
            f"token dtype {token.dtype}")
    if csr_values.shape[-1] != token.nnz:
        raise ValueError(
            f"spineax tokens: {op} values size {csr_values.shape[-1]} != "
            f"token nnz {token.nnz}")
    ir = _as_ir(ir_nsteps)

    B = _batch_of(token)
    if B > 1 or token.id.ndim == 2:
        # batch entry used eagerly (explicit door, or vmap-minted token used
        # outside vmap): one block-diagonal numeric phase
        if csr_values.ndim != 2 or csr_values.shape[0] != B:
            raise ValueError(
                f"spineax tokens: {op} on a batch token expects values "
                f"({B}, {token.nnz}), got {csr_values.shape}")
        token_id = _ffi_numeric(token.id, csr_values, ir, op=op)
        return _with_id(token, token_id)

    token_id = _make_numeric(_suffix(token.dtype), refactor)(
        token.id, csr_values, ir)
    return _with_id(token, token_id)


def factorize(token: FactorToken, csr_values, ir_nsteps=None) -> FactorToken:
    """Full numeric FACTORIZATION (fresh pivoting) of an analyzed token.

    Returns the token (same id — the dataflow ordering the chain). All
    post-factorization data, inertia included, comes from ``query`` /
    ``inertia``.
    """
    return _numeric(token, csr_values, ir_nsteps, refactor=False)


def refactorize(token: FactorToken, csr_values, ir_nsteps=None) -> FactorToken:
    """REFACTORIZATION: new values, reusing the previous pivot order.

    Faster than ``factorize``, but numerically valid only while the old pivot
    order remains stable for the new values — that judgement is the caller's,
    same as cuDSS's contract. Requires a factorized token. Returns the token,
    like ``factorize``.
    """
    return _numeric(token, csr_values, ir_nsteps, refactor=True)


def solve(token: FactorToken, b, ir_nsteps=None):
    """SOLVE phase only: ``x = A^{-1} b`` reusing the token's factorization.

    For a single-system token ``b`` may be ``(n,)`` or a stack ``(..., n)``
    (one multi-RHS cuDSS SOLVE). For a batch token ``b`` is ``(B, n)`` — or
    ``(..., B, n)`` for multiple rhs per block — solved in one block SOLVE.
    Under ``jax.vmap`` both cases likewise collapse to one call.
    """
    if jnp.dtype(b.dtype) != token.dtype:
        raise ValueError(
            f"spineax tokens: rhs dtype {b.dtype} != token dtype {token.dtype}")
    if b.shape[-1] != token.n:
        raise ValueError(
            f"spineax tokens: rhs trailing dim {b.shape[-1]} != token n {token.n}")
    ir = _as_ir(ir_nsteps)

    B = _batch_of(token)
    if B > 1 or token.id.ndim == 2:
        if b.ndim < 2 or b.shape[-2] != B:
            raise ValueError(
                f"spineax tokens: solve on a batch token expects rhs "
                f"(..., {B}, {token.n}), got {b.shape}")
        return _ffi_solve(token.id, b, ir)

    return _make_solve(_suffix(token.dtype))(token.id, b, ir)


_QUERY_FIELDS = (
    "lu_nnz", "npivots", "inertia", "perm_reorder_row", "perm_reorder_col",
    "perm_row", "perm_col", "perm_matching", "diag", "scale_row", "scale_col",
    "nd_partition_tree", "nsuperpanels", "schur_shape",
)


def query(token: FactorToken) -> dict:
    """Read every cuDSS data item from a factorized token.

    Returns a dict with keys: ``lu_nnz`` (int64[1]), ``npivots`` (int32[1]),
    ``inertia`` (int32[2], cuDSS's own, block-global), ``perm_reorder_row`` /
    ``perm_reorder_col`` / ``perm_row`` / ``perm_col`` / ``perm_matching``
    (int32[N]), ``diag`` (dtype[N]), ``scale_row`` / ``scale_col``
    (float32[N]), ``nd_partition_tree`` (int32, cuDSS >= 0.8's successor to
    the elimination tree), ``nsuperpanels`` (int32[1]), ``schur_shape``
    (int64[2]) — where N is the full block-system dimension (batch * n).

    Everything is returned unconditionally; items cuDSS declines for this
    matrix type / reordering algorithm come back zero-filled. Requires a
    factorized token.
    """
    B = _batch_of(token)
    N = B * token.n
    tree = int(_ps.nd_partition_tree_size())
    fn = jax.ffi.ffi_call(
        f"spineax_token_query_{_suffix(token.dtype)}",
        (
            jax.ShapeDtypeStruct((1,), jnp.int64),        # lu_nnz
            jax.ShapeDtypeStruct((1,), jnp.int32),        # npivots
            jax.ShapeDtypeStruct((2,), jnp.int32),        # inertia (cuDSS native)
            jax.ShapeDtypeStruct((N,), jnp.int32),        # perm_reorder_row
            jax.ShapeDtypeStruct((N,), jnp.int32),        # perm_reorder_col
            jax.ShapeDtypeStruct((N,), jnp.int32),        # perm_row
            jax.ShapeDtypeStruct((N,), jnp.int32),        # perm_col
            jax.ShapeDtypeStruct((N,), jnp.int32),        # perm_matching
            jax.ShapeDtypeStruct((N,), token.dtype),      # diag
            jax.ShapeDtypeStruct((N,), jnp.float32),      # scale_row
            jax.ShapeDtypeStruct((N,), jnp.float32),      # scale_col
            jax.ShapeDtypeStruct((tree,), jnp.int32),     # nd_partition_tree
            jax.ShapeDtypeStruct((1,), jnp.int32),        # nsuperpanels
            jax.ShapeDtypeStruct((2,), jnp.int64),        # schur_shape
        ),
        has_side_effect=True,
    )
    return dict(zip(_QUERY_FIELDS, fn(token.id)))


def inertia(data: dict, batch_size: int = 1):
    """Per-block [positive, negative] LDL^T inertia from ``query`` output.

    Pure function over the queried factorization data — do ``query`` once and
    derive what you need from it::

        data = query(token)
        inr = inertia(data, batch_size=B)   # int32[B, 2]

    Returns ``int32[2]`` for ``batch_size=1``, else ``int32[batch_size, 2]``
    (the block dimension is inferred from ``diag``'s length).
    """
    diag = data["diag"]
    diag_real = diag.real if jnp.iscomplexobj(diag) else diag
    n = diag.shape[0] // batch_size
    result = compute_inertia_from_diag(diag_real, batch_size, n)
    return result if batch_size > 1 else result[0]


# registry escape hatches ======================================================
def release(token: FactorToken) -> bool:
    """Eagerly free the registry entry behind ``token`` (outside jit only).

    Returns True if an entry was freed, False if it was already gone. Entries
    are otherwise LRU-evicted (capacity ``SPINEAX_FACTOR_CACHE``, default 8).
    """
    return bool(_ps.token_release(int(jax.device_get(token.id).ravel()[0])))


def registry_size() -> int:
    """Number of live factorizations in the registry."""
    return int(_ps.token_registry_size())


def cache_capacity() -> int:
    """LRU capacity (``SPINEAX_FACTOR_CACHE``, default 8)."""
    return int(_ps.token_cache_capacity())

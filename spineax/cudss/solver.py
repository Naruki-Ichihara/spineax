"""Token-based persistent factorization (docs/token_design.md).

Four free functions mapping 1:1 onto cuDSS phases, plus one data door:

    token = analyze(csr_values, csr_offsets, csr_columns, ...)   # ANALYSIS
    token = factorize(token, csr_values, ...)                    # FACTORIZATION
    token = refactorize(token, csr_values, ...)                  # REFACTORIZATION
    x     = solve(token, b, ...)                                 # SOLVE
    data  = query(token)          # every cuDSS data item, unconditionally
    inr   = inertia(data, batch_size)  # per-block [pos, neg] from that data

The ``FactorToken`` is a pytree: the traced ``int32`` registry id (dataflow
ordering, autodiff residuals, vmap batching) plus ZERO-COPY references
to the caller's CSR arrays (``values``/``offsets``/``columns``) and static
metadata. Holding the CSR arrays in the token keeps them alive exactly as
long as the factorization they produced, and every phase call passes them to
the FFI so cuDSS only ever reads live XLA buffers — no device-side copies
anywhere (design doc step 10). ``values`` is updated by ``factorize`` /
``refactorize``, so solve-time iterative refinement provably refines against
the matrix that produced the factors.

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
diagonal) is a property of the factorization: ``query`` the token after
``factorize`` / ``refactorize`` and pass the result to ``inertia`` — an IPM
can inspect it before paying for a solve.

The raw phase chain is differentiable to ARBITRARY ORDER for every mtype:
``analyze``/``factorize``/``refactorize`` carry recursive identity rules
(the values tangent rides the token's values leaf; nothing differentiates
the factors L/D themselves), and ``solve`` is a ``lax.custom_linear_solve``
whose matvec is the token's operator written in differentiable jnp gathers
— JAX's implicit-function-theorem rules for that primitive are expressed in
terms of the primitive itself, so grads, jvps, hessians etc. all reduce to
extra SOLVEs against existing factors. The one asymmetric cost: cuDSS has
no transpose solve (CUDSS_CONFIG_SOLVE_MODE is unimplemented as of 0.8), so
reverse-mode through a GENERAL (mtype 0) token factorizes A^T on the fly in
the backward pass — one extra analyze+factorize+registry entry per backward
execution. Symmetric/spd reuse the forward factors directly (A^T = A) and
hermitian/hpd via conjugation (A^T x = c  <=>  x = conj(A^-1 conj(c))).
``lx.linear_solve`` via the ``CuDSS`` adapter below is differentiable
independently, through lineax's own machinery.
"""

import dataclasses
from functools import lru_cache

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.flatten_util as jfu
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array

# Force JAX to initialize the CUDA context before importing the native module.
jax.devices()

try:
    from spineax import pbatch_solve as _ps  # native nanobind module
except ImportError as e:
    raise ImportError(
        "spineax.cudss.solver requires the pbatch_solve native module: the "
        "token API treats a single solve as the batch_size=1 case of the "
        "block-diagonal batch construction, which lives there. Was the "
        "package built with its CUDA extension (pip install with CUDA "
        "toolkit + cuDSS >= 0.8 available)?"
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


# the token ====================================================================
class FactorToken(eqx.Module):
    """Handle to a cached cuDSS (block-diagonal) factorization.

    ``id`` is the traced dispatch leaf: dataflow ordering, vjp residuals and
    vmap batching all operate on it. It is ``int32[1]`` for a token minted
    outside vmap, or ``int32[B, 1]`` (B equal copies of one entry id) for a
    token minted by ``vmap(analyze)``.

    ``values``/``offsets``/``columns`` are zero-copy references to the
    caller's CSR arrays (the block pattern exactly as handed to ``analyze``,
    NOT the expanded block-diagonal form). They keep the CSR data alive as
    long as the factorization and are passed to every phase call so cuDSS
    always reads live buffers; ``values`` is the values of the last numeric
    phase. The static fields resolve dispatch at trace time and make the
    token self-describing — a FactorToken is directly a lineax solver state.
    """

    id: Array                                # int32[1] or int32[B, 1] — traced
    values: Array                            # (nnz,) or (B, nnz)
    offsets: Array                           # int32 (n+1,) or (B, n+1)
    columns: Array                           # int32 (nnz,) or (B, nnz)
    kind: str = eqx.field(static=True)       # "single" | "pbatch" (descriptive)
    dtype: jnp.dtype = eqx.field(static=True)
    n: int = eqx.field(static=True)          # BLOCK dimension (one system)
    nnz: int = eqx.field(static=True)        # BLOCK nnz (one system)
    batch_size: int = eqx.field(static=True) # 1 unless minted by the explicit door
    mtype_id: int = eqx.field(static=True)
    mview_id: int = eqx.field(static=True)
    device_id: int = eqx.field(static=True)


def _structure_fingerprint(offsets_bd, columns_bd):
    """Position-weighted checksum (uint32[2]) of the expanded structure.

    The token's offsets/columns leaves are IMMUTABLE by contract — cuDSS's
    analysis and pivot order are tied to the pattern that was analyzed, so a
    swapped same-sized pattern would silently produce garbage factors. The
    fingerprint is computed on-device in the same pass that reads the
    structure anyway, and every phase handler compares it (8 bytes on the
    host) against the value stored at analysis — full content verification
    at zero-copy cost. Position weights make permuted contents distinct.
    """
    w_off = jnp.arange(offsets_bd.shape[0], dtype=jnp.uint32) * jnp.uint32(
        2654435761) + jnp.uint32(0x9E3779B9)
    w_col = jnp.arange(columns_bd.shape[0], dtype=jnp.uint32) * jnp.uint32(
        2246822519) + jnp.uint32(0x85EBCA6B)
    h_off = jnp.sum((offsets_bd.astype(jnp.uint32) + jnp.uint32(1)) * w_off,
                    dtype=jnp.uint32)
    h_col = jnp.sum((columns_bd.astype(jnp.uint32) + jnp.uint32(1)) * w_col,
                    dtype=jnp.uint32)
    return jnp.stack([h_off, h_col])


def _expand_structure(offsets, columns, batch_size):
    """Block-diagonal CSR structure + fingerprint from a block pattern.

    ``offsets``/``columns`` are ``(n+1,)``/``(nnz,)`` for one shared pattern
    or ``(B, n+1)``/``(B, nnz)`` for per-block patterns; the result is the
    expanded ``(B*n + 1,)``/``(B*nnz,)`` int32 structure of the one big
    block-diagonal system plus its fingerprint. B=1 passes the arrays
    through untouched (zero-copy). The expansion is an elementwise int add —
    bandwidth-trivial, XLA-temporary — recomputed per phase call instead of
    persisting in device memory; the fingerprint rides the same pass.
    """
    offsets = offsets.astype(jnp.int32)
    columns = columns.astype(jnp.int32)
    if batch_size == 1:
        offsets_bd = offsets.reshape(-1)
        columns_bd = columns.reshape(-1)
        return offsets_bd, columns_bd, _structure_fingerprint(offsets_bd,
                                                              columns_bd)
    n = offsets.shape[-1] - 1
    nnz = columns.shape[-1]
    shift = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    offs_2d = offsets if offsets.ndim == 2 else offsets[None, :]
    cols_2d = columns if columns.ndim == 2 else columns[None, :]
    body = (offs_2d[:, 1:] + shift * jnp.int32(nnz)).reshape(-1)
    offsets_bd = jnp.concatenate([jnp.zeros((1,), jnp.int32), body])
    columns_bd = (cols_2d + shift * jnp.int32(n)).reshape(-1)
    return offsets_bd, columns_bd, _structure_fingerprint(offsets_bd,
                                                          columns_bd)


def _ir_off():
    """cuDSS-internal iterative refinement is permanently OFF.

    Its refinement SpMV dereferences CSR pointers captured at earlier phase
    calls (compute-sanitizer: out-of-bounds atomics in cudss::spmv_ker when
    the expanded batch structure is an already-freed XLA temporary; keeping
    one persistent copy alive fixes it, but duplicates the pattern across
    shared-pattern batches). ``solve``'s ``ir_nsteps`` is instead implemented
    JAX-side in ``_refined_solve`` — same-precision Richardson refinement,
    which is what cuDSS runs internally anyway — whose residual SpMV
    consumes its expanded indices inside the executable that builds them,
    so nothing needs to outlive the call.
    """
    return jnp.zeros((1,), dtype=jnp.int32)


def _batch_of(token: FactorToken) -> int:
    """Effective batch size: explicit-door static, or vmap-minted id shape."""
    if token.id.ndim == 2:
        return token.id.shape[0]
    return token.batch_size


# raw FFI calls ================================================================
# Every call hands the (expanded) structure and values over as arguments, so
# cuDSS reads live XLA buffers only — the entry owns no CSR data (zero-copy).
def _ffi_analyze(values, offsets_bd, columns_bd, fingerprint, *, batch_size,
                 device_id, mtype_id, mview_id):
    fn = jax.ffi.ffi_call(
        f"spineax_token_analyze_{_suffix(values.dtype)}",
        jax.ShapeDtypeStruct((1,), jnp.int32),
        has_side_effect=True,
    )
    return fn(
        values,
        offsets_bd,
        columns_bd,
        fingerprint,
        batch_size=batch_size,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
    )


def _ffi_numeric(token_id, offsets_bd, columns_bd, fingerprint, values, *, op):
    fn = jax.ffi.ffi_call(
        f"spineax_token_{op}_{_suffix(values.dtype)}",
        jax.ShapeDtypeStruct(token_id.shape, jnp.int32),  # token (same ids)
        has_side_effect=True,
    )
    return fn(token_id, offsets_bd, columns_bd, fingerprint, values, _ir_off())


def _ffi_solve(token_id, offsets_bd, columns_bd, fingerprint, values, b):
    fn = jax.ffi.ffi_call(
        f"spineax_token_solve_{_suffix(b.dtype)}",
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        has_side_effect=True,
    )
    return fn(token_id, offsets_bd, columns_bd, fingerprint, values, b,
              _ir_off())


# vmap-aware wrappers over the single (B=1) view ===============================
# Static config is baked via lru_cache closures so custom_vmap only ever sees
# array arguments. The rules are where vmap becomes block-diagonal batching.

@lru_cache(maxsize=None)
def _make_analyze(suffix, device_id, mtype_id, mview_id):
    del suffix  # dispatch is by values dtype; suffix only keys the cache

    @jax.custom_batching.custom_vmap
    def analyze_id(csr_values, csr_offsets, csr_columns):
        offs_bd, cols_bd, fp = _expand_structure(csr_offsets, csr_columns, 1)
        return _ffi_analyze(csr_values, offs_bd, cols_bd, fp,
                            batch_size=1,
                            device_id=device_id, mtype_id=mtype_id,
                            mview_id=mview_id)

    @analyze_id.def_vmap
    def _(axis_size, in_batched, csr_values, csr_offsets, csr_columns):
        # A batch of systems is ONE block-diagonal system: mint one entry and
        # broadcast its id across the batch. (Unbatched inputs never reach
        # this rule — JAX hoists them out of vmap.)
        vb, _, _ = in_batched
        vals = csr_values if vb else jnp.broadcast_to(
            csr_values, (axis_size,) + csr_values.shape)
        offs_bd, cols_bd, fp = _expand_structure(csr_offsets, csr_columns,
                                                 axis_size)
        token_id = _ffi_analyze(vals, offs_bd, cols_bd, fp,
                                batch_size=axis_size,
                                device_id=device_id, mtype_id=mtype_id,
                                mview_id=mview_id)
        return jnp.broadcast_to(token_id, (axis_size, 1)), True

    return analyze_id


@lru_cache(maxsize=None)
def _make_numeric(suffix, refactor):
    op = "refactorize" if refactor else "factorize"
    del suffix

    @jax.custom_batching.custom_vmap
    def numeric_id(token_id, offsets, columns, csr_values):
        offs_bd, cols_bd, fp = _expand_structure(offsets, columns, 1)
        return _ffi_numeric(token_id, offs_bd, cols_bd, fp, csr_values, op=op)

    @numeric_id.def_vmap
    def _(axis_size, in_batched, token_id, offsets, columns, csr_values):
        tb, _, _, vb = in_batched
        if not tb:
            raise ValueError(
                f"spineax tokens: vmap({op}) with an unbatched token and "
                "batched values — one entry cannot hold a batch of "
                "factorizations. vmap(analyze) over the batch first.")
        vals = csr_values if vb else jnp.broadcast_to(
            csr_values, (axis_size,) + csr_values.shape)
        offs_bd, cols_bd, fp = _expand_structure(offsets, columns, axis_size)
        return _ffi_numeric(token_id, offs_bd, cols_bd, fp, vals, op=op), True

    return numeric_id


@lru_cache(maxsize=None)
def _make_solve(suffix):
    del suffix

    @jax.custom_batching.custom_vmap
    def solve_id(token_id, offsets, columns, values, b_values):
        # The base case already absorbs any leading rhs axes as one
        # multi-RHS SOLVE (nrhs is derived from element counts).
        offs_bd, cols_bd, fp = _expand_structure(offsets, columns, 1)
        return _ffi_solve(token_id, offs_bd, cols_bd, fp, values, b_values)

    @solve_id.def_vmap
    def _(axis_size, in_batched, token_id, offsets, columns, values, b_values):
        # Collapse ONE batch axis, then RECURSE into the wrapped function
        # (never the raw FFI): transforms are free to re-batch the result —
        # jacfwd-of-grad nests a vmap per differentiation order — and each
        # extra axis peels off through this rule again.
        tb, _, _, vb, bb = in_batched
        b = b_values if bb else jnp.broadcast_to(
            b_values, (axis_size,) + b_values.shape)
        if not tb:
            # unbatched token: the batch axis is just more rhs columns
            return solve_id(token_id, offsets, columns, values, b), True
        # batched ids (one block entry): flatten the batch axis into the ONE
        # block-diagonal system of dimension B*n and solve it single-style
        offs_bd, cols_bd, _ = _expand_structure(offsets, columns, axis_size)
        vals = values if vb else jnp.broadcast_to(
            values, (axis_size,) + values.shape)
        b2 = jnp.moveaxis(b, 0, -2)  # (B, ..., n) -> (..., B, n)
        bf = b2.reshape(b2.shape[:-2] + (-1,))
        out = solve_id(token_id[0], offs_bd, cols_bd, vals.reshape(-1), bf)
        return jnp.moveaxis(out.reshape(b2.shape), -2, 0), True

    return solve_id


# free functions ===============================================================
def analyze(csr_values, csr_offsets, csr_columns, *,
            mtype_id=1, mview_id=1, device_id=0) -> FactorToken:
    """Run cuDSS ANALYSIS (reordering, elimination tree) on a CSR system.

    Creates a registry entry (cuDSS handles + factors only — the CSR arrays
    stay the caller's, referenced zero-copy by the returned *analyzed*
    ``FactorToken``). Call ``factorize`` before ``solve``.

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
    core = _make_analyze_ad(_suffix(jnp.dtype(csr_values.dtype)),
                            int(device_id), int(mtype_id), int(mview_id))
    return core(csr_values, csr_offsets.astype(jnp.int32),
                csr_columns.astype(jnp.int32))


@lru_cache(maxsize=None)
def _make_analyze_ad(suffix, device_id, mtype_id, mview_id):
    """The analyze implementation wrapped for autodiff (identity rule)."""

    def impl(csr_values, csr_offsets, csr_columns):
        dtype = jnp.dtype(csr_values.dtype)
        n = csr_offsets.shape[-1] - 1
        nnz = csr_columns.shape[-1]
        if csr_values.ndim == 2:
            # explicit block-diagonal batch door
            batch_size = int(csr_values.shape[0])
            offs_bd, cols_bd, fp = _expand_structure(csr_offsets, csr_columns,
                                                     batch_size)
            token_id = _ffi_analyze(
                csr_values, offs_bd, cols_bd, fp,
                batch_size=batch_size, device_id=device_id,
                mtype_id=mtype_id, mview_id=mview_id)
            return FactorToken(
                id=token_id, values=csr_values, offsets=csr_offsets,
                columns=csr_columns,
                kind="pbatch", dtype=dtype, n=int(n), nnz=int(nnz),
                batch_size=batch_size, mtype_id=mtype_id,
                mview_id=mview_id, device_id=device_id)
        token_id = _make_analyze(suffix, device_id, mtype_id, mview_id)(
            csr_values, csr_offsets, csr_columns)
        return FactorToken(
            id=token_id, values=csr_values, offsets=csr_offsets,
            columns=csr_columns,
            kind="single", dtype=dtype, n=int(n), nnz=int(nnz),
            batch_size=1, mtype_id=mtype_id, mview_id=mview_id,
            device_id=device_id)

    core = jax.custom_jvp(impl)

    @core.defjvp
    def _(primals, tangents):
        # Recursive call (not impl): under higher-order differentiation the
        # primals themselves carry tangents, which must hit this rule again
        # rather than the raw ffi_call inside impl.
        dvals, _, _ = tangents
        token = core(*primals)
        # tangent = values tangent on the values leaf, float0 zeros on the
        # non-differentiable integer leaves
        f0 = lambda x: np.zeros(jnp.shape(x), jax.dtypes.float0)
        dtoken = dataclasses.replace(
            token, id=f0(token.id), values=dvals,
            offsets=f0(token.offsets), columns=f0(token.columns))
        return token, dtoken

    return core


def _numeric(token, csr_values, refactor):
    op = "refactorize" if refactor else "factorize"
    if jnp.dtype(csr_values.dtype) != token.dtype:
        raise ValueError(
            f"spineax tokens: {op} values dtype {csr_values.dtype} != "
            f"token dtype {token.dtype}")
    if csr_values.shape[-1] != token.nnz:
        raise ValueError(
            f"spineax tokens: {op} values size {csr_values.shape[-1]} != "
            f"token nnz {token.nnz}")
    B = _batch_of(token)
    if (B > 1 or token.id.ndim == 2) and (
            csr_values.ndim != 2 or csr_values.shape[0] != B):
        raise ValueError(
            f"spineax tokens: {op} on a batch token expects values "
            f"({B}, {token.nnz}), got {csr_values.shape}")
    return _make_numeric_ad(refactor)(token, csr_values)


def _numeric_impl(token, csr_values, refactor):
    op = "refactorize" if refactor else "factorize"
    B = _batch_of(token)
    if B > 1 or token.id.ndim == 2:
        # batch entry used eagerly (explicit door, or vmap-minted token used
        # outside vmap): one block-diagonal numeric phase
        offs_bd, cols_bd, fp = _expand_structure(token.offsets,
                                                 token.columns, B)
        token_id = _ffi_numeric(token.id, offs_bd, cols_bd, fp, csr_values,
                                op=op)
    else:
        token_id = _make_numeric(_suffix(token.dtype), refactor)(
            token.id, token.offsets, token.columns, csr_values)
    # same entry id (dataflow-ordered), values leaf swapped to the
    # just-factorized values
    return dataclasses.replace(token, id=token_id, values=csr_values)


@lru_cache(maxsize=None)
def _make_numeric_ad(refactor):
    """The numeric-phase implementation wrapped for autodiff.

    The rule is a pure identity pass-through: the tangent of ``csr_values``
    rides into the output token's ``values`` leaf and the FFI id output is
    non-differentiable. This is mathematically complete because ``solve``
    below differentiates through the factorization via the implicit
    function theorem — no derivative of the factors L/D is ever needed.
    The rule calls the wrapped function recursively so it composes to any
    differentiation order (higher-order primals carry tangents that must
    re-enter this rule, not the raw ffi_call).
    """

    def impl(token, csr_values):
        return _numeric_impl(token, csr_values, refactor)

    core = jax.custom_jvp(impl)

    @core.defjvp
    def _(primals, tangents):
        dtoken, dvals = tangents
        out = core(*primals)
        return out, dataclasses.replace(dtoken, values=dvals)

    return core


def factorize(token: FactorToken, csr_values) -> FactorToken:
    """Full numeric FACTORIZATION (fresh pivoting) of an analyzed token.

    Returns the token (same id — the dataflow ordering the chain). All
    post-factorization data, inertia included, comes from ``query`` /
    ``inertia``.
    """
    return _numeric(token, csr_values, refactor=False)


def refactorize(token: FactorToken, csr_values) -> FactorToken:
    """REFACTORIZATION: new values, reusing the previous pivot order.

    Faster than ``factorize``, but numerically valid only while the old pivot
    order remains stable for the new values — that judgement is the caller's,
    same as cuDSS's contract. Requires a factorized token. Returns the token,
    like ``factorize``.
    """
    return _numeric(token, csr_values, refactor=True)


def solve(token: FactorToken, b, ir_nsteps=None):
    if jnp.dtype(b.dtype) != token.dtype:
        raise ValueError(
            f"spineax tokens: rhs dtype {b.dtype} != token dtype {token.dtype}")
    if b.shape[-1] != token.n:
        raise ValueError(
            f"spineax tokens: rhs trailing dim {b.shape[-1]} != token n {token.n}")
    B = _batch_of(token)
    if (B > 1 or token.id.ndim == 2) and (b.ndim < 2 or b.shape[-2] != B):
        raise ValueError(
            f"spineax tokens: solve on a batch token expects rhs "
            f"(..., {B}, {token.n}), got {b.shape}")
    nsteps = 0 if ir_nsteps is None else int(ir_nsteps)  # static: unrolled
    # The solver callables below are pure numerical inverses: all derivative
    # information enters through the matvec closure (implicit function
    # theorem), so the token they use is gradient-stopped.
    tok_ng = jax.lax.stop_gradient(token)

    def mv(x):
        return _matvec(token, x)

    def solve_fn(_mv, rhs):
        return _refined_solve(tok_ng, rhs, nsteps)

    if token.mtype_id in (1, 3):  # A^T = A (incl. complex symmetric): same factors
        return jax.lax.custom_linear_solve(mv, b, solve_fn, solve_fn,
                                           symmetric=True)
    if token.mtype_id in (2, 4):  # hermitian: A^T = conj(A), same factors

        def t_solve(_mv, rhs):
            return jnp.conj(_refined_solve(tok_ng, jnp.conj(rhs), nsteps))

        return jax.lax.custom_linear_solve(mv, b, solve_fn, t_solve)

    def t_solve(_mv, rhs):  # general: cuDSS has no transpose solve
        return _transpose_solve_general(tok_ng, rhs, nsteps)

    return jax.lax.custom_linear_solve(mv, b, solve_fn, t_solve)


def _solve_impl(token, b):
    B = _batch_of(token)
    fn = _make_solve(_suffix(token.dtype))
    if B > 1 or token.id.ndim == 2:
        # batch entry used eagerly: solve the ONE expanded block-diagonal
        # system single-style, through the same custom_vmap wrapper so any
        # transform-added batch axes stay collapsible (multi-RHS)
        offs_bd, cols_bd, _ = _expand_structure(token.offsets,
                                                token.columns, B)
        tid = token.id[0] if token.id.ndim == 2 else token.id
        bf = b.reshape(b.shape[:-2] + (-1,))
        out = fn(tid, offs_bd, cols_bd, token.values.reshape(-1), bf)
        return out.reshape(b.shape)
    return fn(token.id, token.offsets, token.columns, token.values, b)


def _refined_solve(token, b, nsteps):
    """``x = A^{-1} b`` plus ``nsteps`` rounds of JAX-side iterative
    refinement: x += A^{-1}(b - A x), the residual computed by ``_matvec``.

    Refinement lives HERE and never in cuDSS (see ``_ir_off``). Same
    precision as the working dtype — exactly what cuDSS's internal IR does —
    and safe for every door: the residual SpMV consumes its expanded indices
    inside the executable that builds them, so no buffer has to outlive the
    call. ``nsteps`` is static; the loop unrolls into the jaxpr.
    """
    x = _solve_impl(token, b)
    for _ in range(nsteps):
        x = x + _solve_impl(token, b - _matvec(token, x))
    return x


# autodiff for solve ===========================================================

def _pattern_rows(offsets, nnz):
    """Row index of each stored CSR entry, from the (possibly per-block
    batched) offsets pattern."""
    def one(o):
        return jnp.repeat(jnp.arange(o.shape[-1] - 1, dtype=jnp.int32),
                          jnp.diff(o), total_repeat_length=nnz)
    return jax.vmap(one)(offsets) if offsets.ndim == 2 else one(offsets)


def _gather_last(a, idx):
    """a[..., idx] with idx either shared ``(nnz,)`` or per-block
    ``(B, nnz)`` (matching a's ``(..., B, n)`` block axis)."""
    if idx.ndim == 1:
        return jnp.take(a, idx, axis=-1)
    idx_b = jnp.broadcast_to(idx, a.shape[:-1] + (idx.shape[-1],))
    return jnp.take_along_axis(a, idx_b, axis=-1)


def _matvec(token, x):
    B = _batch_of(token)
    offs_bd, cols_bd, _ = _expand_structure(token.offsets, token.columns, B)
    rows_bd = _pattern_rows(offs_bd, B * token.nnz)
    vals = token.values.reshape(-1)
    xf = x.reshape(x.shape[:-2] + (-1,)) if B > 1 else x
    y = jnp.zeros_like(xf).at[..., rows_bd].add(
        vals * _gather_last(xf, cols_bd))
    if token.mview_id in (1, 2):
        mvals = jnp.conj(vals) if token.mtype_id in (2, 4) else vals
        mvals = jnp.where(rows_bd == cols_bd, 0, mvals)  # diag stored once
        y = y.at[..., cols_bd].add(mvals * _gather_last(xf, rows_bd))
    return y.reshape(x.shape)


def _transpose_csr(values, offsets, columns):
    """(values, offsets, columns) of ``A^T`` for one flat CSR system:
    the CSR of the transpose IS the CSC of A, obtained by a stable
    reorder of the entries by (column, row). On-device jnp ops."""
    n = offsets.shape[0] - 1
    rows = _pattern_rows(offsets, columns.shape[-1])
    order = jnp.lexsort((rows, columns))
    t_offsets = jnp.concatenate([
        jnp.zeros((1,), jnp.int32),
        jnp.cumsum(jnp.zeros((n,), jnp.int32).at[columns].add(1),
                   dtype=jnp.int32)])
    return values[order], t_offsets, rows[order]


def _transpose_solve_general(token, rhs, nsteps):
    """``A^-T rhs`` for a general (mtype 0) token.

    cuDSS cannot solve against the transpose of existing factors
    (CUDSS_CONFIG_SOLVE_MODE is "not supported right now" as of 0.8), so
    reverse mode through a general solve transposes the CSR system on device
    and runs a fresh analyze+factorize+solve. That is a full factorization
    AND a new LRU registry entry per backward execution.
    """
    B = _batch_of(token)
    offs_bd, cols_bd, _ = _expand_structure(token.offsets, token.columns, B)
    t_vals, t_offs, t_cols = _transpose_csr(token.values.reshape(-1),
                                            offs_bd, cols_bd)
    t_token = analyze(t_vals, t_offs, t_cols,
                      mtype_id=0, mview_id=0, device_id=token.device_id)
    t_token = factorize(t_token, t_token.values)
    rf = rhs.reshape(rhs.shape[:-2] + (-1,)) if B > 1 else rhs
    return _refined_solve(t_token, rf, nsteps).reshape(rhs.shape)


_QUERY_FIELDS = (
    "lu_nnz", "npivots", "inertia", "perm_reorder_row", "perm_reorder_col",
    "perm_row", "perm_col", "perm_matching", "diag", "scale_row", "scale_col",
    "nd_partition_tree", "nsuperpanels", "schur_shape",
)


# Fields whose (N,)-sized values are in INPUT ORDER, so a block-diagonal
# system splits them cleanly into per-block (B, n) slices. Everything else is
# block-global: under vmap it is broadcast unchanged to every batch element
# (perm VALUES index the whole block system; lu_nnz/npivots/inertia/
# nd_partition_tree/nsuperpanels/schur_shape describe the one factorization).
_QUERY_SPLIT_FIELDS = frozenset({"diag", "scale_row", "scale_col"})


@lru_cache(maxsize=None)
def _make_query(suffix, dtype, n, tree):
    """token.id -> tuple of the 14 query outputs for a system of dimension n."""

    @jax.custom_batching.custom_vmap
    def query_id(token_id):
        fn = jax.ffi.ffi_call(
            f"spineax_token_query_{suffix}",
            (
                jax.ShapeDtypeStruct((1,), jnp.int64),        # lu_nnz
                jax.ShapeDtypeStruct((1,), jnp.int32),        # npivots
                jax.ShapeDtypeStruct((2,), jnp.int32),        # inertia (cuDSS native)
                jax.ShapeDtypeStruct((n,), jnp.int32),        # perm_reorder_row
                jax.ShapeDtypeStruct((n,), jnp.int32),        # perm_reorder_col
                jax.ShapeDtypeStruct((n,), jnp.int32),        # perm_row
                jax.ShapeDtypeStruct((n,), jnp.int32),        # perm_col
                jax.ShapeDtypeStruct((n,), jnp.int32),        # perm_matching
                jax.ShapeDtypeStruct((n,), dtype),            # diag
                jax.ShapeDtypeStruct((n,), jnp.float32),      # scale_row
                jax.ShapeDtypeStruct((n,), jnp.float32),      # scale_col
                jax.ShapeDtypeStruct((tree,), jnp.int32),     # nd_partition_tree
                jax.ShapeDtypeStruct((1,), jnp.int32),        # nsuperpanels
                jax.ShapeDtypeStruct((2,), jnp.int64),        # schur_shape
            ),
            has_side_effect=True,
        )
        return tuple(fn(token_id))

    @query_id.def_vmap
    def _(axis_size, in_batched, token_id):
        # Batched ids are B equal copies of ONE block entry: run a single
        # query on the whole B*n block system, then split the input-ordered
        # per-block fields to (B, n) and broadcast the block-global rest.
        del in_batched
        outs = _make_query(suffix, dtype, axis_size * n, tree)(token_id[0])
        batched = []
        for field, a in zip(_QUERY_FIELDS, outs):
            if field in _QUERY_SPLIT_FIELDS:
                batched.append(a.reshape(axis_size, n))
            else:
                batched.append(jnp.broadcast_to(a, (axis_size,) + a.shape))
        return tuple(batched), (True,) * len(_QUERY_FIELDS)

    return query_id


def query(token: FactorToken) -> dict:
    B = _batch_of(token)
    tree = int(_ps.nd_partition_tree_size())
    fn = _make_query(_suffix(token.dtype), token.dtype, B * token.n, tree)
    return dict(zip(_QUERY_FIELDS, fn(token.id)))


def inertia(data: dict, batch_size: int = 1):
    """Per-block [positive, negative] LDL^T inertia from ``query`` output. CuDSS
    doesnt yet reliably return 0 eigenvalues as of 0.8.
    """
    diag = data["diag"]
    diag_real = diag.real if jnp.iscomplexobj(diag) else diag
    n = diag.shape[0] // batch_size

    out = diag_real.reshape([batch_size, n])

    # cuDSS pivoting threshold seems to be 1e-13. everything above this on
    # plus or minus side seems to reliably indicate that particular inertia value.
    threshold = 1e-13
    positive = jnp.sum(out >= threshold, axis=1)
    negative = jnp.sum(out <= -threshold, axis=1)

    result = jnp.stack([positive, negative], axis=1, dtype=jnp.int32)

    return result if batch_size > 1 else result[0]


# registry escape hatches ======================================================
def release(token: FactorToken) -> bool:
    """manually free registry (only outside jit - otherwise whenever LRU overflows
    according to SPINEAX_FACTOR_CACHE)"""
    return bool(_ps.token_release(int(jax.device_get(token.id).ravel()[0])))


def registry_size() -> int:
    """Number of live factorizations in the registry."""
    return int(_ps.token_registry_size())


def cache_capacity() -> int:
    """LRU capacity (``SPINEAX_FACTOR_CACHE``, default 8)."""
    return int(_ps.token_cache_capacity())


# lineax front door — the default user-facing API ==============================

class CSROperator(lx.AbstractLinearOperator):
    """A square matrix in CSR form (full pattern; values as the one leaf).

    Structure is declared lineax-style via TAGS, not subclasses — exactly
    like ``lx.MatrixLinearOperator(matrix, lx.symmetric_tag)`` for dense:

        CSROperator(vals, offs, cols)                       # general
        CSROperator(vals, offs, cols, lx.symmetric_tag)     # symmetric
        CSROperator(vals, offs, cols,
                    (lx.symmetric_tag,
                     lx.positive_semidefinite_tag))         # SPD

    Always the FULL sparsity pattern (both triangles; cuDSS ``mview_id=0``):
    ``mv`` is a plain BCSR matvec and a general matrix has no triangular
    shorthand anyway. The arrays are referenced zero-copy, same as
    everywhere else.
    """

    values: Array
    offsets: Array
    columns: Array
    tags: frozenset = eqx.field(static=True)

    def __init__(self, values, offsets, columns, tags=()):
        self.values = values
        self.offsets = offsets
        self.columns = columns
        if isinstance(tags, (tuple, list, set, frozenset)):
            self.tags = frozenset(tags)
        else:
            self.tags = frozenset({tags})

    def _bcsr(self):
        n = self.offsets.shape[0] - 1
        return jsparse.BCSR((self.values, self.columns, self.offsets),
                            shape=(n, n))

    def mv(self, vector):
        return self._bcsr() @ vector

    def as_matrix(self):
        return self._bcsr().todense()

    def transpose(self):
        if lx.symmetric_tag in self.tags:
            return self
        t_vals, t_offs, t_cols = _transpose_csr(self.values, self.offsets,
                                                self.columns)
        return CSROperator(t_vals, t_offs, t_cols,
                           lx.transpose_tags(self.tags))

    def in_structure(self):
        n = self.offsets.shape[0] - 1
        return jax.ShapeDtypeStruct((n,), self.values.dtype)

    def out_structure(self):
        return self.in_structure()


# lineax dispatches these predicates by operator class; each reads its tag,
# mirroring how lx.MatrixLinearOperator + tags behaves for dense matrices
for _predicate, _tag in (
    (lx.is_symmetric, lx.symmetric_tag),
    (lx.is_diagonal, lx.diagonal_tag),
    (lx.is_tridiagonal, lx.tridiagonal_tag),
    (lx.is_lower_triangular, lx.lower_triangular_tag),
    (lx.is_upper_triangular, lx.upper_triangular_tag),
    (lx.is_positive_semidefinite, lx.positive_semidefinite_tag),
    (lx.is_negative_semidefinite, lx.negative_semidefinite_tag),
    (lx.has_unit_diagonal, lx.unit_diagonal_tag),
):
    _predicate.register(CSROperator)(
        lambda operator, _tag=_tag: _tag in operator.tags)


@lx.linearise.register(CSROperator)
@lx.materialise.register(CSROperator)
def _(operator):
    return operator


@lx.conj.register(CSROperator)
def _(operator):
    return CSROperator(jnp.conj(operator.values), operator.offsets,
                       operator.columns, operator.tags)


class CuDSS(lx.AbstractLinearSolver):
    """lineax front door for the token API.

    The cuDSS matrix type is resolved from the OPERATOR'S TAGS, the same
    way lineax's own solvers consult ``lx.is_symmetric`` etc.:

        symmetric + positive_semidefinite  ->  mtype 3 (Cholesky)
        symmetric                          ->  mtype 1 (LDL^T)
        untagged                           ->  mtype 0 (general LU)

    General operators are fully supported, gradients included — with the
    documented cost that anything needing ``A^T`` (lineax's backward pass,
    ``solver.transpose``) must factorize the transpose from scratch, since
    cuDSS has no transpose solve (design doc section 8).

    lineax's phase boundary is operator-dependent vs vector-dependent work
    (``lx.Cholesky.init`` runs ``cho_factor``; ``compute`` runs
    ``cho_solve``), so the protocol slots follow that convention:

        init    = analyze + factorize    (all operator-dependent work)
        compute = solve                  (per-vector work only)

    The state is the factorized token, so lineax's ``state=`` argument
    means the same thing it means for every built-in solver: one
    factorization, many right-hand sides.

    Every un-stated ``lx.linear_solve`` call re-inits, minting a registry
    entry whose cuDSS factors occupy device memory (the CSR arrays are
    zero-copy references in the token, never duplicated). Outside jit, free
    it eagerly with ``release(sol.state)``; under jit that is impossible
    and the LRU (``SPINEAX_FACTOR_CACHE``, default 8) bounds the leak by
    evicting oldest-used entries.

    For true control the phases are explicit methods that thread tokens,
    mirroring the ``spineax.cudss`` free functions with operator sugar:

        solver = CuDSS()
        token  = solver.analyze(operator)                 # ANALYSIS
        token  = solver.factorize(token, operator)        # FACTORIZATION
        token  = solver.refactorize(token, new_operator)  # REFACTORIZATION
        x      = solver.solve(token, b)                   # SOLVE (repeatable)
        data   = solver.query(token)                      # every cuDSS data item
    """

    # explicit phases ----------------------------------------------------------

    def analyze(self, operator):
        if lx.is_symmetric(operator):
            mtype_id = 3 if lx.is_positive_semidefinite(operator) else 1
        else:
            mtype_id = 0
        return analyze(operator.values, operator.offsets, operator.columns,
                       mtype_id=mtype_id, mview_id=0)

    def factorize(self, token, operator):
        return factorize(token, operator.values)

    def refactorize(self, token, operator):
        return refactorize(token, operator.values)

    def solve(self, token, vector):
        return solve(token, vector)

    def query(self, token):
        return query(token)

    # lineax protocol ----------------------------------------------------------

    def init(self, operator, options):
        del options
        return self.factorize(self.analyze(operator), operator)

    def compute(self, state, vector, options):
        del options
        vector, unflatten = jfu.ravel_pytree(vector)
        solution = self.solve(state, vector)
        return unflatten(solution), lx.RESULTS.successful, {}

    def transpose(self, state, options):
        if state.mtype_id in (1, 3):
            return state, options  # A^T = A: same factorization
        # general: cuDSS has no transpose solve, so lineax's backward pass
        # pays for a fresh factorization of A^T (one analyze+factorize+
        # registry entry), mirroring the raw autodiff path
        t_vals, t_offs, t_cols = _transpose_csr(state.values, state.offsets,
                                                state.columns)
        t_token = analyze(t_vals, t_offs, t_cols,
                          mtype_id=0, mview_id=0, device_id=state.device_id)
        return factorize(t_token, t_token.values), options

    def conj(self, state, options):
        return state, options  # real-valued operators

    def assume_full_rank(self):
        return True

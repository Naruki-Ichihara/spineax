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
    outside vmap, or ``int32[B1, ..., Bk, 1]`` (equal copies of one entry id,
    one leading axis per vmap level) for a token minted under (possibly
    nested) ``vmap(analyze)``.

    An id names ONE immutable numeric state: ``factorize``/``refactorize``
    consume their input's id and return a fresh one. Using an id that is no
    longer resident (superseded, or LRU-evicted) transparently REBUILDS that
    state from the token's own arrays — only ``query``, which carries no CSR
    data, raises instead. ``rebuild_count()`` counts the heals; a rising
    count means SPINEAX_FACTOR_CACHE is too small for the working set.

    ``values``/``offsets``/``columns`` are zero-copy references to the
    caller's CSR arrays (the block pattern exactly as handed to ``analyze``,
    NOT the expanded block-diagonal form). They keep the CSR data alive as
    long as the factorization and are passed to every phase call so cuDSS
    always reads live buffers; ``values`` is the values of the last numeric
    phase. The static fields resolve dispatch at trace time and make the
    token self-describing — a FactorToken is directly a lineax solver state.
    """

    id: Array                                # int32[1] (+1 axis per vmap level) — traced
    values: Array                            # (nnz,) or (B, nnz)
    offsets: Array                           # int32 (n+1,) or (B, n+1)
    columns: Array                           # int32 (nnz,) or (B, nnz)
    phase: str = eqx.field(static=True)      # "analyzed" | "factorized"
    kind: str = eqx.field(static=True)       # "single" | "pbatch" (descriptive)
    dtype: jnp.dtype = eqx.field(static=True)
    n: int = eqx.field(static=True)          # BLOCK dimension (one system)
    nnz: int = eqx.field(static=True)        # BLOCK nnz (one system)
    batch_size: int = eqx.field(static=True) # 1 unless minted by the explicit door
    mtype_id: int = eqx.field(static=True)
    mview_id: int = eqx.field(static=True)
    device_id: int = eqx.field(static=True)
    reordering_id: int = eqx.field(static=True)  # cudssReorderingAlg_t 0-5
    memory_id: int = eqx.field(static=True)      # 0 device, 1 hybrid host+device


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
    offs_2d = offsets.reshape((-1, offsets.shape[-1]))  # any leading axes
    cols_2d = columns.reshape((-1, columns.shape[-1]))
    body = (offs_2d[:, 1:] + shift * jnp.int32(nnz)).reshape(-1)
    offsets_bd = jnp.concatenate([jnp.zeros((1,), jnp.int32), body])
    columns_bd = (cols_2d + shift * jnp.int32(n)).reshape(-1)
    return offsets_bd, columns_bd, _structure_fingerprint(offsets_bd,
                                                          columns_bd)


def _batch_of(token: FactorToken) -> int:
    """Total block count: one id axis per vmap level times the explicit-door
    static — the two doors compose (e.g. vmap over an explicit batch)."""
    return int(np.prod(token.id.shape[:-1])) * token.batch_size


# raw FFI calls ================================================================
from jax.extend.mlir import ir as _ir
from jax.interpreters import mlir as _jmlir
from jaxlib.mlir.dialects import stablehlo as _stablehlo

_FFI_API_VERSION = 4  # typed FFI: the contract solver.cpp's handlers compile


def _ffi_probe_jaxpr():
    """Trace (never run) a jax.ffi call to harvest jax's own FFI machinery:
    the effect instance (already in jax's control-flow allowlists, so our
    calls are while/cond-legal exactly like jax.ffi ones) and the concrete
    aval class — by value, not by private-module name."""
    fn = jax.ffi.ffi_call("__spineax_probe", jax.ShapeDtypeStruct((1,), jnp.int32),
                          has_side_effect=True)
    return jax.make_jaxpr(lambda: fn())()


_probe = _ffi_probe_jaxpr()
(_SPINEAX_FFI_EFFECT,) = tuple(_probe.effects)
_ShapedArray = type(_probe.out_avals[0])

_ffi_p = jax.extend.core.Primitive("spineax_ffi")
_ffi_p.multiple_results = True


@lru_cache(maxsize=None)
def _ffi_eager(name, out_avals, attrs):
    # eager binds execute through jit (one compiled call per unique config)
    return jax.jit(lambda *a: _ffi_p.bind(*a, name=name, out_avals=out_avals,
                                          attrs=attrs))


_ffi_p.def_impl(lambda *args, name, out_avals, attrs:
                _ffi_eager(name, out_avals, attrs)(*args))


# issue #18 fixed by Igor Kuszczak - no need for effectful things after token rework.
@_ffi_p.def_abstract_eval
def _ffi_abstract_eval(*avals_in, name, out_avals, attrs):
    del avals_in, name, attrs
    return list(out_avals)


def _ir_tensor_type(aval):
    dt = jnp.dtype(aval.dtype)
    if dt == jnp.float32:      et = _ir.F32Type.get()
    elif dt == jnp.float64:    et = _ir.F64Type.get()
    elif dt == jnp.complex64:  et = _ir.ComplexType.get(_ir.F32Type.get())
    elif dt == jnp.complex128: et = _ir.ComplexType.get(_ir.F64Type.get())
    elif dt == jnp.int32:      et = _ir.IntegerType.get_signless(32)
    elif dt == jnp.int64:      et = _ir.IntegerType.get_signless(64)
    elif dt == jnp.uint32:     et = _ir.IntegerType.get_unsigned(32)
    else:
        raise ValueError(f"spineax ffi: unhandled dtype {dt}")
    return _ir.RankedTensorType.get(aval.shape, et)


def _ffi_lowering(ctx, *operands, name, out_avals, attrs):
    del out_avals  # ctx.avals_out is authoritative
    # Build the stablehlo.custom_call directly: the TYPED_FFI form is
    # api_version=4 with a DICTIONARY backend_config (the stablehlo verifier
    # enforces this), which jax's mlir.custom_call helper cannot express —
    # its dict branch is what rewrites to api_version=1 + mhlo.backend_config.
    i64 = _ir.IntegerType.get_signless(64)

    def _layout(a):  # row-major, minor-to-major order (as the FFI expects)
        return _ir.DenseIntElementsAttr.get(
            np.atleast_1d(np.asarray(tuple(reversed(range(a.ndim))), np.int64)),
            type=_ir.IndexType.get())

    attributes = dict(
        call_target_name=_ir.StringAttr.get(name),
        has_side_effect=_ir.BoolAttr.get(True),
        backend_config=_ir.DictAttr.get(
            {k: _ir.IntegerAttr.get(i64, int(v)) for k, v in attrs}),
        api_version=_ir.IntegerAttr.get(
            _ir.IntegerType.get_signless(32), _FFI_API_VERSION),
        called_computations=_ir.ArrayAttr.get([]),
        operand_layouts=_ir.ArrayAttr.get(
            [_layout(a) for a in ctx.avals_in]),
        result_layouts=_ir.ArrayAttr.get(
            [_layout(a) for a in ctx.avals_out]),
    )
    op = _stablehlo.CustomCallOp.build_generic(
        results=[_ir_tensor_type(a) for a in ctx.avals_out],
        operands=list(operands),
        attributes=attributes,
    )
    return op.results


_jmlir.register_lowering(_ffi_p, _ffi_lowering)


def _ffi_call4(name, out_specs, *operands, **attrs):
    """Emit one spineax typed-FFI custom call (explicit api_version 4).

    ``out_specs`` is a tuple of ShapeDtypeStructs; returns a list of arrays.
    Integer ``attrs`` become i64 FFI attributes (solver.cpp Attr<int64_t>).
    """
    out_avals = tuple(_ShapedArray(s.shape, jnp.dtype(s.dtype))
                      for s in out_specs)
    return _ffi_p.bind(*operands, name=name, out_avals=out_avals,
                       attrs=tuple(sorted(attrs.items())))


# The per-entry cuDSS config, threaded as ONE tuple of i64 FFI attrs. It
# rides every phase call (not just analyze) so a non-resident id (renamed
# away by a later numeric phase, or LRU-evicted) can be rebuilt in the
# handler from the call's own operands (self-healing; see solver.cpp).
_CFG_KEYS = ("device_id", "mtype_id", "mview_id", "reordering_id", "memory_id")


def _cfg_of(token):
    return tuple(getattr(token, k) for k in _CFG_KEYS)


def _ffi_analyze(values, offsets_bd, columns_bd, fingerprint, *, batch_size,
                 cfg):
    (token_id,) = _ffi_call4(
        f"spineax_token_analyze_{_suffix(values.dtype)}",
        (jax.ShapeDtypeStruct((1,), jnp.int32),),
        values, offsets_bd, columns_bd, fingerprint,
        batch_size=batch_size, **dict(zip(_CFG_KEYS, cfg)))
    return token_id


def _ffi_numeric(token_id, offsets_bd, columns_bd, fingerprint, values, *, op,
                 cfg):
    (out_id,) = _ffi_call4(
        f"spineax_token_{op}_{_suffix(values.dtype)}",
        (jax.ShapeDtypeStruct(token_id.shape, jnp.int32),),  # token (FRESH id)
        token_id, offsets_bd, columns_bd, fingerprint, values,
        **dict(zip(_CFG_KEYS, cfg)))
    return out_id


def _ffi_solve(token_id, offsets_bd, columns_bd, fingerprint, values, b, *,
               cfg):
    (x,) = _ffi_call4(
        f"spineax_token_solve_{_suffix(b.dtype)}",
        (jax.ShapeDtypeStruct(b.shape, b.dtype),),
        token_id, offsets_bd, columns_bd, fingerprint, values, b,
        **dict(zip(_CFG_KEYS, cfg)))
    return x


# vmap-aware wrappers over the single (B=1) view ===============================
# Static config is baked via lru_cache closures so custom_vmap only ever sees
# array arguments. The rules are where vmap becomes block-diagonal batching.

@lru_cache(maxsize=None)
def _make_analyze(suffix, cfg):
    del suffix  # dispatch is by values dtype; suffix only keys the cache

    @jax.custom_batching.custom_vmap
    def analyze_id(csr_values, csr_offsets, csr_columns):
        offs_bd, cols_bd, fp = _expand_structure(csr_offsets, csr_columns, 1)
        return _ffi_analyze(csr_values, offs_bd, cols_bd, fp,
                            batch_size=1, cfg=cfg)

    @analyze_id.def_vmap
    def _(axis_size, in_batched, csr_values, csr_offsets, csr_columns):
        vb, _, _ = in_batched
        vals = csr_values if vb else jnp.broadcast_to(
            csr_values, (axis_size,) + csr_values.shape)
        offs_bd, cols_bd, _ = _expand_structure(csr_offsets, csr_columns,
                                                axis_size)
        token_id = analyze_id(vals.reshape(-1), offs_bd, cols_bd)
        return jnp.broadcast_to(token_id, (axis_size,) + token_id.shape), True

    return analyze_id


@lru_cache(maxsize=None)
def _make_numeric(suffix, refactor, cfg):
    op = "refactorize" if refactor else "factorize"
    del suffix

    @jax.custom_batching.custom_vmap
    def numeric_id(token_id, offsets, columns, csr_values):
        offs_bd, cols_bd, fp = _expand_structure(offsets, columns, 1)
        return _ffi_numeric(token_id, offs_bd, cols_bd, fp, csr_values, op=op,
                            cfg=cfg)

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
        # Collapse one axis and recurse (see the analyze rule): the batched
        # ids are equal copies of the one block entry, so the first id plus
        # the expanded pattern describe the whole level.
        offs_bd, cols_bd, _ = _expand_structure(offsets, columns, axis_size)
        out_id = numeric_id(token_id[0], offs_bd, cols_bd, vals.reshape(-1))
        return jnp.broadcast_to(out_id, (axis_size,) + out_id.shape), True

    return numeric_id


@lru_cache(maxsize=None)
def _make_solve(suffix, cfg):
    del suffix

    @jax.custom_batching.custom_vmap
    def solve_id(token_id, offsets, columns, values, b_values):
        # The base case already absorbs any leading rhs axes as one
        # multi-RHS SOLVE (nrhs is derived from element counts).
        offs_bd, cols_bd, fp = _expand_structure(offsets, columns, 1)
        return _ffi_solve(token_id, offs_bd, cols_bd, fp, values, b_values,
                          cfg=cfg)

    @solve_id.def_vmap
    def _(axis_size, in_batched, token_id, offsets, columns, values, b_values):
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
            mtype_id=1, mview_id=1, device_id=0,
            reordering_id=0, memory_id=0) -> FactorToken:
    """``reordering_id``: cudssReorderingAlg_t — 0 default, 1 btf_colamd,
    2 colamd, 3 amd, 4 nested_dissection, 5 none (natural order).
    ``memory_id``: 0 device (default), 1 hybrid host+device factors."""
    cfg = (int(device_id), int(mtype_id), int(mview_id),
           int(reordering_id), int(memory_id))
    core = _make_analyze_ad(_suffix(jnp.dtype(csr_values.dtype)), cfg)
    return core(csr_values, csr_offsets.astype(jnp.int32),
                csr_columns.astype(jnp.int32))


@lru_cache(maxsize=None)
def _make_analyze_ad(suffix, cfg):
    """The analyze implementation wrapped for autodiff (identity rule)."""
    statics = dict(zip(_CFG_KEYS, cfg))

    def impl(csr_values, csr_offsets, csr_columns):
        dtype = jnp.dtype(csr_values.dtype)
        n = csr_offsets.shape[-1] - 1
        nnz = csr_columns.shape[-1]
        if csr_values.ndim == 2:
            batch_size = int(csr_values.shape[0])
            offs_bd, cols_bd, _ = _expand_structure(csr_offsets, csr_columns,
                                                    batch_size)
            token_id = _make_analyze(suffix, cfg)(
                csr_values.reshape(-1), offs_bd, cols_bd)
            return FactorToken(
                id=token_id, values=csr_values, offsets=csr_offsets,
                columns=csr_columns, phase="analyzed",
                kind="pbatch", dtype=dtype, n=int(n), nnz=int(nnz),
                batch_size=batch_size, **statics)
        token_id = _make_analyze(suffix, cfg)(
            csr_values, csr_offsets, csr_columns)
        return FactorToken(
            id=token_id, values=csr_values, offsets=csr_offsets,
            columns=csr_columns, phase="analyzed",
            kind="single", dtype=dtype, n=int(n), nnz=int(nnz),
            batch_size=1, **statics)

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


def _require_factorized(token, op):
    # phase is static, so misuse fails at trace time, not on the GPU
    if token.phase != "factorized":
        raise ValueError(
            f"spineax tokens: {op} requires a factorized token "
            "(call factorize first)")


def _numeric(token, csr_values, refactor):
    op = "refactorize" if refactor else "factorize"
    if refactor:
        _require_factorized(token, op)
    if jnp.dtype(csr_values.dtype) != token.dtype:
        raise ValueError(
            f"spineax tokens: {op} values dtype {csr_values.dtype} != "
            f"token dtype {token.dtype}")
    if csr_values.shape[-1] != token.nnz:
        raise ValueError(
            f"spineax tokens: {op} values size {csr_values.shape[-1]} != "
            f"token nnz {token.nnz}")
    B = _batch_of(token)
    if (B > 1 or token.id.ndim >= 2) and (
            csr_values.ndim < 2 or
            int(np.prod(csr_values.shape[:-1])) != B):
        raise ValueError(
            f"spineax tokens: {op} on a batch token expects values "
            f"({B}, {token.nnz}) (leading axes flattening to {B}), "
            f"got {csr_values.shape}")
    return _make_numeric_ad(refactor)(token, csr_values)


def _numeric_impl(token, csr_values, refactor):
    B = _batch_of(token)
    if B > 1 or token.id.ndim >= 2:
        # batch entry used eagerly (explicit door, or vmap-minted token used
        # outside vmap): one block-diagonal numeric phase, through the same
        # custom_vmap wrapper so transform-added axes stay collapsible
        offs_bd, cols_bd, _ = _expand_structure(token.offsets,
                                                token.columns, B)
        tid = token.id.reshape(-1)[:1]
        out_id = _make_numeric(_suffix(token.dtype), refactor, _cfg_of(token))(
            tid, offs_bd, cols_bd, csr_values.reshape(-1))
        token_id = jnp.broadcast_to(out_id, token.id.shape)
    else:
        token_id = _make_numeric(_suffix(token.dtype), refactor, _cfg_of(token))(
            token.id, token.offsets, token.columns, csr_values)
    # FRESH id: the numeric phase consumed the input state's name; values
    # leaf swapped to the just-factorized values
    return dataclasses.replace(token, id=token_id, values=csr_values,
                               phase="factorized")


@lru_cache(maxsize=None)
def _make_numeric_ad(refactor):

    def impl(token, csr_values):
        return _numeric_impl(token, csr_values, refactor)

    core = jax.custom_jvp(impl)

    @core.defjvp
    def _(primals, tangents):
        dtoken, dvals = tangents
        out = core(*primals)
        # phase is a static: the tangent pytree must match the output's
        return out, dataclasses.replace(dtoken, values=dvals,
                                        phase="factorized")

    return core


def factorize(token: FactorToken, csr_values) -> FactorToken:
    return _numeric(token, csr_values, refactor=False)


def refactorize(token: FactorToken, csr_values) -> FactorToken:
    return _numeric(token, csr_values, refactor=True)


def solve(token: FactorToken, b, ir_nsteps=None):
    _require_factorized(token, "solve")
    if jnp.dtype(b.dtype) != token.dtype:
        raise ValueError(
            f"spineax tokens: rhs dtype {b.dtype} != token dtype {token.dtype}")
    if b.shape[-1] != token.n:
        raise ValueError(
            f"spineax tokens: rhs trailing dim {b.shape[-1]} != token n {token.n}")
    B = _batch_of(token)
    if (B > 1 or token.id.ndim >= 2) and (b.ndim < 2 or b.shape[-2] != B):
        raise ValueError(
            f"spineax tokens: solve on a batch token expects rhs "
            f"(..., {B}, {token.n}), got {b.shape}")
    nsteps = 0 if ir_nsteps is None else ir_nsteps  # int OR traced scalar
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
    fn = _make_solve(_suffix(token.dtype), _cfg_of(token))
    if B > 1 or token.id.ndim >= 2:
        # batch entry used eagerly: solve the ONE expanded block-diagonal
        # system single-style, through the same custom_vmap wrapper so any
        # transform-added batch axes stay collapsible (multi-RHS)
        offs_bd, cols_bd, _ = _expand_structure(token.offsets,
                                                token.columns, B)
        tid = token.id.reshape(-1)[:1]
        bf = b.reshape(b.shape[:-2] + (-1,))
        out = fn(tid, offs_bd, cols_bd, token.values.reshape(-1), bf)
        return out.reshape(b.shape)
    return fn(token.id, token.offsets, token.columns, token.values, b)


def _refined_solve(token, b, nsteps):
    """``ir_nsteps`` rounds of Richardson refinement, JAX-side.

    cuDSS-internal IR is permanently OFF: its refinement SpMV dereferences
    CSR pointers captured at earlier phase calls (compute-sanitizer:
    out-of-bounds atomics in cudss::spmv_ker once the expanded batch
    structure is a freed XLA temporary). This same-precision loop is what
    cuDSS runs internally anyway, and its residual SpMV consumes its
    expanded indices inside the executable that builds them.

    ``nsteps`` may be TRACED (runtime-varying under one jit trace): the loop
    becomes a fori_loop then, which is legal here because the solve
    callables sit inside custom_linear_solve and are only ever re-traced
    and re-applied by its IFT rules, never differentiated through.
    """
    x = _solve_impl(token, b)
    refine = lambda _, x: x + _solve_impl(token, b - _matvec(token, x))
    if isinstance(nsteps, jax.Array):
        return jax.lax.fori_loop(0, nsteps, refine, x)
    for i in range(nsteps):  # static: unrolled
        x = refine(i, x)
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
                      mtype_id=0, mview_id=0, device_id=token.device_id,
                      reordering_id=token.reordering_id,
                      memory_id=token.memory_id)
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
        outs = _ffi_call4(
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
            token_id)
        return tuple(outs)

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
    _require_factorized(token, "query")
    B = _batch_of(token)
    tree = int(_ps.nd_partition_tree_size())
    fn = _make_query(_suffix(token.dtype), token.dtype, B * token.n, tree)
    tid = token.id.reshape(-1)[:1] if token.id.ndim >= 2 else token.id
    return dict(zip(_QUERY_FIELDS, fn(tid)))


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
    according to SPINEAX_FACTOR_CACHE). Using the token again after release
    self-heals by rebuilding (query excepted), so this frees memory, not the
    token."""
    return bool(_ps.token_release(int(jax.device_get(token.id).ravel()[0])))


def registry_size() -> int:
    """Number of live factorizations in the registry."""
    return int(_ps.token_registry_size())


def cache_capacity() -> int:
    """LRU capacity (``SPINEAX_FACTOR_CACHE``, default 8)."""
    return int(_ps.token_cache_capacity())


def rebuild_count() -> int:
    """Times a phase call rebuilt a non-resident factorization (self-heal).

    A rising count means live tokens are being evicted and re-factorized —
    raise ``SPINEAX_FACTOR_CACHE`` to fit the working set."""
    return int(_ps.token_rebuild_count())


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
    evicting oldest-used entries. Eviction is not fatal — a stated solve
    self-heals by re-factorizing from the token's own arrays (see
    ``rebuild_count``) — it just costs the repeated work.

    For true control the phases are explicit methods that thread tokens,
    mirroring the ``spineax.cudss`` free functions with operator sugar:

        solver = CuDSS()
        token  = solver.analyze(operator)                 # ANALYSIS
        token  = solver.factorize(token, operator)        # FACTORIZATION
        token  = solver.refactorize(token, new_operator)  # REFACTORIZATION
        x      = solver.solve(token, b)                   # SOLVE (repeatable)
        data   = solver.query(token)                      # every cuDSS data item
    """

    # cuDSS knobs (see ``analyze``): reordering algorithm and factor storage
    reordering_id: int = 0
    memory_id: int = 0

    # explicit phases ----------------------------------------------------------

    def analyze(self, operator):
        if lx.is_symmetric(operator):
            mtype_id = 3 if lx.is_positive_semidefinite(operator) else 1
        else:
            mtype_id = 0
        return analyze(operator.values, operator.offsets, operator.columns,
                       mtype_id=mtype_id, mview_id=0,
                       reordering_id=self.reordering_id,
                       memory_id=self.memory_id)

    def factorize(self, token, operator):
        return factorize(token, operator.values)

    def refactorize(self, token, operator):
        return refactorize(token, operator.values)

    def solve(self, token, vector, ir_nsteps=None):
        return solve(token, vector, ir_nsteps=ir_nsteps)

    def query(self, token):
        return query(token)

    # lineax protocol ----------------------------------------------------------

    def init(self, operator, options):
        del options
        return self.factorize(self.analyze(operator), operator)

    def compute(self, state, vector, options):
        # lineax's per-call channel: lx.linear_solve(..., options={"ir_nsteps": k})
        vector, unflatten = jfu.ravel_pytree(vector)
        solution = self.solve(state, vector, ir_nsteps=options.get("ir_nsteps"))
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
                          mtype_id=0, mview_id=0, device_id=state.device_id,
                          reordering_id=state.reordering_id,
                          memory_id=state.memory_id)
        return factorize(t_token, t_token.values), options

    def conj(self, state, options):
        return state, options  # real-valued operators

    def assume_full_rank(self):
        return True

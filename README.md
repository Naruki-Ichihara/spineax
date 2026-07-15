# Spineax (SParse lINear Solvers in JAX)

This repo integrates existing sparse linear solvers into JAX. I currently 
feature a single GPU-based linear solver (with plans to implement more):
- cuDSS

For those that need sparsity pattern detection for jax jacobians/hessians I also offer this [package](https://github.com/johnviljoen/jax2sympy).

I built this repo as part of a project to GPU-batch solve many IPOPT optimizations in [jaxipm](https://github.com/johnviljoen/jaxipm).

## cuDSS

I expose ***most*** features of cuDSS (as of 0.8.0) to JAX with ***zero-copy arrays*** and ***full FFI jit/vmap integration*** including custom batching functionality to expose more information than cuDSS currently supports.

This currently supports:
- ***zero-copies between JAX and cuDSS***
- ***full FFI jit/vmap integration*** ([example](examples/cudss/))
- ***all*** cuDSS ***datatypes*** (F32, F64, C64, C128) ([example](examples/cudss/))
- ✅ ***all*** cuDSS ***solvers*** (general, symmetric, symmetric positive defnite, hermitian, hermitian positive definite) ([example](examples/cudss/))
- ✅ ***all*** cuDSS ***outputs*** ([example](examples/cudss/outputs.py), even in the batched case!)
- ✅ ***Differentiation*** through cuDSS solvers (and even re-use of factorizations when doing so!) ([example](examples/cudss/))
- ✅ Batches of ***heterogeneous sparsity patterns***, and even ***heterogeneous sizes***! ([example](examples/cudss/))

We have also added a ***new Lineax-based API***, which is now the recommended method of interfacing with spineax.

### Examples

* [JAX transformation composability](examples/cudss/composability.py)
* [Testing all available datatypes](examples/cudss/datatypes.py)
* [Seeing all cuDSS auxilliary outputs](examples/cudss/outputs.py)
* [Testing all available solvers (general, sym, herm, spd, hpd)](examples/cudss/solver_types.py)

# Installation

Requirements:
* An NVIDIA GPU of **Turing generation (compute capability 7.5) or newer**
* **CUDA 13**
* **Python 3.12 or newer**
* Linux x86-64 only

pip:
```bash
pip install spineax
```

uv:
```bash
uv pip install spineax
```

> Using a uv-managed project instead? Just `uv add spineax`.


<!-- ## Troubleshooting

### Build fails with `__cudaGetKernel was not declared`

This happens when your system's CUDA toolkit version doesn't match the pip-installed CUDA runtime.
The `pbatch_solve` native module is the whole cuDSS backend and cannot be skipped, so the fix is to
build with a CUDA toolkit that matches the runtime major version of your `jax[cuda12]`/`jax[cuda13]`
install (e.g. via an `nvidia/cuda` container of the right version). -->

# Citation

```
@article{viljoen2026scaling,
  title={Scaling Nonlinear Optimization: Many Problems One GPU},
  author={Viljoen, John and Haffner, Johanna and Tomizuka, Masayoshi and Mehr, Negar},
  journal={arXiv preprint arXiv:2606.26341},
  year={2026}
}
```




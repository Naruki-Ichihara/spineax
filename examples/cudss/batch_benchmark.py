import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # ensure we use GPU 0 for this benchmark
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.devices()  # force CUDA init before spineax import

import jax.numpy as jnp
import scipy.sparse as sp
from spineax.cudss.solver import CuDSSSolver

# Load the linear system (COO format)
data = np.load('/home/john/code/jax_ssids/systems/quad_kkt_systems/iter_1.npz')
coo_indices = data['lhs_coo_indices']  # (5404, 2)
lhs_data = data['lhs_data']            # (5404,)
rhs = data['rhs']                       # (1128,)

n = rhs.shape[0]
rows = coo_indices[:, 0]
cols = coo_indices[:, 1]

# Convert COO to CSR via scipy
coo = sp.coo_matrix((lhs_data, (rows, cols)), shape=(n, n))
csr = coo.tocsr()

csr_offsets = jnp.array(csr.indptr, dtype=jnp.int32)
csr_columns = jnp.array(csr.indices, dtype=jnp.int32)
csr_values = jnp.array(csr.data, dtype=jnp.float64)
b = jnp.array(rhs, dtype=jnp.float64)

print(f"Matrix size: {n}x{n}, nnz: {csr.nnz}")

# Create solver - symmetric matrix, upper triangular view (KKT system)
solver = CuDSSSolver(csr_offsets, csr_columns, device_id=0, mtype_id=1, mview_id=1)

# Single solve to verify correctness
x, inertia = solver(b, csr_values)
residual = jnp.linalg.norm(jnp.array(csr.toarray()) @ x - b)
print(f"Single solve residual: {residual:.2e}")
print(f"Inertia: {inertia}")

# Batch of 2000 - tile the same system
batch_size = 2000
b_batch = jnp.tile(b[None, :], (batch_size, 1))
csr_values_batch = jnp.tile(csr_values[None, :], (batch_size, 1))

# Warm up jit+vmap (includes first factorization)
batched_solver = jax.jit(jax.vmap(solver))
x_batch, inertia_batch = batched_solver(b_batch, csr_values_batch)
jax.block_until_ready(x_batch)
print(f"\nBatch solve output shape: {x_batch.shape}")

# Time it
num_runs = 10
times = []
for i in range(num_runs):
    start = time.perf_counter()
    x_batch, inertia_batch = batched_solver(b_batch, csr_values_batch)
    jax.block_until_ready(x_batch)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

print(f"\nBatch size: {batch_size}")
print(f"Timing over {num_runs} runs:")
print(f"  Mean:      {np.mean(times)*1000:.2f} ms")
print(f"  Min:       {np.min(times)*1000:.2f} ms")
print(f"  Max:       {np.max(times)*1000:.2f} ms")
print(f"  Per solve: {np.mean(times)/batch_size*1e6:.2f} us")

# =============================================================================
# SPD benchmark (Cholesky) - same sparsity pattern, diagonally dominant values
# =============================================================================
print("\n" + "="*60)
print("SPD (Cholesky) benchmark - same sparsity pattern")
print("="*60)

# Build SPD values: take abs of off-diagonals, make diagonal dominant
csr_np = csr.copy()
diag_idx = (csr_np.indices == np.repeat(np.arange(n), np.diff(csr_np.indptr)))
offdiag_vals = np.abs(csr_np.data.copy())
offdiag_vals[diag_idx] = 0.0

# For each row, sum of |off-diag| in upper triangle
row_offdiag_upper = np.zeros(n)
for i in range(n):
    start, end = csr_np.indptr[i], csr_np.indptr[i + 1]
    row_offdiag_upper[i] = np.sum(offdiag_vals[start:end])

# Lower triangle contributions: for upper entry (i,j), row j also gets |val|
row_offdiag_lower = np.zeros(n)
for i in range(n):
    start, end = csr_np.indptr[i], csr_np.indptr[i + 1]
    for k in range(start, end):
        j = csr_np.indices[k]
        if j != i:
            row_offdiag_lower[j] += offdiag_vals[k]

row_offdiag_total = row_offdiag_upper + row_offdiag_lower

# Set values: abs of off-diagonals, diagonal = sum of row off-diags + 1
spd_data = offdiag_vals.copy()
for i in range(n):
    start, end = csr_np.indptr[i], csr_np.indptr[i + 1]
    for k in range(start, end):
        if csr_np.indices[k] == i:
            spd_data[k] = row_offdiag_total[i] + 1.0

csr_values_spd = jnp.array(spd_data, dtype=jnp.float64)

# SPD solver (mtype_id=3), same upper triangular view
solver_spd = CuDSSSolver(csr_offsets, csr_columns, device_id=0, mtype_id=3, mview_id=1)

# Single solve to verify
x_spd, inertia_spd = solver_spd(b, csr_values_spd)
print(f"Single solve inertia: {inertia_spd}")

# Batch
csr_values_spd_batch = jnp.tile(csr_values_spd[None, :], (batch_size, 1))

# Warm up
batched_solver_spd = jax.jit(jax.vmap(solver_spd))
x_batch_spd, _ = batched_solver_spd(b_batch, csr_values_spd_batch)
jax.block_until_ready(x_batch_spd)
print(f"Batch solve output shape: {x_batch_spd.shape}")

# Time it
times_spd = []
for i in range(num_runs):
    start = time.perf_counter()
    x_batch_spd, _ = batched_solver_spd(b_batch, csr_values_spd_batch)
    jax.block_until_ready(x_batch_spd)
    elapsed = time.perf_counter() - start
    times_spd.append(elapsed)

print(f"\nBatch size: {batch_size}")
print(f"Timing over {num_runs} runs:")
print(f"  Mean:      {np.mean(times_spd)*1000:.2f} ms")
print(f"  Min:       {np.min(times_spd)*1000:.2f} ms")
print(f"  Max:       {np.max(times_spd)*1000:.2f} ms")
print(f"  Per solve: {np.mean(times_spd)/batch_size*1e6:.2f} us")

# Comparison
print("\n" + "="*60)
print("Comparison: LDL (symmetric) vs Cholesky (SPD)")
print("="*60)
print(f"  LDL mean:      {np.mean(times)*1000:.2f} ms")
print(f"  Cholesky mean:  {np.mean(times_spd)*1000:.2f} ms")
print(f"  Speedup:        {np.mean(times)/np.mean(times_spd):.2f}x")

# =============================================================================
# Verify Cholesky (SPD solver) fails on the original indefinite KKT matrix
# =============================================================================
print("\n" + "="*60)
print("Cholesky on indefinite matrix - expect failure")
print("="*60)

solver_spd_indef = CuDSSSolver(csr_offsets, csr_columns, device_id=0, mtype_id=3, mview_id=1)
try:
    x_fail, inertia_fail = solver_spd_indef(b, csr_values)
    jax.block_until_ready(x_fail)
    # Check if cuDSS silently produced garbage (non-zero negative inertia means indefinite)
    inertia_fail = np.array(inertia_fail)
    if inertia_fail[1] > 0:
        print(f"PASS: cuDSS detected indefiniteness via inertia: {inertia_fail}")
    else:
        residual_fail = jnp.linalg.norm(jnp.array(csr.toarray() + csr.toarray().T - np.diag(csr.toarray().diagonal())) @ x_fail - b)
        print(f"WARNING: Cholesky did not raise, residual: {residual_fail:.2e}, inertia: {inertia_fail}")
except Exception as e:
    print(f"PASS: Cholesky correctly failed on indefinite matrix: {type(e).__name__}: {e}")

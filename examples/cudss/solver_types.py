"""Example: all five cuDSS matrix types through the token API."""
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax.cudss import tokens as tk


def test_solver_types(mtype_id):

    # example usage
    # -------------
    M1 = jnp.array([
        [4., 0., 1., 0., 0.],
        [0., 3., 2., 0., 0.],
        [0., 0., 5., 0., 1.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 2.],
    ])

    b1 = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0])

    m1 = M1 + M1.T - jnp.diag(M1) * jnp.eye(M1.shape[0])
    true_x1 = jnp.linalg.solve(m1, b1)

    LHS1 = jsparse.BCSR.fromdense(m1)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data

    # we are passing the whole LHS matrix in FULL (mview_id=0), on GPU 0
    token = tk.analyze(csr_values1, csr_offsets1, csr_columns1,
                       mtype_id=mtype_id, mview_id=0, device_id=0)
    token = tk.factorize(token, csr_values1)
    x = tk.solve(token, b1)

    # check out the values of the various things!
    print(f"x: {x}")
    print(f"max err vs dense solve: {jnp.max(jnp.abs(x - true_x1)):.2e}")

mtypes = [
    "general",
    "symmetric",
    "hermitian",
    "symmetric_positive_definite",
    "hermitian_positive_definite"
]

for mtype_id, mtype in enumerate(mtypes):
    print(f"testing: {mtype}")
    test_solver_types(mtype_id)

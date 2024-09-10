"""This file contains a custom solver mixin.

Using this mixin will make sure that PETSc is used if it is available. If it is not
available, the default from porepy's side (either pypardiso or, worst case scenario, the
standard scipy solver) will be used. Another nifty little property of this solver mixin
is that it checks whether the attributes linear_system_residual and
linear_system_jacobian are available. If they are available, we will use the stored
jacobian instead of assembling it on every time step. Note that the custom
run_linear_model function should be used if this is the case. If not, the residual will
be assembled twice per time step.

"""

import logging

import numpy as np
import scipy.sparse as sps

try:
    from petsc4py import PETSc

except ImportError:
    _IS_PETSC4PY_AVAILABLE: bool = False
else:
    _IS_TQDM_AVAILABLE = True

logger = logging.getLogger(__name__)


class CustomSolverMixin:
    def solve_linear_system(self) -> np.ndarray:
        """After calling the parent method, the global solution is calculated by Schur
        expansion."""
        import time

        petsc_solver_q = self.params.get("petsc_solver_q", False)
        tb = time.time()
        if petsc_solver_q and _IS_PETSC4PY_AVAILABLE:
            try:
                csr_mat = self.linear_system_jacobian
                res_g = self.linear_system_residual
            except AttributeError:
                csr_mat, res_g = self.linear_system()

            NDIM = self.nd
            jac_g = PETSc.Mat().createAIJ(
                size=csr_mat.shape,
                csr=((csr_mat.indptr, csr_mat.indices, csr_mat.data)),
                bsize=NDIM,
            )

            # solving ls
            ksp = PETSc.KSP().create()
            options = PETSc.Options()
            options["pc_type"] = "hypre"
            options["pc_hypre_type"] = "boomeramg"
            options["pc_hypre_boomeramg_max_iter"] = 1
            options["pc_hypre_boomeramg_cycle_type"] = "V"
            options["pc_hypre_boomeramg_truncfactor"] = 0.3
            options.setValue("ksp_type", "gmres")
            options.setValue("ksp_rtol", 1e-8)
            options.setValue("ksp_max_it", 20 * 50)
            options.setValue("ksp_gmres_restart", 50)
            options.setValue("ksp_pc_side", "right")
            options.setValue("ksp_norm_type", "unpreconditioned")
            ksp.setFromOptions()

            ksp.setOperators(jac_g)
            b = jac_g.createVecLeft()
            b.array[:] = res_g
            x = jac_g.createVecRight()

            ksp.setConvergenceHistory()
            ksp.solve(b, x)

            sol = x.array
        else:
            try:
                A = self.linear_system_jacobian
                b = self.linear_system_residual
                t_0 = time.time()
                logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
                logger.debug(
                    f"""Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min
                    {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."""
                )

                solver = self.linear_solver
                if solver == "pypardiso":
                    try:
                        from pypardiso import spsolve as sparse_solver  # type: ignore
                    except ImportError:
                        # Fall back on the standard scipy sparse solver.
                        sparse_solver = sps.linalg.spsolve

                    x = sparse_solver(A, b)

                logger.info(f"Solved linear system in {time.time() - t_0:.2e} seconds.")
                sol = np.atleast_1d(x)
            except AttributeError:
                sol = super().solve_linear_system()
        te = time.time()
        return sol

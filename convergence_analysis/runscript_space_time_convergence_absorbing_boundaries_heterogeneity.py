import os
import sys

sys.path.append("../")

import numpy as np
import porepy as pp

import run_models.run_linear_model as rlm
from convergence_analysis.convergence_analysis_models.model_convergence_ABC_heterogeneity import (
    ABCModelHeterogeneous,
)


# Prepare path for generated output files
folder_name = "convergence_analysis_results"
header = "num_cells, num_time_steps, displacement_error, traction_error\n"
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)


class ConvergenceAnalysisHeterogeneity(ABCModelHeterogeneous):
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.125 / 2 ** (self.refinement), "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


heterogeneity_factors = [
    # Homogeneous case
    1,
    # Heterogeneous case
    1 / 2**5,
    # Heterogeneous case
    1 / 2**8,
]
anisotropy_factors_mu_lambda = [
    # Isotropic case
    (1, 0),
    # Anisotropic case
    (1, 1e2),
    # Anisotropic case
    (1, 1e4),
]

for heterogeneity_factor_index in range(0, len(heterogeneity_factors)):
    h_h = heterogeneity_factors[heterogeneity_factor_index]
    for (
        h_mu,
        h_lambda,
    ) in anisotropy_factors_mu_lambda:
        h_mu_ind = anisotropy_factors_mu_lambda.index((h_mu, h_lambda))
        filename = f"errors_heterogeneity_{str(heterogeneity_factor_index)}_mu_lam_{str(h_mu_ind)}.txt"

        filename = os.path.join(output_dir, filename)

        refinements = np.arange(1, 5)
        for refinement_coefficient in refinements:
            if refinement_coefficient == 1:
                with open(filename, "w") as file:
                    file.write(header)
            tf = 15.0
            time_steps = 15 * (2**refinement_coefficient)
            dt = tf / time_steps

            time_manager = pp.TimeManager(
                schedule=[0.0, tf],
                dt_init=dt,
                constant_dt=True,
            )

            lame_lambda, shear_modulus = 0.01, 0.01

            solid_constants = pp.SolidConstants(
                lame_lambda=lame_lambda, shear_modulus=shear_modulus
            )
            material_constants = {"solid": solid_constants}

            anisotropy_constants = {
                "mu_parallel": shear_modulus,
                "mu_orthogonal": shear_modulus * h_mu,
                "lambda_parallel": 0.0,
                "lambda_orthogonal": lame_lambda * h_lambda,
                "volumetric_compr_lambda": lame_lambda,
            }

            params = {
                "time_manager": time_manager,
                "grid_type": "simplex",
                "progressbars": True,
                "folder_name": "heterogeneity",
                "heterogeneity_factor": h_h,
                "heterogeneity_location": 0.5,
                "material_constants": material_constants,
                "anisotropy_constants": anisotropy_constants,
                "symmetry_axis": [0, 1, 0],
                "meshing_kwargs": {"constraints": [0, 1, 2, 3]},
                # "petsc_solver_q": True,
            }

            model = ConvergenceAnalysisHeterogeneity(params)
            model.refinement = refinement_coefficient
            model.filename_path = filename
            rlm.run_linear_model(model, params)

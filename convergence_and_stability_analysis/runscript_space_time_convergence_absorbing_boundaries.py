import os
import sys

sys.path.append("../")

import numpy as np
import porepy as pp

import run_models.run_linear_model as rlm
from convergence_and_stability_analysis.analysis_models.model_convergence_ABC import (
    ABCModel,
)


# Prepare path for generated output files
folder_name = "convergence_analysis_results"
header = "num_cells, num_time_steps, displacement_error, traction_error\n"
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)


class ConvergenceAnalysisHeterogeneity(ABCModel):
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.125 / 2 ** (self.refinement), "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def set_geometry(self):
        """Perturb all internal boundary nodes randomly to ensure an unstructured
        grid."""

        # Choose a seed for reproducibility.
        np.random.seed(42)
        super().set_geometry()

        sd = self.mdg.subdomains()[0]
        h = self.meshing_arguments()["cell_size"]

        inds = sd.get_internal_nodes()
        inds_not_constraint = np.where(sd.nodes[0, inds] != self.heterogeneity_location)
        inds = inds[inds_not_constraint]

        perturbation = 0.1 * h
        signs = np.random.choice([-1, 1], size=len(inds))
        sd.nodes[:2, inds] += perturbation * signs

        sd.compute_geometry()

    def write_pvd_and_vtu(self) -> None:
        """Override method such that pvd and vtu files are not created."""
        self.data_to_export()


heterogeneity_coefficients = [
    # Homogeneous case
    1,
    # # Heterogeneous case
    1 / 2**4,
    # # Heterogeneous case
    1 / 2**8,
]
anisotropy_coefficients = [
    # Isotropic case
    0,
    # Anisotropic case
    1e1,
    1e2,
]

for heterogeneity_factor_index in range(0, len(heterogeneity_coefficients)):
    r_h = heterogeneity_coefficients[heterogeneity_factor_index]
    for r_a in anisotropy_coefficients:
        h_lambda_ind = anisotropy_coefficients.index(r_a)
        filename = f"errors_heterogeneity_{str(heterogeneity_factor_index)}_mu_lam_{str(h_lambda_ind)}.txt"

        filename = os.path.join(output_dir, filename)

        refinements = np.arange(2, 7)
        for refinement_coefficient in refinements:
            if refinement_coefficient == 2:
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
                "mu_orthogonal": shear_modulus,
                "lambda_parallel": 0.0,
                "lambda_orthogonal": lame_lambda * r_a,
                "volumetric_compr_lambda": lame_lambda,
            }

            params = {
                "time_manager": time_manager,
                "grid_type": "simplex",
                "progressbars": True,
                "heterogeneity_factor": r_h,
                "heterogeneity_location": 0.5,
                "material_constants": material_constants,
                "anisotropy_constants": anisotropy_constants,
                "symmetry_axis": [0, 1, 0],
                "meshing_kwargs": {"constraints": [0, 1, 2, 3]},
                "run_type": "vertical_anisotropy",
            }

            model = ConvergenceAnalysisHeterogeneity(params)
            model.refinement = refinement_coefficient
            model.filename_path = filename
            rlm.run_linear_model(model, params)

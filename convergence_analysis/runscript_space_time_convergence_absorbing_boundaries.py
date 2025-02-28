import os
import sys

sys.path.append("../")

import numpy as np
import porepy as pp

import run_models.run_linear_model as rlm

from convergence_analysis.convergence_analysis_models.model_convergence_ABC import (
    ABCModel,
)

# Prepare path for generated output files
folder_name = "convergence_analysis_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

for keyword in ["isotropy", "anisotropy"]:
    # Define anisotropy coefficients based on the keyword
    anisotropy_coefficients = (
        [1] if keyword == "isotropy" else [10, 50, 100, 500, 1000, 10000]
    )

    refinements = np.arange(0, 5)

    # Create class for convergence analysis
    class ConvergenceAnalysisIsotropyAnisotropy(ABCModel):
        def meshing_arguments(self) -> dict:
            cell_size = self.units.convert_units(0.125 / 2 ** (self.refinement), "m")
            mesh_args: dict[str, float] = {"cell_size": cell_size}
            return mesh_args

    # Loop through anisotropy coefficients and refinements
    for anisotropy_coefficient in anisotropy_coefficients:
        # Dynamically generate the filename for the anisotropy coefficient
        if keyword == "anisotropy":
            filename = f"displacement_and_traction_errors_anisotropy_{anisotropy_coefficient}.txt"
        else:
            filename = "displacement_and_traction_errors_isotropy.txt"

        # Prepare the full path for the file (this will be the same for all refinements under the same coefficient)
        filename_path = os.path.join(output_dir, filename)

        # Loop over refinements and write the results for each refinement into the same file
        for refinement_coefficient in refinements:
            # Open the file for writing (write headers only once for the first refinement)
            if refinement_coefficient == 0:
                with open(filename_path, "w") as file:
                    file.write(
                        "num_cells, num_time_steps, displacement_error, traction_error\n"
                    )
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

            # Parameters for the model
            params = {
                "time_manager": time_manager,
                "grid_type": "simplex",
                "manufactured_solution": "unit_test",
                "progressbars": True,
                "folder_name": "unit_test_check",
                "material_constants": material_constants,
                "heterogeneity_location": [0.3, 0.7],
                "symmetry_axis": [0, 1, 0],
                "meshing_kwargs": {"constraints": [0, 1, 2, 3]},
            }

            # Update params based on isotropy or anisotropy
            if keyword == "anisotropy":
                anisotropy_constants = {
                    "mu_parallel": shear_modulus,
                    "mu_orthogonal": shear_modulus * anisotropy_coefficient,
                    "lambda_parallel": 0.0,
                    "lambda_orthogonal": lame_lambda * anisotropy_coefficient,
                    "volumetric_compr_lambda": lame_lambda,
                }
            else:
                anisotropy_constants = {
                    "mu_parallel": shear_modulus,
                    "mu_orthogonal": shear_modulus,
                    "lambda_parallel": 0.0,
                    "lambda_orthogonal": 0.0,
                    "volumetric_compr_lambda": lame_lambda,
                }
            params["anisotropy_constants"] = anisotropy_constants

            # Create model instance and run the linear model
            model = ConvergenceAnalysisIsotropyAnisotropy(params)
            model.refinement = refinement_coefficient
            model.filename_path = filename_path
            rlm.run_linear_model(model, params)

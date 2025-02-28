import os
import sys

sys.path.append("../")

import numpy as np
import porepy as pp
import utils

import run_models.run_linear_model as rlm
from porepy.applications.convergence_analysis import ConvergenceAnalysis

from convergence_analysis.convergence_analysis_models.model_convergence_ABC import (
    ABCModel,
)

# Prepare path for generated output files
folder_name = "convergence_analysis_results"
filename = "displacement_and_traction_errors_absorbing_boundaries.txt"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, filename)

# Plotting/Save figure or not:
save_figure = True


class ExportData:
    def data_to_export(self):
        data = super().data_to_export()
        if self.time_manager.final_time_reached():
            sd = self.mdg.subdomains(dim=self.nd)[0]
            x_cc = sd.cell_centers[0, :]
            time = self.time_manager.time
            cp = self.primary_wave_speed(is_scalar=True)

            # Exact displacement and traction
            u_exact = np.array([np.sin(time - x_cc / cp), np.zeros(len(x_cc))]).ravel(
                "F"
            )

            u, x, y, t = utils.symbolic_representation(model=self)
            _, sigma, _ = utils.symbolic_equation_terms(model=self, u=u, x=x, y=y, t=t)
            T_exact = self.elastic_force(
                sd=sd, sigma_total=sigma, time=self.time_manager.time
            )

            # Approximated displacement and traction
            displacement_ad = self.displacement([sd])
            u_approximate = self.equation_system.evaluate(displacement_ad)
            traction_ad = self.stress([sd])
            T_approximate = self.equation_system.evaluate(traction_ad)

            # Compute error for displacement and traction
            error_displacement = ConvergenceAnalysis.lp_error(
                grid=sd,
                true_array=u_exact,
                approx_array=u_approximate,
                is_scalar=False,
                is_cc=True,
                relative=True,
            )
            error_traction = ConvergenceAnalysis.lp_error(
                grid=sd,
                true_array=T_exact,
                approx_array=T_approximate,
                is_scalar=False,
                is_cc=False,
                relative=True,
            )

            with open(filename, "a") as file:
                file.write(
                    f"{sd.num_cells}, {self.time_manager.time_index}, {error_displacement}, {error_traction}\n"
                )
        return data


class SpatialRefinementModel(ExportData, ABCModel):
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.25 / 2 ** (self.refinement), "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


with open(filename, "w") as file:
    file.write("num_cells, num_time_steps, displacement_error, traction_error\n")

refinements = np.arange(0, 5)
for refinement_coefficient in refinements:
    tf = 15.0
    time_steps = 15 * (2**refinement_coefficient)
    dt = tf / time_steps

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
    )

    solid_constants = pp.SolidConstants(lame_lambda=0.01, shear_modulus=0.01)
    material_constants = {"solid": solid_constants}

    anisotropy_constants = {
        "mu_parallel": 0.01,
        "mu_orthogonal": 0.01,
        "lambda_parallel": 0.0,
        "lambda_orthogonal": 0.0,
        "volumetric_compr_lambda": 0.01,
    }

    params = {
        "time_manager": time_manager,
        "grid_type": "simplex",
        "manufactured_solution": "unit_test",
        "progressbars": True,
        "folder_name": "unit_test_check",
        "material_constants": material_constants,
        "anisotropy_constants": anisotropy_constants,
        "heterogeneity_location": [0.3, 0.7],
        "symmetry_axis": [0, 1, 0],
        "meshing_kwargs": {"constraints": [0, 1, 2, 3]},
    }

    model = SpatialRefinementModel(params)
    model.refinement = refinement_coefficient
    rlm.run_linear_model(model, params)

import os
import sys

sys.path.append("../")

import numpy as np
import porepy as pp

import run_models.run_linear_model as rlm
from convergence_analysis.convergence_analysis_models.model_convergence_ABC_heterogeneity import (
    ABCModelHeterogeneous,
)

from porepy.applications.convergence_analysis import ConvergenceAnalysis

# Prepare path for generated output files
folder_name = "convergence_analysis_results"
filename = "heterogeneity_errors_refactored.txt"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, filename)

class ExportData:
    def data_to_export(self):
        data = super().data_to_export()
        sd = self.mdg.subdomains(dim=self.nd)[0]

        x = sd.cell_centers[0, :]
        L = self.heterogeneity_location

        left_solution, right_solution = self.heterogeneous_analytical_solution()
        left_layer = x <= L
        right_layer = x > L

        vals = np.zeros((self.nd, sd.num_cells))
        vals[0, left_layer] = left_solution[0](x[left_layer], self.time_manager.time)
        vals[0, right_layer] = right_solution[0](x[right_layer], self.time_manager.time)

        data.append((sd, "analytical", vals))

        if self.time_manager.final_time_reached():
            displacement_ad = self.displacement([sd])
            u_approximate = self.equation_system.evaluate(displacement_ad)
            exact_displacement = vals.ravel("F")

            exact_force = self.evaluate_exact_heterogeneous_force(sd=sd)
            force_ad = self.stress([sd])
            approx_force = self.equation_system.evaluate(force_ad)

            error_displacement = ConvergenceAnalysis.lp_error(
                grid=sd,
                true_array=exact_displacement,
                approx_array=u_approximate,
                is_scalar=False,
                is_cc=True,
                relative=True,
            )
            error_traction = ConvergenceAnalysis.lp_error(
                grid=sd,
                true_array=exact_force,
                approx_array=approx_force,
                is_scalar=False,
                is_cc=False,
                relative=True,
            )
            with open(filename, "a") as file:
                file.write(
                    f"{sd.num_cells}, {self.time_manager.time_index}, {error_displacement}, {error_traction}\n"
                )

        return data
    
class SpatialRefinementModel(ExportData, ABCModelHeterogeneous):
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.125 / 2 ** (self.refinement), "m")
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

    params = {
        "time_manager": time_manager,
        "grid_type": "simplex",
        "manufactured_solution": "simply_zero",
        "progressbars": True,
        "folder_name": "pf",
        "heterogeneity_factor": 1 / 2**2,
        "heterogeneity_location": 0.5,
        "material_constants": material_constants,
        "meshing_kwargs": {"constraints": [0, 1, 2, 3]},
    }

    model = SpatialRefinementModel(params)
    model.refinement = refinement_coefficient
    rlm.run_linear_model(model, params)

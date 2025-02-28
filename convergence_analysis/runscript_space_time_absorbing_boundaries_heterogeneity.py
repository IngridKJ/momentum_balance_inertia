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
filename = "heterogeneity_errors_refactored.txt"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, filename)

with open(filename, "w") as file:
    file.write("num_cells, num_time_steps, displacement_error, traction_error\n")
    
class SpatialRefinementModel(ABCModelHeterogeneous):
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.125 / 2 ** (self.refinement), "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

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
        "progressbars": True,
        "folder_name": "pf",
        "heterogeneity_factor": 1 / 2**2,
        "heterogeneity_location": 0.5,
        "material_constants": material_constants,
        "meshing_kwargs": {"constraints": [0, 1, 2, 3]},
    }

    model = SpatialRefinementModel(params)
    model.refinement = refinement_coefficient
    model.filename_path = filename
    rlm.run_linear_model(model, params)

import os
import sys

sys.path.append("../")

import numpy as np
import porepy as pp

import run_models.run_linear_model as rlm
from convergence_analysis.convergence_analysis_models.model_convergence_3D_heterogeneity import (
    ABCModel3D,
)

# Prepare path for generated output files
folder_name = "convergence_analysis_results"
filename = "heterogeneity_errors.txt"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, filename)


class Geometry:
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}
        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})
        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 5.0 / self.units.m
        y = 0.125 / self.units.m
        z = 0.125 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.125 / 8, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

class RandomProperties:
    @property
    def heterogeneity_location(self):
        return self.params.get("heterogeneity_location", 0.5)

    @property
    def heterogeneity_factor(self):
        return self.params.get("heterogeneity_factor", 0.5)


class SpatialRefinementModel(Geometry, RandomProperties, ABCModel3D):
    def data_to_export(self):
        data = super().data_to_export()
        sd = self.mdg.subdomains(dim=self.nd)[0]

        x = sd.cell_centers[0, :]

        left_solution, right_solution = self.heterogeneous_analytical_solution()
        L = self.heterogeneity_location

        left_layer = x < L
        right_layer = x > L

        vals = np.zeros((self.nd, sd.num_cells))

        vals[0, left_layer] = left_solution[0](x[left_layer], self.time_manager.time)
        vals[0, right_layer] = right_solution[0](x[right_layer], self.time_manager.time)

        data.append((sd, "analytical", vals))

        lambda_vals = np.zeros((self.nd, sd.num_cells))  # Initialize with zeros
        lambda_vals[0, :] = self.lambda_vector  # Assign lambda values to the first row
        data.append((sd, "lambda", lambda_vals))

        return data

refinements = [5] # np.arange(0, 5)
for refinement_coefficient in refinements:
    tf = 8
    time_steps = 4*144
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
        "grid_type": "cartesian",
        "manufactured_solution": "simply_zero",
        "progressbars": True,
        "folder_name": "visualization_3D_heterogeneity",
        "heterogeneity_factor": 0.5,
        "heterogeneity_location": 2.5,
        "material_constants": material_constants,
    }

    model = SpatialRefinementModel(params)
    model.refinement = refinement_coefficient
    rlm.run_linear_model(model, params)

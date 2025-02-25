import os
import sys

sys.path.append("../")

import numpy as np
import porepy as pp

import run_models.run_linear_model as rlm
from convergence_analysis.convergence_analysis_models.model_convergence_ABC_heterogeneity import (
    ABCModel,
)

from porepy.applications.convergence_analysis import ConvergenceAnalysis

# Prepare path for generated output files
folder_name = "convergence_analysis_results"
filename = "heterogeneity_errors.txt"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, filename)


class Geometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}
        box.update({"ymin": 0, "ymax": y})
        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.25 / 2 ** (self.refinement), "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def set_polygons(self):
        L = self.heterogeneity_location
        W = self.domain.bounding_box["xmax"]
        H = self.domain.bounding_box["ymax"]
        west = np.array([[L, L], [0.0, H]])
        north = np.array([[L, W], [H, H]])
        east = np.array([[W, W], [H, 0.0]])
        south = np.array([[W, L], [0.0, 0.0]])
        return west, north, east, south

    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        west, north, east, south = self.set_polygons()

        self._fractures = [
            pp.LineFracture(west),
            pp.LineFracture(north),
            pp.LineFracture(east),
            pp.LineFracture(south),
        ]


class RandomProperties:
    @property
    def heterogeneity_location(self):
        return self.params.get("heterogeneity_location", 0.5)

    @property
    def heterogeneity_factor(self):
        return self.params.get("heterogeneity_factor", 0.5)


class SpatialRefinementModel(Geometry, RandomProperties, ABCModel):
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

        if self.time_manager.final_time_reached():
            # displacement_ad = self.displacement([sd])
            # u_approximate = self.equation_system.evaluate(displacement_ad)
            # exact_displacement

            exact_force = self.evaluate_exact_heterogeneous_force(sd=sd)
            force_ad = self.stress([sd])
            approx_force = self.equation_system.evaluate(force_ad)

            error_traction = ConvergenceAnalysis.lp_error(
                grid=sd,
                true_array=exact_force,
                approx_array=approx_force,
                is_scalar=False,
                is_cc=False,
                relative=True,
            )
            with open(filename, "a") as file:
                file.write(f"{sd.num_cells}, {error_traction}\n")

        return data


with open(filename, "w") as file:
    file.write("num_cells, traction_error\n")

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
        "folder_name": "testing_visualization",
        "heterogeneity_factor": 1.0,
        "heterogeneity_location": 0.0,
        "material_constants": material_constants,
        "meshing_kwargs": {"constraints": [0, 1, 2, 3]},
    }

    model = SpatialRefinementModel(params)
    model.refinement = refinement_coefficient
    rlm.run_linear_model(model, params)

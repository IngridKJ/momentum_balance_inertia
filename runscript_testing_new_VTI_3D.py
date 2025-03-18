import os

N_THREADS = "1"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

import sys

import numpy as np
import porepy as pp

sys.path.append("../")
import run_models.run_linear_model as rlm
from models.elastic_wave_equation_abc_linear import DynamicMomentumBalanceABCLinear
from utils.anisotropy_mixins import TransverselyIsotropicTensorMixin


class Geometry:
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = self.units.convert_units(1.0, "m")
        y = self.units.convert_units(1.0, "m")
        z = self.units.convert_units(1.0, "m")
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.05, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def set_polygons(self):
        west = np.array(
            [[0.3, 0.3, 0.3, 0.3], [0.3, 0.7, 0.7, 0.3], [0.3, 0.3, 0.7, 0.7]]
        )
        east = np.array(
            [[0.7, 0.7, 0.7, 0.7], [0.3, 0.7, 0.7, 0.3], [0.3, 0.3, 0.7, 0.7]]
        )
        south = np.array(
            [[0.3, 0.7, 0.7, 0.3], [0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.7, 0.7]]
        )
        north = np.array(
            [[0.3, 0.7, 0.7, 0.3], [0.7, 0.7, 0.7, 0.7], [0.3, 0.3, 0.7, 0.7]]
        )
        bottom = np.array(
            [[0.3, 0.7, 0.7, 0.3], [0.3, 0.3, 0.7, 0.7], [0.3, 0.3, 0.3, 0.3]]
        )
        top = np.array(
            [[0.3, 0.7, 0.7, 0.3], [0.3, 0.3, 0.7, 0.7], [0.7, 0.7, 0.7, 0.7]]
        )
        return west, east, south, north, bottom, top

    def set_fractures_(self):
        west, east, south, north, bottom, top = self.set_polygons()
        self._fractures = [
            pp.PlaneFracture(west),
            pp.PlaneFracture(east),
            pp.PlaneFracture(south),
            pp.PlaneFracture(north),
            pp.PlaneFracture(bottom),
            pp.PlaneFracture(top),
        ]


class ModelSetupSourceInInnerDomain(
    Geometry,
    TransverselyIsotropicTensorMixin,
    DynamicMomentumBalanceABCLinear,
):
    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial velocity values."""
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        z = sd.cell_centers[2, :]

        vals = np.zeros((self.nd, sd.num_cells))

        theta = 1
        lam = 0.125

        # Define x0, y0, z0
        x0 = 0.5
        y0 = 0.5
        z0 = 0.5

        # Compute the common part once
        common_part = theta * np.exp(
            -np.pi**2 * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / lam**2
        )

        # Reuse the common_part in the calculation of vals
        vals[0] = common_part * (x - x0)
        vals[1] = common_part * (y - y0)
        vals[2] = common_part * (z - z0)

        return vals.ravel("F")


time_steps = 100
tf = 0.15
dt = tf / time_steps


time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
)

anisotropy_constants = {
    "mu_parallel": 1,
    "mu_orthogonal": 2,
    "lambda_parallel": 5,
    "lambda_orthogonal": 5,
    "volumetric_compr_lambda": 1,
}

params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "folder_name": "test_constraints_3d",
    "manufactured_solution": "simply_zero",
    "anisotropy_constants": anisotropy_constants,
    "progressbars": True,
    "meshing_kwargs": {"constraints": [0, 1, 2, 3, 4, 5]},
}

model = ModelSetupSourceInInnerDomain(params)
rlm.run_linear_model(model, params)

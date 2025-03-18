import os

N_THREADS = "1"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

import logging
import sys

import numpy as np
import porepy as pp

sys.path.append("../")
import run_models.run_linear_model as rlm
from models import DynamicMomentumBalanceABCLinear
from utils.anisotropy_mixins import TransverselyIsotropicTensorMixin

logger = logging.getLogger(__name__)


class InitialConditionsAndMaterialProperties:
    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial velocity values."""
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        vals = np.zeros((self.nd, sd.num_cells))

        theta = 1
        lam = 0.125
        x0 = 0.5
        y0 = 0.5

        common_part = theta * np.exp(
            -np.pi**2 * ((x - x0) ** 2 + (y - y0) ** 2) / lam**2
        )

        vals[0] = common_part * (x - x0)
        vals[1] = common_part * (y - y0)
        return vals.ravel("F")


class Geometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update(
            {
                "ymin": 0,
                "ymax": y,
            }
        )

        return pp.Domain(box)

    def set_polygons(self):
        west = np.array([[0.3, 0.3], [0.1, 0.9]])
        north = np.array([[0.3, 0.7], [0.9, 0.7]])
        east = np.array([[0.7, 0.7], [0.7, 0.1]])
        south = np.array([[0.7, 0.3], [0.1, 0.1]])
        # west = np.array([[0.3, 0.3], [0.3, 0.7]])
        # north = np.array([[0.3, 0.7], [0.7, 0.7]])
        # east = np.array([[0.7, 0.7], [0.7, 0.3]])
        # south = np.array([[0.7, 0.3], [0.3, 0.3]])
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

    def set_domain(self) -> None:
        x = self.units.convert_units(1.0, "m")
        y = self.units.convert_units(1.0, "m")
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.01, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class ModelSetupFracturedHeterogeneous(
    InitialConditionsAndMaterialProperties,
    Geometry,
    TransverselyIsotropicTensorMixin,
    DynamicMomentumBalanceABCLinear,
): ...


time_steps = 100
tf = 0.25
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
    "meshing_kwargs": {"constraints": [0, 1, 2, 3]},
    "grid_type": "simplex",
    "folder_name": "visualization",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
    "anisotropy_constants": anisotropy_constants,
}

model = ModelSetupFracturedHeterogeneous(params)
rlm.run_linear_model(model, params)

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

# Coarse/Fine variables
coarse = True

# Only export visualization files corresponding to the ones visualized in the article:
limit_file_export = True
times_in_article = [0.05, 0.125]


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
        cell_size = self.units.convert_units(0.1 if coarse else 0.0125, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def set_polygons(self):
        west = np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75, 0.25],
                [0.25, 0.25, 0.75, 0.75],
            ]
        )
        east = np.array(
            [
                [0.75, 0.75, 0.75, 0.75],
                [0.25, 0.75, 0.75, 0.25],
                [0.25, 0.25, 0.75, 0.75],
            ]
        )
        south = np.array(
            [
                [0.25, 0.75, 0.75, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.75, 0.75],
            ]
        )
        north = np.array(
            [
                [0.25, 0.75, 0.75, 0.25],
                [0.75, 0.75, 0.75, 0.75],
                [0.25, 0.25, 0.75, 0.75],
            ]
        )
        bottom = np.array(
            [
                [0.25, 0.75, 0.75, 0.25],
                [0.25, 0.25, 0.75, 0.75],
                [0.25, 0.25, 0.25, 0.25],
            ]
        )
        top = np.array(
            [
                [0.25, 0.75, 0.75, 0.25],
                [0.25, 0.25, 0.75, 0.75],
                [0.75, 0.75, 0.75, 0.75],
            ]
        )
        return west, east, south, north, bottom, top


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

        common_part = theta * np.exp(
            -(np.pi**2) * ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / lam**2
        )

        vals[0] = common_part * (x - 0.5)
        vals[1] = common_part * (y - 0.5)
        vals[2] = common_part * (z - 0.5)

        return vals.ravel("F")


time_steps = 90
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
    "folder_name": "visualization_example_1_source_in_inner",
    "manufactured_solution": "simply_zero",
    "anisotropy_constants": anisotropy_constants,
    "progressbars": True,
    "petsc_solver_q": True,
    # A value of None for times_to_export means that visualization files for all time
    # steps are created and exported.
    "times_to_export": times_in_article if limit_file_export else None,
}

model = ModelSetupSourceInInnerDomain(params)
rlm.run_linear_model(model, params)

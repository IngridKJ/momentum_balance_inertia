import numpy as np
import porepy as pp
from models import DynamicMomentumBalanceABC2
from utils.discard_equations_mixins import RemoveFractureRelatedEquationsMomentumBalance

import os
N_THREADS = '1'
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS

class InitialConditionsAndMaterialProperties:
    def vector_valued_mu_lambda(self):
        """Setting a layered medium."""
        subdomain = self.mdg.subdomains(dim=self.nd)[0]
        z = subdomain.cell_centers[2, :]

        lmbda1 = self.solid.lame_lambda()
        mu1 = self.solid.shear_modulus()

        lmbda2 = self.solid.lame_lambda() * 2
        mu2 = self.solid.shear_modulus() * 2

        lmbda3 = self.solid.lame_lambda() * 3
        mu3 = self.solid.shear_modulus() * 3

        lmbda_vec = np.ones(subdomain.num_cells)
        mu_vec = np.ones(subdomain.num_cells)

        upper_layer = z >= 0.7
        middle_layer = (z < 0.7) & (z >= 0.3)
        bottom_layer = z < 0.3

        lmbda_vec[upper_layer] *= lmbda3
        mu_vec[upper_layer] *= mu3

        lmbda_vec[middle_layer] *= lmbda2
        mu_vec[middle_layer] *= mu2

        lmbda_vec[bottom_layer] *= lmbda1
        mu_vec[bottom_layer] *= mu1

        self.mu_vector = mu_vec
        self.lambda_vector = lmbda_vec

    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial velocity values."""
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        z = sd.cell_centers[2, :]

        vals = np.zeros((self.nd, sd.num_cells))

        theta = 1
        lam = 0.125
        x0 = 0.75
        y0 = 0.5
        z0 = 0.5

        common_part = theta * np.exp(
            -np.pi**2 * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / lam**2
        )

        vals[0] = common_part * (x - x0)
        vals[1] = common_part * (y - y0)
        vals[2] = common_part * (z - z0)

        return vals.ravel("F")


class MyGeometry:
    def meshing_kwargs(self) -> dict:
        """Keyword arguments for md-grid creation.

        Returns:
            Keyword arguments compatible with pp.create_mdg() method.

        """
        meshing_kwargs = {"constraints": [1, 2]}

        return meshing_kwargs

    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update(
            {
                "ymin": 0,
                "ymax": y,
                "zmin": 0,
                "zmax": z,
            }
        )

        return pp.Domain(box)

    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        # Fracture:
        coords_a = [0.2, 0.2, 0.8, 0.8]  # x
        coords_b = [0.2, 0.8, 0.8, 0.2]  # y
        coords_c = [0.8, 0.8, 0.2, 0.2]  # z

        frac_1_points = np.array([coords_a, coords_b, coords_c])
        frac_1 = pp.PlaneFracture(frac_1_points)

        # Constraint, lower
        coords_a = [0, 0, 1, 1]  # x
        coords_b = [0, 1, 1, 0]  # y
        coords_c = [0.3, 0.3, 0.3, 0.3]  # z

        constraint_1_points = np.array([coords_a, coords_b, coords_c])
        constraint_1 = pp.PlaneFracture(constraint_1_points)

        # Constraint, upper
        coords_a = [0, 0, 1, 1]  # x
        coords_b = [0, 1, 1, 0]  # y
        coords_c = [0.7, 0.7, 0.7, 0.7]  # z

        constraint_2_points = np.array([coords_a, coords_b, coords_c])
        constraint_2 = pp.PlaneFracture(constraint_2_points)

        self._fractures = [
            frac_1,
            constraint_1,
            constraint_2,
        ]

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        z = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.025 / self.units.m}
        return mesh_args


class MomentumBalanceModifiedGeometry(
    InitialConditionsAndMaterialProperties,
    MyGeometry,
    RemoveFractureRelatedEquationsMomentumBalance,
    pp.DiagnosticsMixin,
    DynamicMomentumBalanceABC2,
): ...


time_steps = 100
tf = 0.5
dt = tf / time_steps

time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
)


params = {
    "time_manager": time_manager,
    "grid_type": "simplex",
    "folder_name": "simplex_290k",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
    "prepare_simulation": False,
}

model = MomentumBalanceModifiedGeometry(params)
import time

start = time.time()
model.prepare_simulation()
end = time.time() - start
print("Num dofs system, simplex", model.equation_system.num_dofs())
print("Time for prep sim:", end)

pp.run_time_dependent_model(model, params)

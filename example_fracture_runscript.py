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
from models import DynamicMomentumBalanceABC2

from utils.export_contact_states import ExportContactStates


logger = logging.getLogger(__name__)


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        frac_1_points = self.units.convert_units(
            np.array([[0.0, 0.5], [0.25, 0.25]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)
        self._fractures = [frac_1]

        num_fractures = 3
        # np.random.seed(42)  # For reproducibility
        # for _ in range(num_fractures):
        #     start_point = np.random.rand(2) * 0.5
        #     end_point = np.random.rand(2) * 0.5
        #     fracture_points = self.units.convert_units(
        #         np.array([start_point, end_point]), "m"
        #     )
        #     fracture = pp.LineFracture(fracture_points)
        #     self._fractures.append(fracture)

    def set_domain(self) -> None:
        x = self.units.convert_units(0.5, "m")
        y = self.units.convert_units(0.5, "m")
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        # Finest simplex
        cell_size = self.units.convert_units(0.005, "m")
        # Finest Cartesian
        # cell_size = self.units.convert_units(0.0025, "m")
        # Coarse random
        # cell_size = self.units.convert_units(0.05, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class MomentumBalanceModifiedGeometry(
    MyGeometry,
    ExportContactStates,
    DynamicMomentumBalanceABC2,
):
    def _is_nonlinear_problem(self) -> bool:
        """Asserting problem is nonlinear"""
        return True

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for mechanics.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.north, "dir")
        bc.is_neu[:, domain_sides.east + domain_sides.west + domain_sides.south] = False
        bc.is_rob[:, domain_sides.east + domain_sides.west + domain_sides.south] = True

        bc.internal_to_dirichlet(sd)
        self.assign_robin_weight(sd=sd, bc=bc)
        return bc

    def bc_values_displacement(self, boundary_grid):
        values = np.zeros((self.nd, boundary_grid.num_cells))
        cell_centers_x = boundary_grid.cell_centers[0, :]

        # Find the indices of the True entries
        true_indices = np.where((cell_centers_x > 0.15) & (cell_centers_x < 0.35))[0]

        if self.time_manager.time_index < 50:
            values[0, true_indices] = (2 + np.sin(self.time_manager.time)) * 0.01
            values[1, true_indices] = -(2 + np.sin(self.time_manager.time)) * 0.001
        return values.ravel("F") * 0.0

    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial velocity values."""
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        vals = np.zeros((self.nd, sd.num_cells))

        theta = 1
        lam = 0.125 / 2
        x0 = 0.25
        y0 = 0.4

        common_part = theta * np.exp(
            -np.pi**2 * ((x - x0) ** 2 + (y - y0) ** 2) / lam**2
        )

        vals[0] = common_part * (x - x0)
        vals[1] = common_part * (y - y0)

        return vals.ravel("F")


time_steps = 200
tf = 0.5
dt = tf / time_steps

time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
)

first_try_constants = {
    "friction_coefficient": 0.1,
}
material_constants = {
    "solid": pp.SolidConstants(**first_try_constants),
}

params = {
    "time_manager": time_manager,
    "material_constants": material_constants,
    "grid_type": "cartesian",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
    "max_iterations": 50,
}

model = MomentumBalanceModifiedGeometry(params)
pp.run_time_dependent_model(model, params)

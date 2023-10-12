from models import MomentumBalanceTimeDepSource

from utils import body_force_function

import porepy as pp
import numpy as np


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.05 / self.units.m}
        return mesh_args


class BoundaryAndInitialCondBGrids:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        if self.nd == 2:
            bc = pp.BoundaryConditionVectorial(
                sd,
                bounds.north + bounds.south + bounds.east + bounds.west,
                "rob",
            )
        return bc

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Displacement values for the Dirichlet boundary condition.

        Parameters:
            boundary_grid: Boundary grid to evaluate values on.

        Returns:
            An array with shape (boundary_grid.num_cells,) containing the displacement
            values on the provided boundary grid.

        """
        return np.zeros((self.nd, boundary_grid.num_cells)).ravel("F")

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Stress values for the Nirichlet boundary condition.

        Parameters:
            boundary_grid: Boundary grid to evaluate values on.

        Returns:
            An array with shape (boundary_grid.num_cells,) containing the stress values
            on the provided boundary grid.

        """
        return np.ones((self.nd, boundary_grid.num_cells)).ravel("F") * 0.01

    def bc_values_robin(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        return np.ones((self.nd, boundary_grid.num_cells)).ravel("F") * 5

    def before_nonlinear_loop(self) -> None:
        """Update values of external sources."""
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        if self.nd == 2:
            source_func = body_force_function(self)
        elif self.nd == 3:
            source_func = body_force_function(self, is_2D=False)

        mech_source = self.source_values(source_func, sd, t)
        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, iterate_index=0
        )

        # Reset counter for nonlinear iterations.
        self._nonlinear_iteration = 0
        # Update time step size.
        self.ad_time_step.set_value(self.time_manager.dt)
        # Update the boundary conditions to both the time step and iterate solution.
        self.update_time_dependent_ad_arrays()


class ModelBoundaryGridsYay(
    MyGeometry,
    BoundaryAndInitialCondBGrids,
    MomentumBalanceTimeDepSource,
):
    ...


t_shift = 0.0
tf = 5.0
dt = tf / 500.0


time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)


params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "folder_name": "cartesian_testing",
    "manufactured_solution": "bubble",
    "progressbars": True,
}

model = ModelBoundaryGridsYay(params)
pp.run_time_dependent_model(model, params)
a = 5

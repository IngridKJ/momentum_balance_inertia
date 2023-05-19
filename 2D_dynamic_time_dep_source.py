import porepy as pp
import numpy as np

from models import DynamicMomentumBalance

from utils import body_force_func_time


"""
(time)^2 * (coords[0] * coords[1] * (1 - coords[0]) * (1 - coords[1]) * iHat + coords[0] * coords[1] * (1 - coords[0]) * (1 - coords[1]) * jHat)
"""


class NewmarkConstants:
    @property
    def gamma(self) -> float:
        return 0.5

    @property
    def beta(self) -> float:
        return 0.25


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1 / self.units.m
        y = 1 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.01 / self.units.m}
        return mesh_args


class BoundaryAndInitialCond:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "dir",
        )
        return bc

    def initial_acceleration(self, dofs: int) -> np.ndarray:
        """Initial acceleration values."""
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        vals = np.zeros((self.nd, sd.num_cells))
        manufactured_sol = self.params.get("manufactured_solution", "bubble")
        if manufactured_sol == "bubble":
            vals[0] = 2 * x * y * (1 - x) * (1 - y)
            vals[1] = 2 * x * y * (1 - x) * (1 - y)
        elif manufactured_sol == "quad_time":
            vals[0] = 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
            vals[1] = 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        elif manufactured_sol == "quad_space":
            raise NotImplementedError
        return vals.ravel("F") * 0


class Source:
    def before_nonlinear_loop(self) -> None:
        """Update values of external sources."""
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        source_func = body_force_func_time(self)
        mech_source = self.source_values(source_func, sd, t)

        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, iterate_index=0
        )

    def source_values(self, f, sd, t) -> np.ndarray:
        """Function for computing the source values.

        Parameters:
            f: Function depending on time and space for the source term.
            sd: Subdomain where the source term is defined.
            t: Current time in the time-stepping.

        Returns:
            An array of source values.

        """
        vals = np.zeros((self.nd, sd.num_cells))

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        x_val = f[0](x, y, t)
        y_val = f[1](x, y, t)

        cell_volume = sd.cell_volumes

        vals[0] = x_val * cell_volume
        vals[1] = y_val * cell_volume

        return vals.ravel("F")


class MyMomentumBalance(
    NewmarkConstants,
    BoundaryAndInitialCond,
    MyGeometry,
    Source,
    DynamicMomentumBalance,
):
    ...


time_manager = pp.TimeManager(
    schedule=[0, 1e0],
    dt_init=1e-2,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)


solid_constants = pp.SolidConstants(
    {
        "density": 1,
        "lame_lambda": 1,
        "permeability": 1,
        "porosity": 1,
        "shear_modulus": 1,
    }
)

material_constants = {"solid": solid_constants}
params = {
    "time_manager": time_manager,
    "material_constants": material_constants,
    "folder_name": "test_convergence_viz",
    "manufactured_solution": "bubble",
}
model = MyMomentumBalance(params)

pp.run_time_dependent_model(model, params)

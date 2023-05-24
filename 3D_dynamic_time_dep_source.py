import porepy as pp
import numpy as np

from models import DynamicMomentumBalance

from utils import body_force_func_time_3D


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
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})
        box.update({"zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1 / self.units.m
        y = 2 / self.units.m
        z = 1 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.1 / self.units.m}
        return mesh_args


class BoundaryAndInitialCond:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south,
            "dir",
        )
        return bc

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.AdArray:
        values = []
        for sd in subdomains:
            bounds = self.domain_boundary_sides(sd)
            val_loc = np.zeros((self.nd, sd.num_faces))
            value = 1
            val_loc[1, bounds.north] = -value * 1e-5
            val_loc[1, bounds.south] = value * 1e-5 * 0

            values.append(val_loc)

        values = np.array(values)
        values = values.ravel("F")
        return pp.wrap_as_ad_array(values, name="bc_vals_mechanics")

    def initial_acceleration(self, dofs: int) -> np.ndarray:
        """Initial acceleration values."""
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        z = sd.cell_centers[2, :]

        vals = np.zeros((self.nd, sd.num_cells))
        manufactured_sol = self.params.get("manufactured_solution", "bubble")
        if manufactured_sol == "bubble":
            vals[0] = 2 * x * y * z * (1 - x) * (1 - y) * (1 - z)
            vals[1] = 2 * x * y * z * (1 - x) * (1 - y) * (1 - z)
            vals[2] = 2 * x * y * z * (1 - x) * (1 - y) * (1 - z)
        elif manufactured_sol == "quad_time":
            vals[0] = 2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
            vals[1] = 2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
            vals[2] = 2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
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
        source_func = body_force_func_time_3D(self)
        mech_source = self.source_values(source_func, sd, t)

        pp.set_solution_values(
            name="source_mechanics", values=mech_source * 0, data=data, iterate_index=0
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
        z = sd.cell_centers[2, :]

        x_val = f[0](x, y, z, t)
        y_val = f[1](x, y, z, t)
        z_val = f[2](x, y, z, t)

        cell_volume = sd.cell_volumes

        vals[0] = x_val * cell_volume
        vals[1] = y_val * cell_volume
        vals[2] = z_val * cell_volume

        return vals.ravel("F") * 0


class MyMomentumBalance(
    NewmarkConstants,
    BoundaryAndInitialCond,
    MyGeometry,
    Source,
    DynamicMomentumBalance,
):
    ...


time_manager = pp.TimeManager(
    schedule=[0, 4e0],
    dt_init=0.5e-2,
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
    "folder_name": "3D_time_dep_source_2",
    "manufactured_solution": "bubble",
}
model = MyMomentumBalance(params)

pp.run_time_dependent_model(model, params)

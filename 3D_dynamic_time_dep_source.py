import porepy as pp
import numpy as np

from models import DynamicMomentumBalance

from utils import body_force_function
from utils import u_v_a_wrap


"""
ParaView 3D bubble analytical:
(time)^2 * (coords[0] * coords[1] * coords[2] * (1 - coords[0]) * (1 - coords[1]) * (1 - coords[2]) * iHat + coords[0] * coords[1] * coords[2] * (1 - coords[0]) * (1 - coords[1]) * (1 - coords[2]) * jHat + coords[0] * coords[1] * coords[2] * (1 - coords[0]) * (1 - coords[1]) * (1 - coords[2]) * kHat)
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
        x = 1
        y = 1
        z = 1

        x = self.solid.convert_units(x, "m")
        y = self.solid.convert_units(y, "m")
        z = self.solid.convert_units(z, "m")

        self._domain = self.nd_rect_domain(x, y, z)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.1, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class BoundaryAndInitialCond:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north
            + bounds.south
            + bounds.west
            + bounds.east
            + bounds.top
            + bounds.bottom,
            "dir",
        )
        return bc

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.AdArray:
        values = []
        for sd in subdomains:
            bounds = self.domain_boundary_sides(sd)
            val_loc = np.zeros((self.nd, sd.num_faces))
            value = 1
            val_loc[1, bounds.north] = -value * 1e-5 * 0
            val_loc[1, bounds.south] = value * 1e-5 * 0

            values.append(val_loc)

        values = np.array(values)
        values = values.ravel("F")
        return pp.wrap_as_ad_array(values, name="bc_vals_mechanics")

    def initial_acceleration(self, dofs: int) -> np.ndarray:
        """Initial acceleration values."""
        # 3D hardcoded
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        z = sd.cell_centers[2, :]
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        acceleration_function = u_v_a_wrap(self, is_2D=False, return_ddt=True)
        vals[0] = acceleration_function[0](x, y, z, t)
        vals[1] = acceleration_function[1](x, y, z, t)
        vals[2] = acceleration_function[1](x, y, z, t)

        return vals.ravel("F")

    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial acceleration values."""
        # 3D hardcoded
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        z = sd.cell_centers[2, :]
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        velocity_function = u_v_a_wrap(self, is_2D=False, return_dt=True)
        vals[0] = velocity_function[0](x, y, z, t)
        vals[1] = velocity_function[1](x, y, z, t)
        vals[2] = velocity_function[1](x, y, z, t)

        return vals.ravel("F")

    def initial_displacement(self, dofs: int) -> np.ndarray:
        """Initial displacement values."""
        # 3D hardcoded
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        z = sd.cell_centers[2, :]
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        displacement_function = u_v_a_wrap(self, is_2D=False)
        vals[0] = displacement_function[0](x, y, z, t)
        vals[1] = displacement_function[1](x, y, z, t)
        vals[2] = displacement_function[1](x, y, z, t)

        return vals.ravel("F")


class Source:
    def before_nonlinear_loop(self) -> None:
        """Update values of external sources."""
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        source_func = body_force_function(self, is_2D=False)
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
        z = sd.cell_centers[2, :]

        x_val = f[0](x, y, z, t)
        y_val = f[1](x, y, z, t)
        z_val = f[2](x, y, z, t)

        cell_volume = sd.cell_volumes

        vals[0] = x_val * cell_volume
        vals[1] = y_val * cell_volume
        vals[2] = z_val * cell_volume

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
    schedule=[0, 1e-1],
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

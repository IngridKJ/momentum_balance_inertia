import porepy as pp
import numpy as np

from models import DynamicMomentumBalance

from utils import body_force_func_time
from utils import u_func_time
from utils import v_func_time
from utils import a_func_time
from utils import get_solution_values


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
        mesh_args: dict[str, float] = {"cell_size": 0.1 / self.units.m}
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
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        acceleration_function = a_func_time(self)
        vals[0] = acceleration_function[0](x, y, t)
        vals[1] = acceleration_function[1](x, y, t)

        return vals.ravel("F")


class Source:
    def before_nonlinear_loop(self) -> None:
        """Update values of external sources.

        Currently also used for setting exact values for displacement, velocity and
        acceleration for use in debugging/verification/convergence analysis.

        """
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        source_func = body_force_func_time(self)

        u_func = u_func_time(self)
        v_func = v_func_time(self)
        a_func = a_func_time(self)

        mech_source = self.source_values(source_func, sd, t)

        u_vals = self.cell_center_function_evaluation(u_func, sd, t)
        v_vals = self.cell_center_function_evaluation(v_func, sd, t)
        a_vals = self.cell_center_function_evaluation(a_func, sd, t)

        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, iterate_index=0
        )

        pp.set_solution_values(name="u_e", values=u_vals, data=data, iterate_index=0)

        pp.set_solution_values(name="v_e", values=v_vals, data=data, iterate_index=0)

        pp.set_solution_values(name="a_e", values=a_vals, data=data, iterate_index=0)

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

    def cell_center_function_evaluation(self, f, sd, t) -> np.ndarray:
        """Function for computing cell center values for a given function.

        Parameters:
            f: Function depending on time and space.
            sd: Subdomain where cell center values are to be computed.
            t: Current time in the time-stepping.

        Returns:
            An array of function values.

        """
        vals = np.zeros((self.nd, sd.num_cells))

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        x_val = f[0](x, y, t)
        y_val = f[1](x, y, t)

        vals[0] = x_val
        vals[1] = y_val

        return vals.ravel("F")

    def after_simulation(self) -> None:
        """Run at the end of simulation. Can be used for cleanup etc."""
        sd = self.mdg.subdomains(dim=2)[0]
        data = self.mdg.subdomain_data(sd)

        u_e = get_solution_values(name="u_e", data=data, iterate_index=0)
        u_h = get_solution_values(name="u", data=data, iterate_index=0)
        v_e = get_solution_values(name="v_e", data=data, iterate_index=0)
        v_h = get_solution_values(name="velocity", data=data, iterate_index=0)
        a_e = get_solution_values(name="a_e", data=data, iterate_index=0)
        a_h = get_solution_values(name="acceleration", data=data, iterate_index=0)

        cell_volumes = sd.cell_volumes

        norm_u_e = np.sqrt(np.sum(np.sum(u_e * u_e, axis=0) * cell_volumes))
        du = u_h - u_e
        error_u = np.sqrt(np.sum(np.sum(du * du, axis=0) * cell_volumes)) / norm_u_e

        norm_v_e = np.sqrt(np.sum(np.sum(v_e * v_e, axis=0) * cell_volumes))
        dv = v_h - v_e
        error_v = np.sqrt(np.sum(np.sum(dv * dv, axis=0) * cell_volumes)) / norm_v_e

        norm_a_e = np.sqrt(np.sum(np.sum(a_e * a_e, axis=0) * cell_volumes))
        da = a_h - a_e
        error_a = np.sqrt(np.sum(np.sum(da * da, axis=0) * cell_volumes)) / norm_a_e

        print("u_er =", error_u)
        print("v_er =", error_v)
        print("a_er =", error_a)


class MyMomentumBalance(
    NewmarkConstants,
    BoundaryAndInitialCond,
    MyGeometry,
    Source,
    DynamicMomentumBalance,
):
    ...


time_manager = pp.TimeManager(
    schedule=[0, 1.0],
    dt_init=0.05 / 2,
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
    "grid_type": "simplex",
    "material_constants": material_constants,
    "folder_name": "test_convergence_viz",
    "manufactured_solution": "bubble",
}
model = MyMomentumBalance(params)

pp.run_time_dependent_model(model, params)

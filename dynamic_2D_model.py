import porepy as pp
import numpy as np

from models import DynamicMomentumBalance

from utils import body_force_function
from utils import u_v_a_wrap
from utils import get_solution_values
from utils import cell_center_function_evaluation


class NewmarkConstants:
    @property
    def gamma(self) -> float:
        return 0.5

    @property
    def beta(self) -> float:
        return 0.25


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        # 2D hardcoded
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        # 2D hardcoded
        x = 1 / self.units.m
        y = 1 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.25 / 8.0 / self.units.m}
        return mesh_args


class BoundaryAndInitialCond:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        # 2D hardcoded
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "dir",
        )
        return bc

    def initial_displacement(self, dofs: int) -> np.ndarray:
        """Initial displacement values."""
        # 2D hardcoded
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        displacement_function = u_v_a_wrap(self)
        vals[0] = displacement_function[0](x, y, t)
        vals[1] = displacement_function[1](x, y, t)

        return vals.ravel("F")

    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial velocity values."""
        # 2D hardcoded
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        velocity_function = u_v_a_wrap(self, return_dt=True)
        vals[0] = velocity_function[0](x, y, t)
        vals[1] = velocity_function[1](x, y, t)

        return vals.ravel("F")

    def initial_acceleration(self, dofs: int) -> np.ndarray:
        """Initial acceleration values."""
        # 2D hardcoded
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        acceleration_function = u_v_a_wrap(self, return_ddt=True)
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
        source_func = body_force_function(self)
        mech_source = self.source_values(source_func, sd, t)
        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, iterate_index=0
        )

        # For convergence analysis
        u_func = u_v_a_wrap(self)
        v_func = u_v_a_wrap(self, return_dt=True)
        a_func = u_v_a_wrap(self, return_ddt=True)

        u_vals = cell_center_function_evaluation(self, u_func, sd, t)
        v_vals = cell_center_function_evaluation(self, v_func, sd, t)
        a_vals = cell_center_function_evaluation(self, a_func, sd, t)

        pp.set_solution_values(name="u_e", values=u_vals, data=data, iterate_index=0)
        pp.set_solution_values(name="v_e", values=v_vals, data=data, iterate_index=0)
        pp.set_solution_values(name="a_e", values=a_vals, data=data, iterate_index=0)

    def source_values(self, f, sd, t) -> np.ndarray:
        """Computes the integrated source values by the source function.

        Parameters:
            f: Function depending on time and space for the source term.
            sd: Subdomain where the source term is defined.
            t: Current time in the time-stepping.

        Returns:
            An array of source values.

        """
        # 2D hardcoded
        vals = np.zeros((self.nd, sd.num_cells))

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        x_val = f[0](x, y, t)
        y_val = f[1](x, y, t)

        cell_volume = sd.cell_volumes

        vals[0] = x_val * cell_volume
        vals[1] = y_val * cell_volume

        return vals.ravel("F")

    # def after_simulation(self) -> None:
    #     """Run at the end of simulation.

    #     Here only used for computing the error of displacement, velocity and
    #     acceleration.

    #     """
    #     sd = self.mdg.subdomains(dim=2)[0]
    #     data = self.mdg.subdomain_data(sd)

    #     u_e = get_solution_values(name="u_e", data=data, iterate_index=0)
    #     u_h = get_solution_values(name="u", data=data, time_step_index=0)
    #     v_e = get_solution_values(name="v_e", data=data, iterate_index=0)
    #     v_h = get_solution_values(name="velocity", data=data, iterate_index=0)
    #     a_e = get_solution_values(name="a_e", data=data, iterate_index=0)
    #     a_h = get_solution_values(name="acceleration", data=data, iterate_index=0)

    #     cell_volumes = sd.cell_volumes

    #     nc = len(cell_volumes)

    #     u_e = np.array(np.split(u_e, nc)).T
    #     u_h = np.array(np.split(u_h, nc)).T
    #     v_e = np.array(np.split(v_e, nc)).T
    #     v_h = np.array(np.split(v_h, nc)).T
    #     a_e = np.array(np.split(a_e, nc)).T
    #     a_h = np.array(np.split(a_h, nc)).T

    #     norm_u_e = np.sqrt(np.sum(np.sum(u_e * u_e, axis=0) * cell_volumes))
    #     du = u_h - u_e
    #     error_u = np.sqrt(np.sum(np.sum(du * du, axis=0) * cell_volumes))  # / norm_u_e

    #     norm_v_e = np.sqrt(np.sum(np.sum(v_e * v_e, axis=0) * cell_volumes))
    #     dv = v_h - v_e
    #     error_v = np.sqrt(np.sum(np.sum(dv * dv, axis=0) * cell_volumes))  # / norm_v_e

    #     norm_a_e = np.sqrt(np.sum(np.sum(a_e * a_e, axis=0) * cell_volumes))
    #     da = a_h - a_e
    #     error_a = np.sqrt(np.sum(np.sum(da * da, axis=0) * cell_volumes))  # / norm_a_e

    #     # print("Errors at time =", self.time_manager.time)
    #     # print("u_er =", error_u)
    #     # print("v_er =", error_v)
    #     # print("a_er =", error_a)


class MyMomentumBalance(
    NewmarkConstants,
    BoundaryAndInitialCond,
    MyGeometry,
    Source,
    DynamicMomentumBalance,
):
    ...


t_shift = 0.0
dt = 1.0 / 10.0
tf = 1.0

time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
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

if __name__ == "__main__":
    pp.run_time_dependent_model(model, params)
    a = 5

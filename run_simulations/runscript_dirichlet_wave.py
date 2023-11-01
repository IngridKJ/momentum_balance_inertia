"""Do not touch this file. It is used for convergence analysis of the ABCs."""
import porepy as pp
import numpy as np

from model_one_orthogonal_wave import BaseScriptModel
from porepy.applications.convergence_analysis import ConvergenceAnalysis

from utils import get_boundary_cells
from utils import u_v_a_wrap

with open("error_stuff.txt", "w") as file:
    pass


class EntireDomainWave:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        value = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F")

        bounds = self.domain_boundary_sides(sd)

        robin_weight_shear = self.robin_weight_value(direction="shear")
        robin_weight_tensile = self.robin_weight_value(direction="tensile")

        # Assigning tensile weight to the boundaries who have x-direction as tensile
        # direction.
        value[0][0][bounds.east] *= robin_weight_tensile

        # Assigning shear weight to the boundaries who have y-direction as shear
        # direction.
        value[1][1][bounds.east] *= robin_weight_shear

        # Choosing type of boundary condition for the different domain sides.
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.east + bounds.west,
            "rob",
        )

        # Only the eastern boundary will be Robin (absorbing)
        bc.is_rob[:, bounds.west] = False

        # Western boundary is Dirichlet
        bc.is_dir[:, bounds.west] = True

        bc.robin_weight = value
        return bc

    def bc_values_robin(self, bg: pp.BoundaryGrid) -> np.ndarray:
        face_areas = bg.cell_volumes
        data = self.mdg.boundary_grid_data(bg)

        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)
        if self.time_manager.time_index > 1:
            sd = bg.parent
            displacement_boundary_operator = self.boundary_displacement([sd])
            displacement_values = displacement_boundary_operator.evaluate(
                self.equation_system
            ).val

            displacement_values = bg.projection(self.nd) @ displacement_values

        elif self.time_manager.time_index == 1:
            displacement_values = pp.get_solution_values(
                name="bc_robin", data=data, time_step_index=0
            )

        elif self.time_manager.time_index == 0:
            return self.initial_condition_bc(bg)

        displacement_values = np.reshape(
            displacement_values, (self.nd, bg.num_cells), "F"
        )

        # Values for the absorbing boundary
        values[0][bounds.east] += (
            self.robin_weight_value(direction="tensile", side="east")
            * displacement_values[0][bounds.east]
        ) * face_areas[bounds.east]

        return values.ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        t = self.time_manager.time

        displacement_values = np.zeros((self.nd, bg.num_cells))

        # Time dependent sine Dirichlet condition
        values[0][bounds.west] += np.ones(
            len(displacement_values[0][bounds.west])
        ) * np.sin(t)

        return values.ravel("F")

    def initial_condition_bc(self, bg):
        """Assigning initial bc values."""
        sd = bg.parent

        t = 0
        x = sd.face_centers[0, :]
        y = sd.face_centers[1, :]

        inds_east = get_boundary_cells(self=self, sd=sd, side="east", return_faces=True)
        inds_west = get_boundary_cells(self=self, sd=sd, side="west", return_faces=True)

        bc_vals = np.zeros((sd.dim, sd.num_faces))

        displacement_function = u_v_a_wrap(model=self)

        # East
        bc_vals[0, :][inds_east] = displacement_function[0](
            x[inds_east], y[inds_east], t
        )

        # West
        bc_vals[0, :][inds_west] = displacement_function[0](
            x[inds_west], y[inds_west], t
        )

        bc_vals = bg.projection(self.nd) @ bc_vals.ravel("F")

        return bc_vals


class MyGeometry7Meter:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 10.0 / self.units.m
        y = 10.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.25 / self.units.m}
        return mesh_args


class Exporting:
    def data_to_export(self):
        """Define the data to export to vtu.

        Returns:
            list: List of tuples containing the subdomain, variable name,
            and values to export.

        """
        data = super().data_to_export()
        for sd in self.mdg.subdomains(dim=self.nd):
            x = sd.cell_centers[0, :]
            t = self.time_manager.time
            cp = self.primary_wave_speed

            d = self.mdg.subdomain_data(sd)

            u_e = np.sin(t - x / cp)
            u_h = pp.get_solution_values(name="u", data=d, time_step_index=0)

            u_h = np.reshape(u_h, (self.nd, sd.num_cells), "F")[0]

            error = ConvergenceAnalysis.l2_error(
                grid=sd,
                true_array=u_e,
                approx_array=u_h,
                is_scalar=True,
                is_cc=True,
                relative=True,
            )

            data.append((sd, "analytical", u_e))
            data.append((sd, "diff", u_h - u_e))

            if t * cp >= 10:
                with open("error_stuff.txt", "a") as file:
                    file.write(f"{error},")

        return data


class Model(
    EntireDomainWave,
    MyGeometry7Meter,
    Exporting,
    BaseScriptModel,
):
    ...


t_shift = 0.0
tf = 20.0
time_steps = 6400.0
dt = tf / time_steps


time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

time_manager.time_steps = time_steps

params = {
    "time_manager": time_manager,
    "grid_type": "simplex",
    "folder_name": "testing_visualization_orthogonal_",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
    "write_errors": True,
}

model = Model(params)

pp.run_time_dependent_model(model, params)

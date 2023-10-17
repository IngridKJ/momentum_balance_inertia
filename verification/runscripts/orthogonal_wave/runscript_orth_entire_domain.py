import porepy as pp
import numpy as np

from base_script import BaseScriptModel

from utils import get_boundary_cells
from utils import u_v_a_wrap

from plot_l2_error import plot_the_error

with open("errors.txt", "w") as file:
    pass


class EntireDomainWave:
    def source_values(self, f, sd, t) -> np.ndarray:
        """Computes the integrated source values by the source function.

        Parameters:
            f: Function depending on time and space for the source term.
            sd: Subdomain where the source term is defined.
            t: Current time in the time-stepping.

        Returns:
            An array of source values.

        """
        vals = np.zeros((self.nd, sd.num_cells))
        return vals.ravel("F")

    def initial_condition_bc(self, bg):
        """Assigning initial bc values."""
        t = 0
        sd = bg.parent
        x = sd.face_centers[0, :]
        y = sd.face_centers[1, :]

        inds_east = get_boundary_cells(self=self, sd=sd, side="east", return_faces=True)

        bc_vals = np.zeros((self.nd, sd.num_faces))

        displacement_function = u_v_a_wrap(model=self)

        # East
        bc_vals[0, :][inds_east] = displacement_function[0](
            x[inds_east], y[inds_east], t
        )
        bc_vals[1, :][inds_east] = displacement_function[1](
            x[inds_east], y[inds_east], t
        )

        bc_vals = bg.projection(self.nd) @ bc_vals.ravel("F")
        return bc_vals


class ExportErrors:
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
            disp_vals = pp.get_solution_values(name="u", data=d, time_step_index=0)

            u_h = np.reshape(disp_vals, (self.nd, sd.num_cells), "F")[0]

            u_e = np.sin(t - x / cp)

            du = u_e - u_h

            relative_l2_error = pp.error_computation.l2_error(
                grid=sd,
                true_array=u_e,
                approx_array=u_h,
                is_scalar=True,
                is_cc=True,
                relative=True,
            )

            data.append((sd, "diff_anal_num", du))
            data.append((sd, "analytical_solution", u_e))

            if self.params.get("write_errors", False):
                with open("errors.txt", "a") as file:
                    file.write("," + str(relative_l2_error))
                if (
                    int(self.time_manager.time_final / self.time_manager.dt)
                ) == self.time_manager.time_index:
                    plot_the_error("errors.txt", write_stats=True)

        return data


class MyGeometry7Meter:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 5.0 / self.units.m
        y = 5.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.2 / self.units.m}
        return mesh_args


class Model(
    EntireDomainWave,
    MyGeometry7Meter,
    # ExportErrors,
    BaseScriptModel,
):
    ...


t_shift = 0.0
tf = 20.0
time_steps = 200
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
    "grid_type": "cartesian",
    "folder_name": "testing_visualization",
    "manufactured_solution": "unit_test",
    "progressbars": True,
    "write_errors": True,
}

model = Model(params)

pp.run_time_dependent_model(model, params)

# dx = model.meshing_arguments()["cell_size"]
# dt = time_manager.dt
# cp = model.primary_wave_speed

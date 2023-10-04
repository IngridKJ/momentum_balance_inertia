import porepy as pp
import numpy as np

from model_one_orthogonal_wave import BaseScriptModel

from utils import get_boundary_cells


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

    def initial_condition(self):
        """Assigning initial bc values."""
        super().initial_condition()

        sd = self.mdg.subdomains(dim=self.nd)[0]
        data = self.mdg.subdomain_data(sd)

        cp = self.primary_wave_speed
        t = 0
        x = sd.face_centers[0, :]
        y = sd.face_centers[1, :]

        inds_east = get_boundary_cells(self=self, sd=sd, side="east", return_faces=True)
        inds_west = get_boundary_cells(self=self, sd=sd, side="west", return_faces=True)

        bc_vals = np.zeros((sd.dim, sd.num_faces))

        # East
        bc_vals[0, :][inds_east] = np.sin(
            t - (x[inds_east] * np.cos(0) + y[inds_east] * np.sin(0)) / (cp)
        )

        # West
        bc_vals[0, :][inds_west] = np.sin(
            t - (x[inds_west] * np.cos(0) + y[inds_west] * np.sin(0)) / (cp)
        )

        bc_vals = bc_vals.ravel("F")

        pp.set_solution_values(
            name=self.bc_values_mechanics_key,
            values=bc_vals,
            data=data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name=self.bc_values_mechanics_key,
            values=bc_vals,
            data=data,
            iterate_index=0,
        )


class MyGeometry7Meter:
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


class Model(EntireDomainWave, MyGeometry7Meter, BaseScriptModel):
    ...


t_shift = 0.0
tf = 0.8 * 4
time_steps = 2304
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
    "folder_name": "testing_visualization_",
    "manufactured_solution": "unit_test",
    "progressbars": True,
    "write_errors": True,
}

model = Model(params)

pp.run_time_dependent_model(model, params)

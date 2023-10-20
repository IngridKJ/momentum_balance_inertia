import porepy as pp
import numpy as np

from model_one_orthogonal_wave import BaseScriptModel

from utils import get_boundary_cells
from utils import u_v_a_wrap


class EntireDomainWave:
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
        mesh_args: dict[str, float] = {"cell_size": 0.2 / self.units.m}
        return mesh_args


class Model(EntireDomainWave, MyGeometry7Meter, BaseScriptModel):
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
    "folder_name": "testing_visualization_orthogonal",
    "manufactured_solution": "unit_test",
    "progressbars": True,
    "write_errors": True,
}

model = Model(params)

pp.run_time_dependent_model(model, params)

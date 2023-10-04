import porepy as pp
import numpy as np

from model_one_diagonal_wave import BaseScriptModel


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


class TestSetup(
    MyGeometry,
    BaseScriptModel,
):
    @property
    def rotation_angle(self) -> float:
        return np.pi / 10


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

params = {
    "time_manager": time_manager,
    "grid_type": "simplex",
    "folder_name": "testing_visualization_",
    "manufactured_solution": "diag_wave",
    "progressbars": True,
}

model = TestSetup(params)

pp.run_time_dependent_model(model, params)

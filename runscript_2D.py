import porepy as pp

import numpy as np

from models import DynamicMomentumBalanceABC2


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
        mesh_args: dict[str, float] = {"cell_size": 0.01 / self.units.m}
        return mesh_args


class MomentumBalanceModifiedGeometry(
    MyGeometry,
    DynamicMomentumBalanceABC2,
):
    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial velocity values."""
        sd = self.mdg.subdomains()[0]
        t = self.time_manager.time

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        vals = np.zeros((self.nd, sd.num_cells))

        theta = 1
        lam = 0.125

        common_part = theta * np.exp(
            -np.pi**2 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / lam**2
        )

        vals[0] = common_part * (x - 0.5)

        vals[1] = common_part * (y - 0.5)

        return vals.ravel("F")


t_shift = 0.0
time_steps = 100
tf = 0.45
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
    # "grid_type": "simplex",
    "grid_type": "cartesian",
    "folder_name": "center_source",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
}

model = MomentumBalanceModifiedGeometry(params)
pp.run_time_dependent_model(model, params)

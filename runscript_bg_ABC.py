from models import MomentumBalanceABC
import porepy as pp

import numpy as np


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


class Model(MyGeometry, MomentumBalanceABC):
    ...


t_shift = 0.0
tf = 5.0
dt = tf / 500.0


time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)


params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "folder_name": "cartesian_testing",
    "manufactured_solution": "bubble",
    "progressbars": True,
}

model = Model(params)
pp.run_time_dependent_model(model, params)

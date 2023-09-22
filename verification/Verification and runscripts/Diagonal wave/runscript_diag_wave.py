import porepy as pp
import numpy as np

from base_script_diag import BaseScriptModel


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 5.0 / self.units.m
        y = 5.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.05 / self.units.m}
        return mesh_args


class TestSetup(
    MyGeometry,
    BaseScriptModel,
):
    @property
    def rotation_angle(self) -> float:
        return np.pi / 6


t_shift = 0.0
tf = 5.0
time_steps = 500
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
    "grid_type": "cartesian",
    "folder_name": "testing_visualization",
    "manufactured_solution": "diag_wave",
    "progressbars": True,
}

model = TestSetup(params)

import os
import sys

runscript_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
model.filename = f"ts{time_steps}_{runscript_name}.txt"

pp.run_time_dependent_model(model, params)

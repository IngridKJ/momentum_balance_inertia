import porepy as pp

import numpy as np

from models import MomentumBalanceTimeDepSource


class MyGeometry:
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        z = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.155 / 2**3 / self.units.m}
        return mesh_args


class MomentumBalanceModifiedGeometry(
    MyGeometry,
    MomentumBalanceTimeDepSource,
):
    ...


t_shift = 0.0
time_steps = 1.0
tf = 1.0
dt = tf / time_steps


time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "density": 1.0,
        "lame_lambda": 1.0e0,
        "shear_modulus": 1.0e0,
    }
)

material_constants = {"solid": solid_constants}

params = {
    "time_manager": time_manager,
    "grid_type": "simplex",
    # "grid_type": "cartesian",
    "material_constants": material_constants,
    "folder_name": "center_source",
    "manufactured_solution": "bubble",
    "progressbars": True,
}

model = MomentumBalanceModifiedGeometry(params)
pp.run_time_dependent_model(model, params)

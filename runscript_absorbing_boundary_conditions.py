import porepy as pp
import numpy as np

from models import MomentumBalanceABC


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 100.0 / self.units.m
        y = 500.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 1.0 / self.units.m}
        return mesh_args


class MomentumBalanceABCModifiedGeometry(MyGeometry, MomentumBalanceABC):
    ...


t_shift = 0.0
tf = 1.0
dt = tf / 100.0

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
        "lame_lambda": 1.0,
        "shear_modulus": 1.0,
    }
)

material_constants = {"solid": solid_constants}

params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "material_constants": material_constants,
    "folder_name": "testing_2",
    "manufactured_solution": "unit_test",
    "progressbars": True,
}

model = MomentumBalanceABCModifiedGeometry(params)
pp.run_time_dependent_model(model, params)

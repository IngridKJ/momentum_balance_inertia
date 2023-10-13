import porepy as pp

from base_script_fixed_time import time_manager_tf5_ts100
from base_script import BaseScriptModel


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
        mesh_args: dict[str, float] = {"cell_size": 0.1 / self.units.m}
        return mesh_args


class Model(MyGeometry7Meter, BaseScriptModel):
    ...


time_manager = time_manager_tf5_ts100

params = {
    "time_manager": time_manager,
    "grid_type": "simplex",
    "folder_name": "testing_visualization",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
    "write_errors": False,
}

model = Model(params)

pp.run_time_dependent_model(model, params)

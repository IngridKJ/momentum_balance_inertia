import porepy as pp
import numpy as np

from base_script import BaseScriptModel


class MyGeometry7Meter:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 7.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.5**4 / self.units.m}
        return mesh_args


class MyGeometry15Meter:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 15.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.5**4 / self.units.m}
        return mesh_args


class FixedGridTestModel7Meter(MyGeometry7Meter, BaseScriptModel):
    ...


class FixedGridTestModel7MeterRefined(MyGeometry7Meter, BaseScriptModel):
    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.5**7 / self.units.m}
        return mesh_args


class FixedGridTestModel15Meter(MyGeometry15Meter, BaseScriptModel):
    ...

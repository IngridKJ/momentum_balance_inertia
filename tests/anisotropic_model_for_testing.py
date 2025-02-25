import sys

import numpy as np
import porepy as pp

sys.path.append("../")

from models import DynamicMomentumBalanceABC
from utils import TransverselyIsotropicTensorMixin


class AnisotropyModelForTesting(
    TransverselyIsotropicTensorMixin,
    DynamicMomentumBalanceABC,
): 
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = self.units.convert_units(5.0, "m")
        y = self.units.convert_units(5.0, "m")
        z = self.units.convert_units(5.0, "m")
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def set_polygons(self):
        west = np.array([[1, 1, 1, 1], [1, 4, 4, 1], [1, 1, 4, 4]])
        east = np.array([[4, 4, 4, 4], [1, 4, 4, 1], [1, 1, 4, 4]])
        south = np.array([[1, 4, 4, 1], [1, 1, 1, 1], [1, 1, 4, 4]])
        north = np.array([[1, 4, 4, 1], [4, 4, 4, 4], [1, 1, 4, 4]])
        bottom = np.array([[1, 4, 4, 1], [1, 1, 4, 4], [1, 1, 1, 1]])
        top = np.array([[1, 4, 4, 1], [1, 1, 4, 4], [4, 4, 4, 4]])
        return west, east, south, north, bottom, top

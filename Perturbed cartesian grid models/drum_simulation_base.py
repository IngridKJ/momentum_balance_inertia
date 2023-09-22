import porepy as pp

import numpy as np

import sys

sys.path.append("../")

from models import MomentumBalanceTimeDepSource


class SolutionStratABC:
    def source_values(self, f, sd, t) -> np.ndarray:
        """Computes the integrated source values by the source function.

        Parameters:
            f: Function depending on time and space for the source term.
            sd: Subdomain where the source term is defined.
            t: Current time in the time-stepping.

        Returns:
            An array of source values.

        """
        cell_volume = sd.cell_volumes
        vals = np.zeros((self.nd, sd.num_cells))

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        x_val = f[0](x, y, t)
        y_val = f[1](x, y, t)

        vals[0] = x_val * cell_volume
        vals[1] = y_val * cell_volume
        return vals.ravel("F")


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
        mesh_args: dict[str, float] = {"cell_size": 0.0125 / self.units.m}
        return mesh_args


class BaseClass(
    MyGeometry,
    SolutionStratABC,
    MomentumBalanceTimeDepSource,
):
    ...

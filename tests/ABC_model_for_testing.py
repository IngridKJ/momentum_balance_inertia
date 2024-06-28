import sys

import numpy as np
import porepy as pp

sys.path.append("../")

from models.elastic_wave_equation_abc_linear import DynamicMomentumBalanceABC2Linear


class GeometryAndInitialCondition:
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

    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = self.solid.convert_units(1.0, "m")
        y = self.solid.convert_units(1.0, "m")
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.05, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class MomentumBalanceABCForTesting(
    GeometryAndInitialCondition,
    DynamicMomentumBalanceABC2Linear,
): ...

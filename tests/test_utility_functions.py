"""Very preliminary testing, not very nicely done at the moment. But it is a start.

Testing fetch inner domain cells of a 5m by 5m cartesian domain with 25 cells, where
the inner domain is 3 cells wide."""

import sys

sys.path.append("../")

import numpy as np
import porepy as pp
from anisotropic_model_for_testing import AnisotropyModelForTesting
from utils import inner_domain_cells


class Model(AnisotropyModelForTesting):
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = self.solid.convert_units(5.0, "m")
        y = self.solid.convert_units(5.0, "m")
        z = self.solid.convert_units(5.0, "m")
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


params = {
    "grid_type": "cartesian",
    "manufactured_solution": "simply_zero",
    "inner_domain_width": 3,
}

model = Model(params)

# Set geometry and discretization parameters
model.set_materials()
model.set_geometry()


def test_inner_domain_cells():
    sd = model.mdg.subdomains(dim=3)[0]

    correct_inner_domain_cells = np.array(
        [
            31,
            32,
            33,
            36,
            37,
            38,
            41,
            42,
            43,
            56,
            57,
            58,
            61,
            62,
            63,
            66,
            67,
            68,
            81,
            82,
            83,
            86,
            87,
            88,
            91,
            92,
            93,
        ]
    )

    # Sort before the comparison just in case the cell numbers are in different order.
    correct_inner_domain_cells_sorted = np.sort(correct_inner_domain_cells)
    inner_domain_cells_method = np.sort(inner_domain_cells(self=model, sd=sd, width=3))

    assert np.all(correct_inner_domain_cells_sorted == inner_domain_cells_method)

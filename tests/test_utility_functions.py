"""Very preliminary testing, not very nicely done at the moment. But it is a start.

Testing fetch inner domain cells of a 5m by 5m cartesian domain with 25 cells, where the
inner domain is 3 cells wide. The test is performed by use of two different functions
for fetching the inner domain cells."""

import sys

sys.path.append("../")

import numpy as np

from anisotropic_model_for_testing import AnisotropyModelForTesting

from utils import inner_domain_cells, use_constraints_for_inner_domain_cells


def test_inner_domain_cells():
    params = {
        "grid_type": "cartesian",
        "manufactured_solution": "simply_zero",
        "inner_domain_width": 3,
    }

    model = AnisotropyModelForTesting(params)

    # Set geometry and discretization parameters
    model.set_materials()
    model.set_geometry()
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

    # Sorting before the comparison just in case the cell numbers are in different 
    # order.
    correct_inner_domain_cells_sorted = np.sort(correct_inner_domain_cells)

    # Inner domain cells by two different methods:
    inner_domain_cells_method = np.sort(inner_domain_cells(self=model, sd=sd, width=3))
    inner_domain_cells_by_polygons_method = np.sort(
        use_constraints_for_inner_domain_cells(self=model, sd=sd)
    )

    assert np.all(correct_inner_domain_cells_sorted == inner_domain_cells_method)
    assert np.all(
        correct_inner_domain_cells_sorted == inner_domain_cells_by_polygons_method
    )

import porepy as pp

import numpy as np

from drum_simulation_base import BaseClass
from utils import get_boundary_cells


class BoundaryPerturbedGeometry:
    def set_geometry(self):
        """Perturb a "boundary" node.

        Perturbes one arbitrary west boundary cell node. The node that is perturbed is
        not actually a boundary node, but rather one of the in-domain nodes of a
        boundary cell.

        """
        super().set_geometry()
        sd = self.mdg.subdomains()[0]
        h = self.meshing_arguments()["cell_size"]

        inds = sd.get_internal_nodes()
        nodes = sd.cell_nodes()

        # Here follows the most brute force way of fecthing a non-corner "internal"
        # western boundary cell node that I could think of. Please ignore.
        cont = True
        cells_south = get_boundary_cells(self, sd, side="south")
        cells_north = get_boundary_cells(self, sd, side="north")
        cell_number = np.random.choice(get_boundary_cells(self, sd, side="west"))
        while cont == True:
            if cell_number in cells_south or cell_number in cells_north:
                cell_number = np.random.choice(
                    get_boundary_cells(self, sd, side="west")
                )
            else:
                cont = False

        boundary_cell_node_indices = [
            i for i, x in enumerate(nodes[:, cell_number]) if x
        ]

        non_boundary_nodes_of_cell = [
            i for i in boundary_cell_node_indices if i in inds
        ]
        non_boundary_node = non_boundary_nodes_of_cell[0]

        sd.nodes[:2, non_boundary_node] += (np.random.rand(1) - 0.5) * h / 4
        sd.compute_geometry()


class BoundaryPerturbedModel(
    BoundaryPerturbedGeometry,
    BaseClass,
):
    ...


t_shift = 0.0
tf = 5.0
dt = tf / 750.0


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
    "folder_name": "perturbed_node",
    "manufactured_solution": "drum_solution",
    "progressbars": True,
}

model = BoundaryPerturbedModel(params)
pp.run_time_dependent_model(model, params)

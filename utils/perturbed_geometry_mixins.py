"""File for perturbed geometry mixins. 

Currently, the following geometries are found here:
    * Geometry where all nodes are perturbed.
    * Geometry where only one node among the "second closest" to the boundary is
      perturbed.
    * Geometry were one internal node is perturbed.
    
"""

import porepy as pp
import numpy as np

from . import get_boundary_cells


class AllPerturbedGeometry:
    def set_geometry(self):
        """Perturb all internal boundary nodes randomly."""
        super().set_geometry()
        sd = self.mdg.subdomains()[0]
        h = self.meshing_arguments()["cell_size"]
        inds = sd.get_internal_nodes()
        sd.nodes[:2, inds] += (np.random.rand(len(inds)) - 0.5) * h / 4
        sd.compute_geometry()


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


class InternalPerturbedGeometry:
    def set_geometry(self):
        """Perturb an internal node.

        Perturbes an arbitrary internal node. The node might be an internal node of a
        boundary cell. For consistently perturbing nodes belonging to a boundary cell,
        see the partially_perturbed_boundary.py file.

        """
        super().set_geometry()
        sd = self.mdg.subdomains()[0]
        h = self.meshing_arguments()["cell_size"]

        inds = sd.get_internal_nodes()
        random_index = np.random.choice(inds)

        sd.nodes[:2, random_index] += (np.random.rand(1) - 0.5) * h / 4
        sd.compute_geometry()

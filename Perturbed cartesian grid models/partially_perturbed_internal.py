import porepy as pp

import numpy as np

from drum_simulation_base import BaseClass


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


class InternalPerturbedModel(
    InternalPerturbedGeometry,
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
    "folder_name": "perturbed_nodes",
    "manufactured_solution": "drum_solution",
    "progressbars": True,
}

model = InternalPerturbedModel(params)
pp.run_time_dependent_model(model, params)

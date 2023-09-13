import porepy as pp

import numpy as np

from drum_simulation_base import BaseClass


class InternalPerturbedGeometry:
    def set_geometry(self):
        super().set_geometry()
        sd = self.mdg.subdomains()[0]
        h = self.meshing_arguments()["cell_size"]

        # Fetching a random internal node. The internal node might belong to a boundary
        # cell, which is really a test performed in partially_perturbed_boundary.py. It
        # is however unlikely that such a node is chosen over an actual internal node.
        inds = sd.get_internal_nodes()
        random_index = np.random.choice(inds)

        sd.nodes[:2, random_index] += (np.random.rand(1) - 0.5) * h / 4
        sd.compute_geometry()

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.0125 / self.units.m}
        return mesh_args


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
    "manufactured_solution": "bubble",
    "progressbars": True,
}

model = InternalPerturbedModel(params)
pp.run_time_dependent_model(model, params)

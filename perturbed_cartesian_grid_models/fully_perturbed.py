import porepy as pp

import numpy as np

from drum_simulation_base import BaseClass


class AllPerturbedGeometry:
    def set_geometry(self):
        """Perturb all internal boundary nodes randomly."""
        super().set_geometry()
        sd = self.mdg.subdomains()[0]
        h = self.meshing_arguments()["cell_size"]
        inds = sd.get_internal_nodes()
        sd.nodes[:2, inds] += (np.random.rand(len(inds)) - 0.5) * h / 4
        sd.compute_geometry()


class AllPerturbedModel(
    AllPerturbedGeometry,
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

model = AllPerturbedModel(params)
pp.run_time_dependent_model(model, params)

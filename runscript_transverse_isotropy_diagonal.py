import porepy as pp
import numpy as np

from models import MomentumBalanceABC

from utils import InnerDomainVTIStiffnessTensorMixin


class ModifiedGeometry:
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 20.0 / self.units.m
        y = 20.0 / self.units.m
        z = 20.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 1.0 / self.units.m}
        return mesh_args


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
        vals = np.zeros((self.nd, sd.num_cells))

        return vals.ravel("F")


class EntireAnisotropy3DModel(
    ModifiedGeometry,
    SolutionStratABC,
    InnerDomainVTIStiffnessTensorMixin,
    MomentumBalanceABC,
):
    @property
    def rotation_angle(self):
        return np.pi / 8


t_shift = 0.0
tf = 15.0
time_steps = 150.0
dt = tf / time_steps


time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

anisotropy_constants = {
    "mu_parallel": 4,
    "mu_orthogonal": 500,
    "lambda_parallel": 2,
    "lambda_orthogonal": 100,
    "volumetric_compr_lambda": 50,
}


params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "folder_name": "testing_diagonal_vti_",
    "manufactured_solution": "diag_wave",
    "inner_domain_width": 10,
    "progressbars": True,
    "anisotropy_constants": anisotropy_constants,
}

model = EntireAnisotropy3DModel(params)

pp.run_time_dependent_model(model=model, params=params)

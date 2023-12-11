import porepy as pp
import numpy as np

from models import MomentumBalanceABC

from utils import TransverselyAnisotropicStiffnessTensor


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
        mesh_args: dict[str, float] = {"cell_size": 0.5 / self.units.m}
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

        # Assigning a one-cell source term in the middle of the domain. Only for testing
        # purposes.
        x_point = self.domain.bounding_box["xmax"] / 2.0
        y_point = self.domain.bounding_box["ymax"] / 2.0
        z_point = self.domain.bounding_box["zmax"] / 2.0

        closest_cell = sd.closest_cell(np.array([[x_point], [y_point], [z_point]]))[0]

        vals[0][closest_cell] = 1
        vals[1][closest_cell] = 1
        vals[2][closest_cell] = 1

        if self.time_manager.time_index <= 10:
            return vals.ravel("F") * self.time_manager.time
        else:
            return vals.ravel("F") * 0


class EntireAnisotropy3DModel(
    ModifiedGeometry,
    TransverselyAnisotropicStiffnessTensor,
    SolutionStratABC,
    MomentumBalanceABC,
):
    ...


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

solid_constants = pp.SolidConstants(
    {
        "lame_lambda": 1.0,
        "shear_modulus": 1.0,
    }
)

material_constants = {"solid": solid_constants}
anisotropy_constants = {
    "mu_parallel": 5,
    "mu_orthogonal": 5,
    "lambda_parallel": 2,
    "lambda_orthogonal": 2,
    "volumetric_compr_lambda": 4,
}

params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "folder_name": "visualization_cs",
    "manufactured_solution": "simply_zero",
    "inner_domain_width": 10,
    "progressbars": True,
    "anisotropy_constants": anisotropy_constants,
}

model = EntireAnisotropy3DModel(params)

pp.run_time_dependent_model(model=model, params=params)

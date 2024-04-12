import porepy as pp

import numpy as np

from models.elastic_wave_equation_abc import DynamicMomentumBalanceABC2
from utils import TransverselyAnisotropicStiffnessTensor


class MyGeometry:
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        z = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.02 / self.units.m}
        return mesh_args


class MomentumBalanceABCModifiedGeometry(
    MyGeometry,
    TransverselyAnisotropicStiffnessTensor,
    DynamicMomentumBalanceABC2,
):

    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial velocity values."""
        sd = self.mdg.subdomains()[0]
        t = self.time_manager.time

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        z = sd.cell_centers[2, :]

        vals = np.zeros((self.nd, sd.num_cells))

        theta = 1
        lam = 0.125

        common_part = theta * np.exp(
            -np.pi**2 * ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.7) ** 2) / lam**2
        )

        vals[0] = common_part * (x - 0.5)

        vals[1] = common_part * (y - 0.5)

        vals[2] = common_part * (z - 0.7)

        return vals.ravel("F")

    def data_to_export(self):
        """Define the data to export to vtu.

        Returns:
            list: List of tuples containing the subdomain, variable name,
            and values to export.

        """
        data = super().data_to_export()
        for sd in self.mdg.subdomains(dim=self.nd):
            vel_op = self.velocity_time_dep_array([sd]) * self.velocity_time_dep_array(
                [sd]
            )
            vel_op_int = self.volume_integral(integrand=vel_op, grids=[sd], dim=3)
            vel_op_int_val = vel_op_int.value(self.equation_system)

            vel = self.velocity_time_dep_array([sd]).value(self.equation_system)

            data.append((sd, "energy", vel_op_int_val))
            data.append((sd, "velocity", vel))
        return data


tf = 0.15
dt = tf / 90.0


time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
)

anisotropy_constants = {
    "mu_parallel": 1,
    "mu_orthogonal": 1,
    "lambda_parallel": 5,
    "lambda_orthogonal": 5,
    "volumetric_compr_lambda": 10,
}

params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "folder_name": "source_in_outer",
    "manufactured_solution": "simply_zero",
    "anisotropy_constants": anisotropy_constants,
    "progressbars": True,
    "inner_domain_width": 0.5,
    "inner_domain_center": (0.5, 0.5, 0.3),
}

model = MomentumBalanceABCModifiedGeometry(params)
pp.run_time_dependent_model(model, params)

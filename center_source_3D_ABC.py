import porepy as pp

import numpy as np

from models import MomentumBalanceABC


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

        if self.time_manager.time_index <= 20:
            return vals.ravel("F") * self.time_manager.time
        else:
            return vals.ravel("F") * 0


class MyGeometry:
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 8.05 / self.units.m
        y = 8.05 / self.units.m
        z = 8.05 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.35 / self.units.m}
        return mesh_args


class MomentumBalanceABCModifiedGeometry(
    MyGeometry,
    SolutionStratABC,
    MomentumBalanceABC,
):
    ...


t_shift = 0.0
tf = 2.5
dt = tf / 250.0


time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "density": 1.0,
        "lame_lambda": 1.0e1,
        "shear_modulus": 1.0e1,
    }
)

material_constants = {"solid": solid_constants}

params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "material_constants": material_constants,
    "folder_name": "center_source",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
}

model = MomentumBalanceABCModifiedGeometry(params)
pp.run_time_dependent_model(model, params)

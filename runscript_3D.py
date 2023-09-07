import porepy as pp

import numpy as np

from models import MomentumBalanceTimeDepSource


class BCVals:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north
            + bounds.south
            + bounds.west
            + bounds.east
            + bounds.top
            + bounds.bottom,
            "dir",
        )
        return bc

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.AdArray:
        values = []
        for sd in subdomains:
            bounds = self.domain_boundary_sides(sd)
            val_loc = np.zeros((self.nd, sd.num_faces))
            values.append(val_loc)

        values = np.array(values)
        values = values.ravel("F")
        return pp.wrap_as_ad_array(values, name="bc_vals_mechanics")


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
        mesh_args: dict[str, float] = {"cell_size": 0.125 / 2 / self.units.m}
        return mesh_args


class MomentumBalanceABCModifiedGeometry(
    BCVals,
    MyGeometry,
    MomentumBalanceTimeDepSource,
):
    ...


t_shift = 0.0
time_steps = 20.0
tf = 10.0
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
        "density": 1.0,
        "lame_lambda": 1.0e0,
        "shear_modulus": 1.0e0,
    }
)

material_constants = {"solid": solid_constants}

params = {
    "time_manager": time_manager,
    "grid_type": "simplex",
    "material_constants": material_constants,
    "folder_name": "center_source",
    "manufactured_solution": "bubble",
    "progressbars": True,
}

model = MomentumBalanceABCModifiedGeometry(params)
pp.run_time_dependent_model(model, params)
sd = model.mdg.subdomains(dim=3)[0]
data = model.mdg.subdomain_data(sd)

from utils import get_solution_values

t = tf
x = sd.cell_centers[0, :]
y = sd.cell_centers[1, :]
z = sd.cell_centers[2, :]
ones = np.ones(len(x))

u1 = u2 = u3 = t**2 * x * y * z * (1 - x) * (1 - y) * (1 - z)
u_e = np.array([u1, u2, u3])
u_e = u_e.ravel("F")
u = get_solution_values(name="u", data=data, time_step_index=0)

error = pp.error_computation.l2_error(
    grid=sd, true_array=u_e, approx_array=u, is_scalar=False, is_cc=True, relative=True
)

print(error)

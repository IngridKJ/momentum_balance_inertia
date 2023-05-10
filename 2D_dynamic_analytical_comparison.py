import porepy as pp
import numpy as np

from models import DynamicMomentumBalance

from utils import body_force_func_time


class NewmarkConstants:
    @property
    def gamma(self) -> float:
        return 0.5


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1 / self.units.m
        y = 1 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.1 / self.units.m}
        return mesh_args


class MomentumBalanceBC:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "dir",
        )
        return bc

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.AdArray:
        values = []
        for sd in subdomains:
            bounds = self.domain_boundary_sides(sd)
            val_loc = np.zeros((self.nd, sd.num_faces))
            # See section on scaling for explanation of the conversion.
            value = 1
            val_loc[1, bounds.north] = -value * 1e-11 * 0

            values.append(val_loc)

        values = np.array(values)
        values = values.ravel("F")
        return pp.wrap_as_ad_array(values, name="bc_vals_mechanics")

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force integrated over the subdomain cells.

        Parameters:
            subdomains: List of subdomains where the body force is defined.

        Returns:
            Operator for the body force.

        """

        source_term = body_force_func_time(model)

        num_cells = sum([sd.num_cells for sd in subdomains])
        vals = np.zeros((self.nd, num_cells))
        dt = self.time_manager.dt_init
        for sd in subdomains:
            x = sd.cell_centers[0, :]
            y = sd.cell_centers[1, :]

            x_val = source_term[0](x, y, dt)
            y_val = source_term[1](x, y, dt)

            cell_volume = sd.cell_volumes

        vals[0] = -x_val * cell_volume
        vals[1] = -y_val * cell_volume
        source = pp.ad.DenseArray(vals.ravel("F"), "body_force")
        return source


class MyMomentumBalance(
    NewmarkConstants,
    MomentumBalanceBC,
    MyGeometry,
    DynamicMomentumBalance,
):
    ...


time_manager = pp.TimeManager(
    schedule=[0, 1e-1],
    dt_init=1e-3,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "density": 1,
        "lame_lambda": 1,
        "permeability": 1,
        "porosity": 1,
        "shear_modulus": 1,
    }
)

material_constants = {"solid": solid_constants}
params = {
    "time_manager": time_manager,
    "material_constants": material_constants,
    "folder_name": "visualization_2D_dynamic",
}
model = MyMomentumBalance(params)

pp.run_time_dependent_model(model, params)

import porepy as pp
import numpy as np

from models import DynamicMomentumBalance


class NewmarkConstants:
    @property
    def gamma(self) -> float:
        return 0.4


class MyGeometry:
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})
        box.update({"zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 10 / self.units.m
        y = 10 / self.units.m
        z = 10 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 1 / self.units.m}
        return mesh_args


class MomentumBalanceBC:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.south + bounds.north, "dir")
        return bc

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.AdArray:
        values = []
        for sd in subdomains:
            bounds = self.domain_boundary_sides(sd)
            val_loc = np.zeros((self.nd, sd.num_faces))
            # See section on scaling for explanation of the conversion.
            value = 1
            val_loc[1, bounds.north] = -value * 1e-11
            val_loc[1, bounds.south] = 0

            values.append(val_loc)

        values = np.array(values)
        values = values.ravel("F")
        return pp.wrap_as_ad_array(values, name="bc_vals_mechanics")


class MyInitialValues:
    def initial_acceleration(self, dofs: int) -> np.ndarray:
        """Initial acceleration values."""
        return np.ones(dofs * self.nd) * 0.0000001


class MyMomentumBalance(
    NewmarkConstants,
    MomentumBalanceBC,
    MyGeometry,
    MyInitialValues,
    DynamicMomentumBalance,
):
    ...


time_manager = pp.TimeManager(
    schedule=[0, 0.05],
    dt_init=0.0005,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "density": 2700,
        "lame_lambda": 1.067 * 1e10,
        "permeability": 1e-15,
        "porosity": 1e-2,
        "shear_modulus": 1.7 * 1e10,
    }
)

material_constants = {"solid": solid_constants}
params = {
    "time_manager": time_manager,
    "material_constants": material_constants,
    "folder_name": "visualization_3D_dynamic",
}
model = MyMomentumBalance(params)
pp.run_time_dependent_model(model, params)

pp.plot_grid(
    grid=model.mdg, vector_value="u", figsize=(10, 8), title="Displacement", alpha=0.5
)

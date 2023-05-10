import porepy as pp
import numpy as np

from porepy.models.momentum_balance import MomentumBalance

from utils import body_force_func


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


class MomentumBalanceBCAndSource:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd, bounds.north + bounds.south + bounds.west + bounds.east, "dir"
        )
        return bc

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force integrated over the subdomain cells.

        Parameters:
            subdomains: List of subdomains where the body force is defined.

        Returns:
            Operator for the body force.

        """

        source_term = body_force_func(model)

        num_cells = sum([sd.num_cells for sd in subdomains])
        vals = np.zeros((self.nd, num_cells))
        for sd in subdomains:
            x = sd.cell_centers[0, :]
            y = sd.cell_centers[1, :]
            x_val = source_term[0](x, y)
            y_val = source_term[1](x, y)
            cell_volume = sd.cell_volumes

        vals[0] = -x_val * cell_volume
        vals[1] = -y_val * cell_volume
        source = pp.ad.DenseArray(vals.ravel("F"), "body_force")
        return source


class MyMomentumBalance(
    MyGeometry,
    MomentumBalanceBCAndSource,
    MomentumBalance,
):
    ...


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
    "material_constants": material_constants,
    "folder_name": "2D_static_analytical",
}

model = MyMomentumBalance(params)
pp.run_time_dependent_model(model, params)

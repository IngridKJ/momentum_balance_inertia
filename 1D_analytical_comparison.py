"""A try on numerical vs. manufactured solution in quasi 1D (1 x N x 1 cells in x-, y-
and z-direction, respectively). 

It doesn't seem to work as the numerical and manufactured solutions differ with a
factor of 1/5.

Hypothesis: This quasi 1D thing does not work as intended.

"""

import porepy as pp
import numpy as np

from porepy.models.momentum_balance import MomentumBalance


class MyGeometry:
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})
        box.update({"zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 0.05 / self.units.m
        y = 1 / self.units.m
        z = 0.05 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.05 / self.units.m}
        return mesh_args


class MomentumBalanceBCAndSource:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.north + bounds.south, "dir")
        return bc

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force integrated over the subdomain cells.

        Parameters:
            subdomains: List of subdomains where the body force is defined.

        Returns:
            Operator for the body force.

        """

        num_cells = sum([sd.num_cells for sd in subdomains])
        vals = np.zeros((self.nd, num_cells))
        for sd in subdomains:
            y_coords = sd.cell_centers[1, :]
            # Note that these source terms are not time-dependent. The indended use was
            # to see one time-step and compare to the analytical solution.
            if self.params["analytical_solution"] == "pol":
                # Corresponds to analytical solution u(y) = y*(1 - y)
                y_values = (
                    -4 * self.solid.shear_modulus() - 2 * self.solid.lame_lambda()
                )
            elif self.params["analytical_solution"] == "sin":
                # Corresponds to analytical solution u(y) = np.sin(np.pi * y)
                y_values = -2 * np.pi**2 * self.solid.shear_modulus() * np.sin(
                    np.pi * y_coords
                ) - np.pi**2 * self.solid.lame_lambda() * np.sin(np.pi * y_coords)

            cell_volume = sd.cell_volumes

        vals[1] = -y_values * cell_volume
        source = pp.ad.DenseArray(vals.ravel("F"), "body_force")
        return source


class MyMomentumBalance(
    MyGeometry,
    MomentumBalanceBCAndSource,
    MomentumBalance,
):
    ...


analytical_sol = "sin"

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
    "folder_name": "1D_static_analytical" + "_" + analytical_sol,
    "analytical_solution": analytical_sol,
}

model = MyMomentumBalance(params)
pp.run_time_dependent_model(model, params)

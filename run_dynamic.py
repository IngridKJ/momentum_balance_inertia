import porepy as pp
import numpy as np

import models


class MomentumBalanceBC:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.north + bounds.south, "dir")
        return bc

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.AdArray:
        values = []
        for sd in subdomains:
            bounds = self.domain_boundary_sides(sd)
            val_loc = np.zeros((self.nd, sd.num_faces))
            # See section on scaling for explanation of the conversion.
            value = 1
            # val_loc[0, :] = value
            val_loc[0, bounds.north] = -value

            values.append(val_loc)

        values = np.array(values)
        values = values.ravel("F")
        return pp.wrap_as_ad_array(values, name="bc_vals_mechanics")


class Run(
    MomentumBalanceBC,
    models.DynamicMomentumBalance,
):
    ...


params = {}
model = Run(params)


pp.run_time_dependent_model(model, params)

pp.plot_grid(
    grid=model.mdg, vector_value="u", figsize=(10, 8), title="Displacement", alpha=0.5
)

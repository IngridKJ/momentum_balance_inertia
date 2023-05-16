import porepy as pp

import sys

sys.path.append("../")

from models.no_inertia_momentum_balance import MomentumBalanceModified

params = {}
model = MomentumBalanceModified(params)
pp.run_time_dependent_model(model, params)

pp.plot_grid(
    grid=model.mdg, vector_value="u", figsize=(10, 8), title="Displacement", alpha=0.5
)

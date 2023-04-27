import porepy as pp
from models.no_inertia_momentum_balance import MyMomentumBalance

import utils as ut
import time_derivatives as td
import equations as eq

params = {}
model = MyMomentumBalance(params)
pp.run_time_dependent_model(model, params)

pp.plot_grid(model.mdg)

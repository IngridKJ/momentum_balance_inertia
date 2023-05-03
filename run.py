import porepy as pp
from models.no_inertia_momentum_balance import MyMomentumBalance

params = {}
model = MyMomentumBalance(params)
pp.run_time_dependent_model(model, params)

pp.plot_grid(model.mdg)

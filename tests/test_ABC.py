import sys

import numpy as np
import porepy as pp
from ABC_model_for_testing import MomentumBalanceABCForTesting

sys.path.append("../")
import run_models.run_linear_model as rlm


def test_energy_decay():
    tf = 1.0
    dt = tf / 100

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
    )

    params = {
        "time_manager": time_manager,
        "grid_type": "cartesian",
        "manufactured_solution": "simply_zero",
    }

    model = MomentumBalanceABCForTesting(params)
    rlm.run_linear_model(model, params)

    for sd in model.mdg.subdomains(dim=model.nd):
        vel_op = model.velocity_time_dep_array([sd]) * model.velocity_time_dep_array(
            [sd]
        )
        vel_op_int = model.volume_integral(integrand=vel_op, grids=[sd], dim=2)
        vel_op_int_val = vel_op_int.value(model.equation_system)
        assert np.isclose(np.linalg.norm(vel_op_int_val), 3.859854428354451e-08)

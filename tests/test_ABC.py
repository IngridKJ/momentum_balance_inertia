import porepy as pp
import numpy as np

from ABC_model_for_testing import MomentumBalanceABCForTesting


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
    pp.run_time_dependent_model(model, params)

    for sd in model.mdg.subdomains(dim=model.nd):
        vel_op = model.velocity_time_dep_array([sd]) * model.velocity_time_dep_array(
            [sd]
        )
        vel_op_int = model.volume_integral(integrand=vel_op, grids=[sd], dim=2)
        vel_op_int_val = vel_op_int.value(model.equation_system)

        # The comparison value here is obtained by running the non-refactored ABC2. To
        # be compared with the refactored ABC2 when I get to it.
        assert np.isclose(np.linalg.norm(vel_op_int_val), 3.859854428354451e-08)

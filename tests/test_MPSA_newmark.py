import sys
from copy import deepcopy

import pytest

sys.path.append("../")
import numpy as np
import porepy as pp
from convergence_and_stability_analysis.analysis_models.manufactured_solution_dynamic_2D import (
    ManuMechSetup2d,
)
from convergence_and_stability_analysis.analysis_models.manufactured_solution_dynamic_3D import (
    ManuMechSetup3d,
)
from porepy.applications.convergence_analysis import ConvergenceAnalysis


@pytest.mark.parametrize(
    "ManuSetup, expected_order_of_convergence, error_displacement, error_force",
    [
        (
            ManuMechSetup2d,
            {"ooc_displacement": 2.173359730130687, "ooc_force": 2.2161318563634813},
            (0.3237071890356826, 0.07176387389579363),
            (0.3621648470265387, 0.07794425866447736),
        ),
        (
            ManuMechSetup3d,
            {"ooc_displacement": 2.150222354959266, "ooc_force": 1.9984124039928801},
            (0.32852333912390724, 0.07400904533902304),
            (0.33113535330348437, 0.08287498692468019),
        ),
    ],
)
def test_convergence_MPSA_Newmark_in_2D_and_3D(
    ManuSetup, expected_order_of_convergence, error_displacement, error_force
):
    """Tests the MPSA-Newmark in 2D and 3D.

    The expected convergence rates and errors are calculated after a convergence run has
    been executed. These values are saved and hardcoded into this test for later
    comparison.

    Parameters:
        ManuSetup: Manufactured solution setup.
        expected_order_of_convergence: The expected order of convergence for force and
            displacement errors.
        error_displacement: Expected errors for the displacement for two refinement
            levels.
        error_force: Expected errors for the force for two refinement levels.

    """
    # 20 time steps might seem unnecessary, but a certain ratio is needed between the
    # cell size and the time step size. These values are thus not (necessarily)
    # trivially decreased.
    dt = 1.0 / 20.0
    tf = 1.0

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
    )

    params = {
        "time_manager": time_manager,
        "manufactured_solution": "sin_bubble",
        "grid_type": "cartesian",
        "meshing_arguments": {"cell_size": 0.25 / 1.0},
        "plot_results": False,
    }

    conv_analysis = ConvergenceAnalysis(
        model_class=ManuSetup,
        model_params=deepcopy(params),
        levels=2,
        spatial_refinement_rate=2,
        temporal_refinement_rate=2,
    )

    results = conv_analysis.run_analysis()

    # Checking convergence order
    actual_order_of_convergence = conv_analysis.order_of_convergence(results)
    for key, value in expected_order_of_convergence.items():
        assert np.isclose(value, actual_order_of_convergence[key])

    # Checking error values
    for ind, result in enumerate(results):
        assert np.isclose(
            error_displacement[ind], getattr(result, "error_displacement")
        )
        assert np.isclose(error_force[ind], getattr(result, "error_force"))

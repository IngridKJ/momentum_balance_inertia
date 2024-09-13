import os

N_THREADS = "1"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

from copy import deepcopy

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis

from manufactured_solution_dynamic_3D import ManuMechSetup3d
from utils_convergence_analysis import export_errors_to_txt, run_analysis

time_steps = 4
tf = 1.0
dt = tf / time_steps

time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)


params = {
    "time_manager": time_manager,
    "manufactured_solution": "sin_bubble",
    "grid_type": "simplex",
    "meshing_arguments": {"cell_size": 0.03125},
    "plot_results": False,
}

conv_analysis = ConvergenceAnalysis(
    model_class=ManuMechSetup3d,
    model_params=deepcopy(params),
    levels=4,
    spatial_refinement_rate=1,
    temporal_refinement_rate=2,
)
ooc: list[list[dict[str, float]]] = []
ooc_setup: list[dict[str, float]] = []

results = run_analysis(conv_analysis)
ooc_setup.append(
    conv_analysis.order_of_convergence(
        results,
        x_axis="time_step",
    )
)
ooc.append(ooc_setup)
print(ooc_setup)
export_errors_to_txt(
    self=conv_analysis,
    list_of_results=results,
    file_name="displacement_and_traction_errors_time.txt",
)

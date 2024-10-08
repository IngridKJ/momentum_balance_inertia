from copy import deepcopy

import porepy as pp
from manufactured_solution_dynamic_2D import ManuMechSetup2d
from porepy.applications.convergence_analysis import ConvergenceAnalysis

t_shift = 0.0
dt = 1.0 / 100.0
tf = 1.0

time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)


params = {
    "time_manager": time_manager,
    "folder_name": "analysis_viz",
    # "manufactured_solution": "bubble",
    "manufactured_solution": "sin_bubble",
    "grid_type": "cartesian",
    "meshing_arguments": {"cell_size": 0.25 / 1.0},
    "plot_results": False,
}

conv_analysis = ConvergenceAnalysis(
    model_class=ManuMechSetup2d,
    model_params=deepcopy(params),
    levels=2,
    spatial_refinement_rate=2,
    temporal_refinement_rate=2,
)
ooc: list[list[dict[str, float]]] = []
ooc_setup: list[dict[str, float]] = []

results = conv_analysis.run_analysis()
# Uncomment one of these for single refinement (either space or time)
# ooc_setup.append(
#     conv_analysis.order_of_convergence(
#         results,
#         x_axis="cell_diameter",
#     )
# )
# ooc.append(ooc_setup)
# print(ooc_setup)
# conv_analysis.export_errors_to_txt(list_of_results=results)

# ooc_setup.append(conv_analysis.order_of_convergence(results, x_axis="time_step"))
# ooc.append(ooc_setup)
# print(ooc_setup)
# conv_analysis.export_errors_to_txt(list_of_results=results)

# Uncomment this for double refinement:
ooc_setup.append(
    conv_analysis.order_of_convergence(
        results,
    )
)
ooc.append(ooc_setup)
print(ooc_setup)
conv_analysis.export_errors_to_txt(list_of_results=results)

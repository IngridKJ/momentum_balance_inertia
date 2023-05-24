import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from manufactured_solution_dynamic import ManuMechSetup2d

import numpy as np

from copy import deepcopy


class ConvTest(ManuMechSetup2d):
    ...


time_manager = pp.TimeManager(
    schedule=[0, 1e0],
    dt_init=1e-1,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)


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
    "time_manager": time_manager,
    "folder_name": "analysis_viz",
    "manufactured_solution": "bubble",
    # "manufactured_solution": "quad_time",
    # "manufactured_solution": "cub_cub",
    # "manufactured_solution": "cub_quad",
    # "manufactured_solution": "quad_space",
    "grid_type": "simplex",
    "material_constants": material_constants,
    "meshing_arguments": {"cell_size": 0.25},
    "plot_results": False,
}

model = ConvTest(params)

pp.run_time_dependent_model(model, params)

conv_analysis = ConvergenceAnalysis(
    model_class=ManuMechSetup2d,
    model_params=deepcopy(params),
    levels=5,
    spatial_refinement_rate=1,
    temporal_refinement_rate=2,
)
ooc: list[list[dict[str, float]]] = []
ooc_setup: list[dict[str, float]] = []

results = conv_analysis.run_analysis()
# ooc_setup.append(conv_analysis.order_of_convergence(results, x_axis="cell_diameter"))
ooc_setup.append(conv_analysis.order_of_convergence(results, x_axis="time_step"))

ooc.append(ooc_setup)
print(ooc_setup)
conv_analysis.export_errors_to_txt(list_of_results=results)

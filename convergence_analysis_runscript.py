import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from manu_dynamic_mech_nofrac import ManuMechSetup2d


from copy import deepcopy


class ConvTest(ManuMechSetup2d):
    ...


time_manager = pp.TimeManager(
    schedule=[0, 8 * 0.5e-1],
    dt_init=0.5e-2,
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
    "grid_type": "cartesian",
    "material_constants": material_constants,
    "meshing_arguments": {"cell_size": 0.25},
    "plot_results": False,
}

model = ConvTest(params)

pp.run_time_dependent_model(model, params)

conv_analysis = ConvergenceAnalysis(
    model_class=ManuMechSetup2d,
    model_params=deepcopy(params),
    levels=3,
    spatial_refinement_rate=2,
    temporal_refinement_rate=2,
)
ooc: list[list[dict[str, float]]] = []
ooc_setup: list[dict[str, float]] = []

results = conv_analysis.run_analysis()
ooc_setup.append(conv_analysis.order_of_convergence(results))
ooc.append(ooc_setup)
print(ooc)

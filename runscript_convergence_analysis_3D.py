import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from manufactured_solution_dynamic_3D import ManuMechSetup3d

from copy import deepcopy


class ConvTest(ManuMechSetup3d):
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        z = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")


t_shift = 0.0
time_steps = 2
tf = 1.0
dt = tf / time_steps

time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)


solid_constants = pp.SolidConstants(
    {
        "density": 1,
        "lame_lambda": 1,
        "permeability": 1,
        "shear_modulus": 1,
    }
)

material_constants = {"solid": solid_constants}
params = {
    "time_manager": time_manager,
    "folder_name": "analysis_viz",
    # "manufactured_solution": "bubble",
    "manufactured_solution": "sin_bubble",
    "grid_type": "simplex",
    "material_constants": material_constants,
    "meshing_arguments": {"cell_size": 0.125 / 2},
    "plot_results": False,
    "progress_bars": True,
}

model = ConvTest(params)

conv_analysis = ConvergenceAnalysis(
    model_class=ManuMechSetup3d,
    model_params=deepcopy(params),
    levels=5,
    spatial_refinement_rate=1,
    temporal_refinement_rate=2,
)
ooc: list[list[dict[str, float]]] = []
ooc_setup: list[dict[str, float]] = []

results = conv_analysis.run_analysis()
# ooc_setup.append(
#     conv_analysis.order_of_convergence(
#         results, x_axis="cell_diameter", data_range=slice(-2, None, None)
#     )
# )
ooc_setup.append(
    conv_analysis.order_of_convergence(
        results, x_axis="time_step", data_range=slice(-2, None, None)
    )
)

ooc.append(ooc_setup)
print(ooc_setup)
conv_analysis.export_errors_to_txt(list_of_results=results)

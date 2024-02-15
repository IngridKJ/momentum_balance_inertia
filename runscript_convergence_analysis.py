import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from manufactured_solution_dynamic import ManuMechSetup2d

from copy import deepcopy


class ConvTest(ManuMechSetup2d):
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")


t_shift = 0.0
dt = 1.0 / 600.0
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
    "grid_type": "simplex",
    "meshing_arguments": {"cell_size": 0.25 / 1.0},
    "plot_results": False,
}

model = ConvTest(params)

conv_analysis = ConvergenceAnalysis(
    model_class=ManuMechSetup2d,
    model_params=deepcopy(params),
    levels=5,
    spatial_refinement_rate=2,
    temporal_refinement_rate=1,
)
ooc: list[list[dict[str, float]]] = []
ooc_setup: list[dict[str, float]] = []

results = conv_analysis.run_analysis()
ooc_setup.append(
    conv_analysis.order_of_convergence(
        results,
        x_axis="cell_diameter",
    )
)
# ooc_setup.append(conv_analysis.order_of_convergence(results, x_axis="time_step"))

ooc.append(ooc_setup)
print(ooc_setup)
conv_analysis.export_errors_to_txt(list_of_results=results)

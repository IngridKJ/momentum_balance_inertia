import os

N_THREADS = "1"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp

sys.path.append("../")
from plotting.plot_utils import draw_multiple_loglog_slopes, fetch_numbers_from_file
from porepy.applications.convergence_analysis import ConvergenceAnalysis

from convergence_analysis_models.manufactured_solution_dynamic_3D import ManuMechSetup3d
from utils_convergence_analysis import export_errors_to_txt, run_analysis

# Prepare path for generated output files
folder_name = "convergence_analysis_results"
filename = "displacement_and_traction_errors_space_time.txt"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, filename)

# Coarse/Fine variables and plotting (save figure)
coarse = True
save_figure = True

# Simulation details from here and onwards
time_steps = 150
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
    "meshing_arguments": {"cell_size": 0.25 / 1.0},
    "plot_results": False,
}

conv_analysis = ConvergenceAnalysis(
    model_class=ManuMechSetup3d,
    model_params=deepcopy(params),
    levels=2 if coarse else 4,
    spatial_refinement_rate=2,
    temporal_refinement_rate=2,
)
ooc: list[list[dict[str, float]]] = []
ooc_setup: list[dict[str, float]] = []

results = run_analysis(conv_analysis)
ooc_setup.append(
    conv_analysis.order_of_convergence(
        results,
    )
)
ooc.append(ooc_setup)
print(ooc_setup)
export_errors_to_txt(
    self=conv_analysis,
    list_of_results=results,
    file_name=filename,
)

# Plotting from here and downwards.
if save_figure:
    values = fetch_numbers_from_file(filename)
    num_cells = np.array(values["num_cells"]) ** (1 / 3)
    y_disp = values["error_displacement"]
    y_trac = values["error_force"]

    # Plot the sample data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(
        num_cells,
        y_disp,
        "o--",
        color="firebrick",
        label="Displacement",
    )
    ax.loglog(
        num_cells,
        y_trac,
        "o--",
        color="royalblue",
        label="Traction",
    )
    ax.set_title("Convergence analysis: Spatial and temporal")
    ax.set_xlabel("$(Number\ of\ cells)^{1/3}$")
    ax.set_ylabel("Relative $L^2$ error")
    ax.legend()

    if not coarse:
        # Draw the convergence triangle with multiple slopes
        draw_multiple_loglog_slopes(
            fig,
            ax,
            origin=(0.915 * num_cells[-1], 1.05 * y_disp[-1]),
            triangle_width=1.0,
            slopes=[-2],
            inverted=True,
            labelcolor=(0.33, 0.33, 0.33),
        )

        draw_multiple_loglog_slopes(
            fig,
            ax,
            origin=(1.1 * num_cells[-2], y_trac[-2]),
            triangle_width=1.0,
            slopes=[-1.5],
            inverted=False,
            labelcolor=(0.33, 0.33, 0.33),
        )

    ax.grid(True, which="both", color=(0.87, 0.87, 0.87))

    folder_name = "figures"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "space_time_convergence_dirichlet_boundaries.png")
    )

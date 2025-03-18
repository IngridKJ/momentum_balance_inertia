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

from analysis_models.manufactured_solution_dynamic_3D import ManuMechSetup3d
from utils_convergence_analysis import export_errors_to_txt, run_analysis

# Prepare path for generated output files
folder_name = "convergence_analysis_results"
filename = "displacement_and_traction_errors_space_time.txt"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, filename)

# Coarse/Fine variables, grid type (either "simplex" or "cartesian") and plotting (save
# figure)
coarse = True
grid_type = "simplex"
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
    "manufactured_solution": "different_x_y_z_components",
    "grid_type": grid_type,
    "meshing_arguments": {"cell_size": 0.25 / 1.0},
    "plot_results": False,
    "petsc_solver_q": True,
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
    if coarse:
        time_step_numbers = np.array([150, 300])
    else:
        time_step_numbers = np.array([150, 300, 600, 1200])
    num_cells = (np.array(values["num_cells"]) * time_step_numbers) ** (1 / 4)
    y_disp = values["error_displacement"]
    y_trac = values["error_force"]

    # Plot the sample data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(
        num_cells,
        y_disp,
        "o--",
        color="#55A1FF",
        label=r"$\mathcal{E}_u$",
        linewidth=2.5,
        markersize=8,
    )
    ax.loglog(
        num_cells,
        y_trac,
        "o--",
        color="#003FBB",
        label=r"$\mathcal{E}_T$",
        linewidth=2.5,
        markersize=8,
    )
    ax.set_title("Convergence analysis: Setup with Dirichlet boundaries", fontsize=18)
    ax.set_xlabel("$(N_x \\cdot N_t)^{1/4}$", fontsize=16)
    ax.set_ylabel("Relative $L^2$ error", fontsize=16)
    ax.legend(handlelength=2.3, handleheight=2, fontsize=14, labelspacing=0.2)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.xaxis.set_tick_params(which="both", labelsize=14)
    ax.yaxis.set_tick_params(which="both", labelsize=14)

    if not coarse:
        # Draw the convergence triangle with multiple slopes
        draw_multiple_loglog_slopes(
            fig,
            ax,
            origin=(0.915 * num_cells[-1], 1.05 * y_disp[-1]),
            triangle_width=1.4,
            slopes=[-2],
            inverted=True,
            labelcolor=(0.33, 0.33, 0.33),
        )

        draw_multiple_loglog_slopes(
            fig,
            ax,
            origin=(1.1 * num_cells[-2], y_trac[-2]),
            triangle_width=1.4,
            slopes=[-1.5],
            inverted=False,
            labelcolor=(0.33, 0.33, 0.33),
        )

    folder_name = "figures"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "space_time_convergence_dirichlet_boundaries.png")
    )

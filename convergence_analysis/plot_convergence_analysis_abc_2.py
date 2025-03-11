import os
import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from plotting.plot_utils import draw_multiple_loglog_slopes

sys.path.append("../")

# Manually defined dictionary for legend labels
label_dict = {
    (0, 0, 0): r"$r_h = 1, r_{a} = 0$",
    (1, 0, 0): r"$r_h = 2^{-5}, r_{a} = 0$",
    (2, 0, 0): r"$r_h = 2^{-8}, r_{a} = 0$",
    (0, 1, 1): r"$r_h = 1, r_{a} = 10^{4}$",
    (1, 1, 1): r"$r_h = 2^{-5}, r_{a} = 10^{4}$",
    (2, 1, 1): r"$r_h = 2^{-8}, r_{a} = 10^{4}$",
}

# Prepare path for generated output files
folder_name = "convergence_analysis_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

# Find all heterogeneity error files
heterogeneity_files = [
    f
    for f in os.listdir(output_dir)
    if f.startswith("errors_heterogeneity") and f.endswith(".txt")
]

# Extract unique factors and anisotropy coefficients
file_info = []
factors = set()
anisotropy_pairs = set()

for filename in heterogeneity_files:
    parts = filename.replace("errors_heterogeneity_", "").replace(".txt", "").split("_")
    factor = float(parts[0])
    mu_lam_index = int(parts[-1])
    mu = lam = mu_lam_index
    file_info.append((factor, mu, lam, filename))
    factors.add(factor)
    anisotropy_pairs.add((mu, lam))

file_info.sort()
factors = sorted(factors)
anisotropy_pairs = sorted(anisotropy_pairs)

# Define customizable styles
custom_colors_displacement = [
    "#D8B5DE",
    "black",
    "#9FE6A2",
    "black",
    "#55A1FF",
    "black",
]
custom_colors_traction = [
    "#A45892",
    "darkgray",
    "#00630C",
    "darkgray",
    "#003FBB",
    "darkgray",
]
custom_linestyles = [
    "-",
    ":",
    "-",
    ":",
    "-",
    ":",
]
custom_markers = [
    "o",
    "s",
    "o",
    "D",
    "o",
    "d",
]

# Assign styles uniquely for each combination
style_map = {}
for i, combo in enumerate(product(factors, anisotropy_pairs)):
    style_map[combo] = {
        "color_displacement": custom_colors_displacement[
            i % len(custom_colors_displacement)
        ],
        "color_traction": custom_colors_traction[i % len(custom_colors_traction)],
        "linestyle": custom_linestyles[i % len(custom_linestyles)],
        "marker": custom_markers[i % len(custom_markers)],
    }

# Create figure
fig, ax = plt.subplots(figsize=(16, 9))

# Lists to store handles and labels for displacement (u) and traction (T)
handles_u = []
labels_u = []
handles_T = []
labels_T = []

# Loop through all data and plot in the same figure
for factor, mu, lam, filename in file_info:
    filepath = os.path.join(output_dir, filename)
    num_cells, num_time_steps, displacement_errors, traction_errors = np.loadtxt(
        filepath, delimiter=",", skiprows=1, unpack=True, dtype=float
    )

    x_axis = (num_cells * num_time_steps) ** (1 / 4)
    style = style_map[(factor, (mu, lam))]
    label_name = label_dict.get(
        (factor, mu, lam), f"Unknown Case ({factor},{mu},{lam})"
    )

    # Plot displacement error
    (line_u,) = ax.loglog(
        x_axis,
        displacement_errors,
        linestyle=style["linestyle"],
        marker=style["marker"],
        color=style["color_displacement"],
        label=f"{label_name}",
        markersize=8 if style["marker"] != "o" else 14,
        linewidth=5,
    )
    handles_u.append(line_u)
    labels_u.append(f"{label_name}")

    # Plot traction error
    (line_T,) = ax.loglog(
        x_axis,
        traction_errors,
        linestyle=style["linestyle"],
        marker=style["marker"],
        color=style["color_traction"],
        label=f"{label_name}",
        markersize=8 if style["marker"] != "o" else 14,
        linewidth=5,
    )
    handles_T.append(line_T)
    labels_T.append(f"{label_name}")

    # Draw convergence slope indicator (only once per unique factor)
    if factor == factors[-1] and (mu, lam) == anisotropy_pairs[-1]:
        draw_multiple_loglog_slopes(
            fig,
            ax,
            origin=(1.05 * x_axis[-2], 1.05 * displacement_errors[-2]),
            triangle_width=2.25,
            slopes=[-2],
            inverted=False,
        )

# Configure plot
ax.set_xlabel(r"$(N_x \cdot N_t)^{1/4}$", fontsize=20)
ax.set_ylabel("Relative $L^2$ error", fontsize=20)
ax.set_title("Convergence analysis results", fontsize=24)
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.xaxis.set_tick_params(which="both", labelsize=18)
ax.yaxis.set_tick_params(which="both", labelsize=18)
# ax.set_xlim(right=3e2)
ax.set_ylim(top=10e1)

# Custom Legend
handles = handles_u + handles_T
labels = labels_u + labels_T


# Create custom legend labels for the top of each column
# Add these headers as custom labels for "e_u" and "e_T"
handles_u_header = plt.Line2D([0], [0], color="white", linewidth=0)  # Invisible handle
handles_T_header = plt.Line2D([0], [0], color="white", linewidth=0)  # Invisible handle
labels_u_header = r"$\mathcal{E}_u$"
labels_T_header = r"$\mathcal{E}_T$"

# Add custom headers to the top of each column in the legend
ax.legend(
    handles=[handles_u_header] + handles_u + [handles_T_header] + handles_T,
    labels=[labels_u_header] + labels_u + [labels_T_header] + labels_T,
    fontsize=15,
    loc="upper right",
    bbox_to_anchor=(0.995, 0.995),
    borderaxespad=0,
    ncol=2,
    frameon=True,
    handleheight=2,
    handlelength=1.5,
)

# Save and show plot
figures_folder = "figures"
figures_output_dir = os.path.join(script_dir, figures_folder)
os.makedirs(figures_output_dir, exist_ok=True)
plt.savefig(
    os.path.join(figures_output_dir, "heterogeneity_errors_single_plot.png"),
    bbox_inches="tight",
)

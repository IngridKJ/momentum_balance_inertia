import os
import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from plotting.plot_utils import draw_multiple_loglog_slopes

# Manually defined dictionary for legend labels
label_dict = {
    (0, 0, 0): r"$r_h = 1, r_{a} = 0$",
    (1, 0, 0): r"$r_h = 2^{-5}, r_{a} = 0$",
    (2, 0, 0): r"$r_h = 2^{-8}, r_{a} = 0$",
    (0, 2, 2): r"$r_h = 1, r_{a} = 10^{4}$",
    (1, 2, 2): r"$r_h = 2^{-5}, r_{a} = 10^{4}$",
    (2, 2, 2): r"$r_h = 2^{-8}, r_{a} = 10^{4}$",
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

# Define unique colors for (factor, mu, lam) combinations
unique_combinations = list(product(factors, anisotropy_pairs))

# Somewhat orange, silver, somewhat blue, slategrey, somewhat redish, black
color_list = ["#FFA500", "silver", "#0289FF", "dimgray", "#B52727", "black"]
color_map = {pair: color_list[i] for i, pair in enumerate(unique_combinations)}

# Define distinct linestyles for increasing (mu, lam)
linestyles = ["--", ":"]  # No solid lines
mu_lam_styles = {pair: linestyles[i] for i, pair in enumerate(anisotropy_pairs)}


# Create figure
fig, ax = plt.subplots(figsize=(14, 9))

# Loop through all data and plot in the same figure
for factor, mu, lam, filename in file_info:
    filepath = os.path.join(output_dir, filename)
    num_cells, num_time_steps, displacement_errors, traction_errors = np.loadtxt(
        filepath, delimiter=",", skiprows=1, unpack=True, dtype=float
    )

    x_axis = (num_cells * num_time_steps) ** (1 / 4)

    # Define marker shape and color based on (mu, lam)
    marker = "D" if mu == 2 else "o"
    color_displacement = color_map[(factor, (mu, lam))]  # Custom color for displacement
    color_traction = color_map[(factor, (mu, lam))]  # Custom color for traction

    linestyle = mu_lam_styles[(mu, lam)]  # Different linestyles per anisotropy pair

    # Fetch label from dictionary (fallback to default if not found)
    label_name = label_dict.get(
        (factor, mu, lam), f"Unknown Case ({factor},{mu},{lam})"
    )

    # Plot displacement error
    ax.loglog(
        x_axis,
        displacement_errors,
        linestyle,
        marker=marker,
        color=color_displacement,
        label=f"{label_name}: Displacement",
        markersize=8 if marker == "D" else 6,
        linewidth=5 if linestyle == ".." else 3,
    )

    # Plot traction error
    ax.loglog(
        x_axis,
        traction_errors,
        linestyle,
        marker=marker,
        color=color_traction,
        alpha=0.65,
        label=f"{label_name}: Traction",
        markersize=8 if marker == "D" else 6,
        linewidth=5 if linestyle == ".." else 3,
    )

    # Draw convergence slope indicator (only once per unique factor)
    if factor == factors[-1] and (mu, lam) == anisotropy_pairs[-1]:
        draw_multiple_loglog_slopes(
            fig,
            ax,
            origin=(1.05 * x_axis[-2], displacement_errors[-2]),
            triangle_width=2.0,
            slopes=[-2],
        )

# Configure plot
ax.set_xlabel(r"$(N_x \cdot N_t)^{1/4}$", fontsize=20)
ax.set_ylabel("Relative $L^2$ error", fontsize=20)
ax.set_title("Convergence analysis results", fontsize=24)

ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# Set tick label size
ax.xaxis.set_tick_params(which="both", labelsize=18)
ax.yaxis.set_tick_params(which="both", labelsize=18)

ax.set_ylim(bottom=None, top=1e2)

# Move the legend slightly down inside the plot
ax.legend(
    fontsize=13,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=3,
    borderaxespad=0,
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

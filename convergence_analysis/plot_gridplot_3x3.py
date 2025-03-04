import os
import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from plotting.plot_utils import draw_multiple_loglog_slopes

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

# Iterate through the files to extract the information
for filename in heterogeneity_files:
    # Extract parts from the filename
    parts = filename.replace("errors_heterogeneity_", "").replace(".txt", "").split("_")

    # Extract the heterogeneity factor and mu_lam_index
    factor = float(parts[0])  # Heterogeneity factor
    mu_lam_index = int(parts[-1])  # mu_lam index (as a single number)

    # Use the index directly to assign (mu, lam)
    # The index itself corresponds to a position in the (mu, lam) grid.
    mu = lam = mu_lam_index  # No modification, just use the index directly

    file_info.append((factor, mu, lam, filename))
    factors.add(factor)
    anisotropy_pairs.add((mu, lam))

# Sort factors and anisotropy pairs
factors = sorted(factors)
anisotropy_pairs = sorted(anisotropy_pairs)

# Custom titles for subfigures
title_dict = {
    (factors[0], anisotropy_pairs[0]): "a)",
    (factors[0], anisotropy_pairs[1]): "b)",
    (factors[0], anisotropy_pairs[2]): "c)",
    (factors[1], anisotropy_pairs[0]): "d)",
    (factors[1], anisotropy_pairs[1]): "e)",
    (factors[1], anisotropy_pairs[2]): "f)",
    (factors[2], anisotropy_pairs[0]): "g)",
    (factors[2], anisotropy_pairs[1]): "h)",
    (factors[2], anisotropy_pairs[2]): "i)",
}

# Create figure
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 12), sharex=True, sharey=True)
colors = ["blue", "red"]

# Loop through factors and anisotropy pairs
for i, factor in enumerate(factors):
    for j, (mu, lam) in enumerate(anisotropy_pairs):
        ax = axes[i, j]
        matching_files = [
            f for f in file_info if f[0] == factor and f[1] == mu and f[2] == lam
        ]

        for _, (_, _, _, filename) in enumerate(matching_files):
            filepath = os.path.join(output_dir, filename)
            num_cells, num_time_steps, displacement_errors, traction_errors = (
                np.loadtxt(
                    filepath, delimiter=",", skiprows=1, unpack=True, dtype=float
                )
            )

            x_axis = (num_cells * num_time_steps) ** (1 / 4)

            ax.loglog(
                x_axis,
                displacement_errors,
                "o--",
                color="blue",
                label="Displacement error",
            )
            ax.loglog(
                x_axis,
                traction_errors,
                "o-.",
                color="red",
                alpha=0.6,
                label="Traction error",
            )

        # Configure subplot
        ax.set_title(
            title_dict.get(
                (factor, (mu, lam)), f"Factor {factor}, Mu {mu}, Lambda {lam}"
            )
        )
        ax.grid(True, which="both", color=(0.87, 0.87, 0.87))
        if i == 2:
            ax.set_xlabel(r"$(N_x \cdot N_t)^{1/4}$")
        if j == 0:
            ax.set_ylabel("Relative $L^2$ error")

        ax.legend()

        # Draw convergence slope triangle
        if matching_files:
            draw_multiple_loglog_slopes(
                fig,
                ax,
                origin=(1.05 * x_axis[-2], displacement_errors[-2]),
                triangle_width=0.7,
                slopes=[-2],
            )

# Adjust layout and save figure
fig.suptitle("Convergence analysis", fontsize=16)
figures_folder = "figures"
figures_output_dir = os.path.join(script_dir, figures_folder)
os.makedirs(figures_output_dir, exist_ok=True)
plt.savefig(os.path.join(figures_output_dir, "heterogeneity_errors_grid.png"))
plt.show()

import os
import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from plotting.plot_utils import draw_multiple_loglog_slopes

# Set paths for modules and data directories
folder_name = "convergence_analysis_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

# Define legend labels using a dictionary
label_dict = {
    (0, 0, 0): r"$r_h = 1, r_{a} = 0$",
    (1, 0, 0): r"$r_h = 2^{-4}, r_{a} = 0$",
    (2, 0, 0): r"$r_h = 2^{-8}, r_{a} = 0$",
    (0, 1, 1): r"$r_h = 1, r_{a} = 10$",
    (1, 1, 1): r"$r_h = 2^{-4}, r_{a} = 10$",
    (2, 1, 1): r"$r_h = 2^{-8}, r_{a} = 10$",
}

# Get list of heterogeneity error files
heterogeneity_files = [
    f
    for f in os.listdir(output_dir)
    if f.startswith("errors_heterogeneity") and f.endswith(".txt")
]

# Parse file info and extract unique factors and anisotropy coefficients
file_info = []
factors = set()
anisotropy_pairs = set()

for filename in heterogeneity_files:
    parts = filename.replace("errors_heterogeneity_", "").replace(".txt", "").split("_")
    factor, mu_lam_index = float(parts[0]), int(parts[-1])
    file_info.append((factor, mu_lam_index, mu_lam_index, filename))
    factors.add(factor)
    anisotropy_pairs.add((mu_lam_index, mu_lam_index))

# Sort the file info and factor/aniso lists
file_info.sort()
factors = sorted(factors)
anisotropy_pairs = sorted(anisotropy_pairs)

# Define custom styles for plotting
custom_styles = {
    "displacement": ["#D8B5DE", "black", "#9FE6A2", "black", "#55A1FF", "black"],
    "traction": ["#A45892", "darkgray", "#01840C", "darkgray", "#01308E", "darkgray"],
    "linestyles": ["-", ":", "-", ":", "-", ":"],
    "markers": ["o", "s", "o", "D", "o", "d"],
    "alpha": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

# Map styles to each combination of factor and anisotropy
style_map = {}
for i, combo in enumerate(product(factors, anisotropy_pairs)):
    style_map[combo] = {
        "color_displacement": custom_styles["displacement"][
            i % len(custom_styles["displacement"])
        ],
        "color_traction": custom_styles["traction"][i % len(custom_styles["traction"])],
        "linestyle": custom_styles["linestyles"][i % len(custom_styles["linestyles"])],
        "marker": custom_styles["markers"][i % len(custom_styles["markers"])],
        "alpha": custom_styles["alpha"][i % len(custom_styles["alpha"])],
    }

# Create figure for plotting
fig, ax = plt.subplots(figsize=(10, 9))

# Initialize lists for handles and labels
handles_u, labels_u, handles_T, labels_T = [], [], [], []

# Plot data for each file
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
        alpha=style["alpha"],
        markersize=8 if style["marker"] != "o" else 16,
        linewidth=5,
    )

    # Plot traction error
    (line_T,) = ax.loglog(
        x_axis,
        traction_errors,
        linestyle=style["linestyle"],
        marker=style["marker"],
        color=style["color_traction"],
        markersize=8 if style["marker"] != "o" else 16,
        linewidth=5,
    )

    handles_u.append(line_u)
    handles_T.append(line_T)

    # Draw convergence slope indicator for the last factor and anisotropy pair
    if factor == factors[-1] and (mu, lam) == anisotropy_pairs[-1]:
        draw_multiple_loglog_slopes(
            fig,
            ax,
            origin=(1.05 * x_axis[-2], 1.05 * displacement_errors[-2]),
            triangle_width=1.45,
            slopes=[-2],
            inverted=False,
        )

# Create 7 additional white lines and add them to the legend. This is done to ease
# further customization with the legend.
invisible_lines = [plt.Line2D([0], [0], color="white") for _ in range(7)]

# These are the labels which are common for the displacement and traction errors for
# each of the simulation runs:
common_labels = [
    "",
    r"$r_h = 1$" + ",     " + r"$ r_{a} = 0$",
    r"$r_h = 1$" + ",     " + r"$ r_{a} = 10$",
    r"$r_h = 2^{-4}$" + ",  " + r"$r_{a} = 0$",
    r"$r_h = 2^{-4}$" + ",  " + r"$r_{a} = 10$",
    r"$r_h = 2^{-8}$" + ",  " + r"$ r_{a} = 0$",
    r"$r_h = 2^{-8}$" + ",  " + r"$ r_{a} = 10$",
]


# Configure plot labels, title, grid and ticks
ax.set_xlabel(r"$(N_x \cdot N_t)^{1/4}$", fontsize=16)
ax.set_ylabel("Relative $L^2$ error", fontsize=16)
ax.set_title("Convergence analysis results", fontsize=22)
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.xaxis.set_tick_params(which="both", labelsize=15)
ax.yaxis.set_tick_params(which="both", labelsize=15)
ax.set_ylim(top=2e1)

# Create custom legend with headers for displacement and traction errors
handles_u_header = plt.Line2D([0], [0], color="white", linewidth=0)
handles_T_header = plt.Line2D([0], [0], color="white", linewidth=0)
labels_u_header, labels_T_header = (
    r"$\mathcal{E}_u$",
    r"$\mathcal{E}_T$",
)

# Modify labels_u to have empty strings for the first column
labels_u_empty = [""] * 6

# Create the legend, adjust spacing and center the headers
handles = (
    [handles_u_header] + handles_u + [handles_T_header] + handles_T + invisible_lines
)
labels = (
    [labels_u_header]
    + labels_u_empty
    + [labels_T_header]
    + labels_u_empty
    + common_labels
)

leg = ax.legend(
    handles,
    labels,
    fontsize=15,
    loc="upper right",
    bbox_to_anchor=(0.999, 0.999),
    ncol=3,
    frameon=True,
    handleheight=1.7,
    handlelength=2.2,
    columnspacing=0.8,
    labelspacing=0.275,
)

# Adjust alignment and further refine the positioning of legend entries
for vpack in leg._legend_handle_box.get_children():
    for hpack in vpack.get_children()[:1]:
        hpack.get_children()[0].set_width(0)


for vpack in leg._legend_handle_box.get_children()[:1]:
    for hpack in vpack.get_children():
        hpack.get_children()[0].set_width(0)

for vpack in leg._legend_handle_box.get_children()[2:3]:
    for hpack in vpack.get_children():
        handle = hpack.get_children()[0]
        handle.set_visible(False)


# Save the plot to a file
figures_folder = "figures"
figures_output_dir = os.path.join(script_dir, figures_folder)
os.makedirs(figures_output_dir, exist_ok=True)

plt.savefig(
    os.path.join(
        figures_output_dir, "space_time_convergence_analysis_absorbing_boundaries.png"
    ),
    bbox_inches="tight",
)

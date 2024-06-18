import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from plotting.plot_utils import draw_multiple_loglog_slopes

num_cells_exp_1_over_dim = np.array(
    [
        42,
        162,
        616,
        2400,
        9520,
    ]
) ** (1 / 2)
displacement_errors = np.array(
    [
        0.9100445389106124,
        0.14035586787738652,
        0.0322893784769148,
        0.007983883631581182,
        0.0019903374534015563,
    ]
)
force_errors = np.array(
    [
        0.8046601217605108,
        0.13703667068474937,
        0.03495333820045177,
        0.008655378820472213,
        0.002176351687069857,
    ]
)

# Plot the sample data
fig, ax = plt.subplots()
ax.loglog(
    num_cells_exp_1_over_dim,
    displacement_errors,
    "o--",
    color="firebrick",
    label="Displacement",
)
ax.loglog(
    num_cells_exp_1_over_dim,
    force_errors,
    "o--",
    color="royalblue",
    label="Traction",
)

ax.set_title("Combined temporal and spatial convergence, orthogonal wave")
ax.set_ylabel("Relative $L^2$ error")
ax.set_xlabel("$(Number\ of\ cells)^{1/2}$")
ax.legend()

# Draw the convergence triangle with multiple slopes
draw_multiple_loglog_slopes(
    fig,
    ax,
    origin=(1.1 * num_cells_exp_1_over_dim[-2], force_errors[-2]),
    triangle_width=1.0,
    slopes=[-2],
    inverted=False,
    # color=(150 / 255, 123 / 255, 151 / 255),
    labelcolor=(0.33, 0.33, 0.33),
)

ax.grid(True, which="both", color=(0.87, 0.87, 0.87))
plt.show()

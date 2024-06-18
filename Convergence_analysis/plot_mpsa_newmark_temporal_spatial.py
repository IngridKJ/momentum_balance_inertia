import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")

from plotting.plot_utils import draw_multiple_loglog_slopes

num_cells_exp_1_over_dim = np.array(
    [
        387,
        2570,
        19071,
        148129,
    ]
) ** (1 / 3)

y_disp = np.array([3.32e-01, 5.88e-02, 1.45e-02, 3.6e-03])
y_trac = np.array([3.42e-01, 7.38e-02, 2.26e-02, 8.00e-03])

# Plot the sample data
fig, ax = plt.subplots()
ax.loglog(
    num_cells_exp_1_over_dim,
    y_disp,
    "o--",
    color="firebrick",
    label="Displacement",
)
ax.loglog(
    num_cells_exp_1_over_dim,
    y_trac,
    "o--",
    color="royalblue",
    label="Traction",
)
ax.set_title("Convergence analysis: Spatial and temporal")
ax.set_xlabel("$(Number\ of\ cells)^{1/3}$")
ax.set_ylabel("Relative $L^2$ error")
ax.legend()

# Draw the convergence triangle with multiple slopes
draw_multiple_loglog_slopes(
    fig,
    ax,
    origin=(0.915 * num_cells_exp_1_over_dim[-1], 1.05 * y_disp[-1]),
    triangle_width=1.0,
    slopes=[-2],
    inverted=True,
    labelcolor=(0.33, 0.33, 0.33),
)

draw_multiple_loglog_slopes(
    fig,
    ax,
    origin=(1.1 * num_cells_exp_1_over_dim[-2], y_trac[-2]),
    triangle_width=1.0,
    slopes=[-1.5],
    inverted=False,
    labelcolor=(0.33, 0.33, 0.33),
)

ax.grid(True, which="both", color=(0.87, 0.87, 0.87))
plt.show()

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from plotting.plot_utils import draw_multiple_loglog_slopes, fetch_numbers_from_file

values = fetch_numbers_from_file("displacement_and_traction_errors_abc.txt")
num_cells = np.array(values["num_cells"]) ** (1 / 2)
displacement_error = values["displacement_error"]
force_errors = values["traction_error"]

# Plot the sample data
fig, ax = plt.subplots()
ax.loglog(
    num_cells,
    displacement_error,
    "o--",
    color="firebrick",
    label="Displacement",
)
ax.loglog(
    num_cells,
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
    origin=(1.1 * num_cells[-2], force_errors[-2]),
    triangle_width=1.0,
    slopes=[-2],
    inverted=False,
    # color=(150 / 255, 123 / 255, 151 / 255),
    labelcolor=(0.33, 0.33, 0.33),
)

ax.grid(True, which="both", color=(0.87, 0.87, 0.87))
plt.show()

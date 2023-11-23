"""Domain not filled with wave."""

import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("../")

import plotting.plot_utils as pu


def fetch_numbers_from_file(filename) -> np.ndarray:
    """Extracts numbers from a text file and puts them in a list.
    The file should have numbers separated by ",".
    """
    with open(filename, "r") as file:
        content = file.read()
        numbers = [float(num) for num in content.split(",")[:-1]]
        return numbers


def plot_the_error(filename, write_stats: bool = True, color: tuple = (0, 0, 0)):
    """Plots the error values fetched from a text file and computes max/min/average."""
    numbers = fetch_numbers_from_file(filename)[500:]

    start_from = 500
    time_steps = np.linspace(1, len(numbers), len(numbers))

    if write_stats:
        with open("error_stats.txt", "w") as file:
            file.write(f"Maximum value: {np.max(numbers)}\n")
            file.write(f"Mean value: {np.mean(numbers)}\n")
            file.write(f"Minimum value: {np.min(numbers)}\n")
            file.write(f"Last value: {numbers[-1]}\n")

    print(
        f"\n Maximum value: {np.max(numbers)}\n Mean value: {np.mean(numbers)}\n Last value: {numbers[-1]}"
    )
    plt.plot(time_steps, numbers)
    plt.xlim(start_from, len(numbers))
    plt.title(filename)


plt.show()

errors = np.array(
    [
        0.21318820784866074,
        0.07610938562077103,
        0.01931304141697523,
        0.006837330341689268,
        0.0028005276194816233,
        0.001719848260489023,
    ]
)

cell_sizes = np.array([2.0, 1.0, 0.5, 0.25, 0.125, 0.0625])


plt.loglog(
    cell_sizes,
    errors,
    "--o",
    color=pu.RGB(157 - 20, 77 + 40, 159 + 20),
    label="Displacement",
)
plt.ylabel("Relative l2 error", fontsize=13)
plt.loglog(
    cell_sizes[:-4],
    0.0015 * cell_sizes[:-4],
    color=pu.RGB(157 - 20, 77 + 70, 159 + 20),
    label="Slope 1 reference line",
)
plt.loglog(
    cell_sizes[:-4],
    0.005 * cell_sizes[:-4] ** 2,
    color=pu.RGB(200, 77 + 70, 159 + 60),
    label="Slope 2 reference line",
)
plt.xlabel("Cell size [m]", fontsize=13)
plt.title("Spatial convergence unit test", fontsize=13)
plt.grid(True, which="both", color=(0.87, 0.87, 0.87))
plt.legend(fontsize=12)
plt.show()

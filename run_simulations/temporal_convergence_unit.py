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
    numbers = fetch_numbers_from_file(filename)[50:]

    start_from = 50
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
    # plt.show()


# plot_the_error("error_100_ts.txt")
# plot_the_error("error_200_ts.txt")
# plot_the_error("error_400_ts.txt")
# plot_the_error("error_800_ts.txt")
# plot_the_error("error_1600_ts.txt")
# plot_the_error("error_3200_ts.txt")

# plt.show()

errors = np.array(
    [
        0.09911769096837494,
        0.052971377181679664,
        0.0275207821680987,
        0.015761422023380052,
        0.010423886548409166,
    ]
)

cell_sizes = np.array(
    [
        0.2,
        0.1,
        0.05,
        0.025,
        0.0125,
    ]
)

plt.loglog(
    cell_sizes,
    errors,
    "--o",
    color=pu.RGB(157 - 20, 77 + 40, 159 + 20),
    label="Displacement",
)

plt.loglog(
    cell_sizes[:-3],
    0.15 * cell_sizes[:-3],
    color=pu.RGB(157 - 20, 77 + 70, 159 + 20),
    label="Slope 1 reference line",
)
plt.ylabel("Relative l2 error", fontsize=13)
plt.xlabel("Time-step size [s]", fontsize=13)
plt.title("Temporal convergence unit test", fontsize=13)
plt.grid(True, which="both", color=(0.87, 0.87, 0.87))
plt.legend(fontsize=12)
plt.show()

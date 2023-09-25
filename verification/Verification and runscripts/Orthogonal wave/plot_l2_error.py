import numpy as np
import matplotlib.pyplot as plt


def fetch_numbers_from_file(filename) -> np.ndarray:
    """Extracts numbers from a text file and puts them in a list.

    The file should have numbers separated by ",".

    """
    with open(filename, "r") as file:
        content = file.read()
        numbers = [float(num) for num in content.split(",")[1:]]
        return numbers


def plot_the_error(filename, write_stats: bool = False):
    """Plots the error values fetched from a text file and computes max/min/average."""
    numbers = fetch_numbers_from_file(filename)

    start_from = 0
    time_steps = np.linspace(1, len(numbers), len(numbers))

    if write_stats:
        with open("error_stats.txt", "w") as file:
            file.write(f"Maximum value: {np.max(numbers)}\n")
            file.write(f"Mean value: {np.mean(numbers)}\n")
            file.write(f"Minimum value: {np.min(numbers)}\n")

    plt.plot(time_steps, numbers)
    plt.xlim(start_from, len(numbers))
    plt.title(filename)

    plt.show()

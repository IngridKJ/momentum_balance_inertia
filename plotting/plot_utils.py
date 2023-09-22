import numpy as np


def RGB(r, g, b):
    return (r / 255, g / 255, b / 255)


def convergence_ratio(error_values):
    ratios = np.zeros(len(error_values) - 1)
    for i in range(len(error_values) - 1):
        ratios[i] = error_values[i] / error_values[i + 1]
    return ratios

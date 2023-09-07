import numpy as np
import matplotlib.pyplot as plt


def RGB(r, g, b):
    return (r / 255, g / 255, b / 255)


def convergence_ratio(error_values):
    ratios = np.zeros(len(error_values) - 1)
    for i in range(len(error_values) - 1):
        ratios[i] = error_values[i] / error_values[i + 1]
    return ratios


def read_convergence_files(filename: str) -> dict:
    """Read file with error analysis.

    Hardcoded for the error_analysis.txt file created by the convergence analysis for the momentum balance. It has four columns:
        * cell_diameter
        * time_step
        * error_displacement
        * error_force

    Parameters:
        filename: The name of the file

    Returns:
        A dictionary with the error ratios for displacement and force.
    """
    info_dict = {}

    info_dict["cell_diameter"] = np.loadtxt(filename)[:, 0]
    info_dict["time_step"] = np.loadtxt(filename)[:, 1]
    info_dict["error_displacement"] = np.loadtxt(filename)[:, 2]
    info_dict["error_force"] = np.loadtxt(filename)[:, 3]

    ratio_displacement = convergence_ratio(info_dict["error_displacement"])
    ratio_force = convergence_ratio(info_dict["error_force"])

    info_dict["ratio_displacement"] = ratio_displacement
    info_dict["ratio_force"] = ratio_force

    return info_dict


def plotting(keyword: str, dic: dict) -> None:
    """"""
    if keyword == "temporal":
        axis_name = "time_step"
    elif keyword == "spatial":
        axis_name = "cell_diameter"

    plt.loglog(
        dic[axis_name],
        dic["error_displacement"],
        "o--",
        label="displacement",
        color=RGB(157, 77, 159),
    )
    plt.loglog(
        dic[axis_name],
        dic["error_force"],
        "o-.",
        label="force",
        color=RGB(49, 135, 152),
    )

    plt.title(f"Convergence rates: {keyword}")
    plt.xlabel(axis_name)
    plt.ylabel("Relative l2 error")

    plt.grid(True, which="both", linestyle="--", color=(0.87, 0.87, 0.87))
    plt.legend()

    plt.show()


# Uncomment the one you are interested in:
# plotting(
#     keyword="spatial",
#     dic=read_convergence_files(filename="error_analysis.txt"),
# )

plotting(
    keyword="temporal",
    dic=read_convergence_files(filename="error_analysis.txt"),
)

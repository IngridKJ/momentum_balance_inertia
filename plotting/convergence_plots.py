import numpy as np
import matplotlib.pyplot as plt

import plot_utils as pu


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

    ratio_displacement = pu.convergence_ratio(info_dict["error_displacement"])
    ratio_force = pu.convergence_ratio(info_dict["error_force"])

    info_dict["ratio_displacement"] = ratio_displacement
    info_dict["ratio_force"] = ratio_force

    return info_dict


def plotting(
    keyword: str,
    dic: dict,
    ref_1: bool = False,
    ref_2: bool = False,
    scaling: float = 1,
    unit: str = "m",
) -> None:
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
        color=pu.RGB(157 - 20 * 0.5, 77 + 40 * 0.5, 159 + 20 * 0.5),
    )
    plt.loglog(
        dic[axis_name],
        dic["error_force"],
        "o--",
        label="traction",
        color=pu.RGB(49, 135, 152),
    )

    plt.title(f"Convergence rates: {keyword}", fontsize=13)
    plt.xlabel(f"{axis_name} [{unit}]", fontsize=13)
    plt.ylabel("Relative l2 error", fontsize=13)

    plt.grid(True, which="both", linestyle="--", color=(0.87, 0.87, 0.87))

    # Reference slope lines
    if ref_1 is True:
        plt.loglog(
            dic[axis_name][:2],
            0.1 * dic[axis_name][:2] * scaling,
            label="Slope 1 reference line",
            color=pu.RGB(49, 135, 152),
        )
    if ref_2 is True:
        plt.loglog(
            dic[axis_name][:2],
            scaling * 0.25 * dic[axis_name][:2] ** 2,
            label="Slope 2 reference line",
            color=pu.RGB(157 - 20 * 0.5, 77 + 40 * 0.5, 159 + 20 * 0.5),
        )

    plt.legend()
    plt.show()


# Uncomment the one you are interested in:
plotting(
    keyword="spatial",
    dic=read_convergence_files(filename="spatial_10_ts_6_levels.txt"),
    ref_1=True,
    ref_2=True,
    scaling=0.0065,
)

plotting(
    keyword="temporal",
    dic=read_convergence_files(filename="temporal_128_6_levels.txt"),
    ref_2=True,
    scaling=1,
    unit="s",
)

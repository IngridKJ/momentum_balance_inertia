import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")


# -------------------------------------------------------------------------------------
def RGB(r: float, g: float, b: float) -> tuple:
    """Converts RGB values between 0 and 255 to values between 0 and 1.

    Mostly used for determining color when plotting using matplotlib, which takes values
    between 0 and 1 for colors.

    Parameters:
        r: Value for red.
        g: Value for green.
        b: Value for blue.

    Returns:
        A tuple with 0-1 values for RGB.

    """
    return (r / 255, g / 255, b / 255)


# -------------------------------------------------------------------------------------
def read_float_values(filename: str) -> np.ndarray:
    """Reads float values separated by a comma (,) from a text file.

    Parameters:
        filename: The name of the file that the float values are to be read from.

    Returns:
        An array of all the values from the file.

    """
    with open(filename, "r") as file:
        content = file.read()
        numbers = np.array([float(num) for num in content.split(",")[:-1]])
        return numbers


# -------------------------------------------------------------------------------------
def fetch_numbers_from_file(file_path: str) -> dict:
    """Fetches numbers from files with a header.

    Files supported are on the form:
        Line 1: header1 header2 header3 header4 ....
        Line 2: num num num num ...
        Line 3: num num num num ...
        ...

    Besides that, if there are any hashtags or commas in the file, these are ignored. The numbers are stored as values in a dictionary with their header as key.

    Parameters:
        file_path: Path to file/filename.

    """
    data = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip().replace("#", "")  # Remove the # character
            line = line.replace(",", "")  # Remove all commas
            if not line:
                continue  # Skip empty lines

            parts = line.split()
            if not data:  # First non-empty line is the header
                header = parts
                data = {head: [] for head in header}
            else:
                for i, part in enumerate(parts):
                    value = (
                        int(part)
                        if i == 0 and header[i].lower() in ["num_cells"]
                        else float(part)
                    )
                    data[header[i]].append(value)
    return data


# -------------------------------------------------------------------------------------


def draw_multiple_loglog_slopes(
    fig,
    ax,
    origin,
    triangle_width,
    slopes,
    dashed_extra_slopes=False,
    inverted=False,
    color=None,
    label=True,
    labelcolor=None,
):
    """This function draws slopes or "convergence triangles" into loglog plots.

    References:
        The creation of the main triangle is from:
        https://gist.github.com/w1th0utnam3/a0189dc8a2c067ccb56e1de8c317b190

        All other functionality (insertion of extra slope lines, adaptation of label
        locations etc.) are inspired from the original reference.

    Parameters:
        fig: The figure.
        ax: The axes object to draw to.
        origin: The origin coordinates of the triangle.
        triangle_width: The width in inches of the triangle.
        slopes: The list of slopes to be drawn. That is, orders of convergence for the
            "convergence triangle(s)".
        dashed_extra_slopes: Bool value of whether the non-max slopes should be dashed
        inverted: Whether to mirror the triangle (if the 90 degree angle of the
            triangle is in the lower right or upper left).
        color: Color of the triangle edges.
        label: Whether to enable labeling of the slopes. Defaults to True.
        labelcolor: The color of the slope labels. Defaults to edge color.

    Example:
        # Create a figure with plotted array values
            fig, ax = plt.subplots()
            ax.loglog(x, y)

        # Then call this function using fig, ax and other parameters:
            draw_multiple_loglog_slopes(
                fig,
                ax,
                origin=(x[-1], y_disp[-1]),
                triangle_width=1.0,
                slopes=[1, 2],
                labelcolor=(0.33, 0.33, 0.33),
            )
            plt.show()
    """

    polygon_kwargs = {}
    label_kwargs = {}
    zorder = 10
    alpha = 0.25

    if color is not None:
        polygon_kwargs["color"] = color
    else:
        polygon_kwargs["color"] = (0.25, 0.25, 0.25)
        color = (0.25, 0.25, 0.25)

    if labelcolor is not None:
        label_kwargs["color"] = labelcolor
    else:
        label_kwargs["color"] = polygon_kwargs["color"]

    polygon_kwargs["linewidth"] = 0.8 * plt.rcParams["lines.linewidth"]
    label_kwargs["fontsize"] = 0.8 * plt.rcParams["font.size"]

    if inverted:
        triangle_width = -triangle_width

    # Convert the origin into figure coordinates in inches
    origin_disp = ax.transData.transform(origin)
    origin_dpi = fig.dpi_scale_trans.inverted().transform(origin_disp)

    # Obtain the bottom-right corner in data coordinates
    corner_dpi = origin_dpi + triangle_width * np.array([1.0, 0.0])
    corner_disp = fig.dpi_scale_trans.transform(corner_dpi)
    corner = ax.transData.inverted().transform(corner_disp)

    (x1, y1) = (origin[0], origin[1])
    x2 = corner[0]

    # The width of the triangle in data coordinates
    width = x2 - x1

    # Compute offset of the slope
    log_offset = y1 / (x1 ** max(slopes))

    y2 = log_offset * ((x1 + width) ** max(slopes))

    # The vertices of the slope
    a = origin
    b = corner
    c = [x2, y2]

    # Draw the slope triangle
    X = np.array([a, b, c])
    triangle = plt.Polygon(
        X[:3, :], alpha=alpha, fill=True, zorder=zorder, **polygon_kwargs
    )
    ax.add_patch(triangle)

    # Convert vertices into display space
    a_disp = ax.transData.transform(a)
    b_disp = ax.transData.transform(b)
    c_disp = ax.transData.transform(c)

    # Figure out the center of the triangle sides in display space
    bottom_center_disp = a_disp + 0.5 * (b_disp - a_disp)
    bottom_center = ax.transData.inverted().transform(bottom_center_disp)

    right_center_disp = b_disp + 0.5 * (c_disp - b_disp)
    right_center = ax.transData.inverted().transform(right_center_disp)

    # Label alignment depending on inversion parameter and whether or not the slope is
    # negative or positive
    if np.any(slopes > np.zeros(len(slopes))):
        va_xlabel = "top" if not inverted else "bottom"
        ha_ylabel = "left" if not inverted else "right"

        # Label offset depending on inversion parameter
        offset_xlabel = (
            [0.0, -0.33 * label_kwargs["fontsize"]]
            if not inverted
            else [0.0, 0.33 * label_kwargs["fontsize"]]
        )
        offset_ylabel = (
            [0.33 * label_kwargs["fontsize"], 0.0]
            if not inverted
            else [-0.33 * label_kwargs["fontsize"], 0.0]
        )
    elif np.any(slopes < np.zeros(len(slopes))):
        va_xlabel = "bottom" if not inverted else "top"
        ha_ylabel = "right" if not inverted else "left"

        # Label offset depending on inversion parameter
        offset_xlabel = (
            [0.0, 0.33 * label_kwargs["fontsize"]]
            if not inverted
            else [0.0, -0.33 * label_kwargs["fontsize"]]
        )
        offset_ylabel = (
            [-0.33 * label_kwargs["fontsize"], 0.0]
            if not inverted
            else [0.33 * label_kwargs["fontsize"], 0.0]
        )

    # Draw the slope labels
    ax.annotate(
        "$1$",
        bottom_center,
        xytext=offset_xlabel,
        textcoords="offset points",
        ha="center",
        va=va_xlabel,
        zorder=zorder,
        **label_kwargs,
    )
    if label:
        if not inverted:
            label_point = [
                x2 + 0.02 * x2,
                y2 - 0.03 * y2,
            ]  # Upper corner if not inverted
            label_va = "center"
            ha = "left"
            if len(slopes) == 1:
                label_point[1] = (y2 + X[1][1]) / 2
                label_va = "top"
        else:
            label_point = [x2 - 0.02 * x2, y2]  # Lower corner if inverted
            label_va = "center"
            ha = "right"
            if len(slopes) == 1:
                label_point[1] = (y2 + X[1][1]) / 2
                label_va = "top"

        ax.annotate(
            f"${abs(max(slopes))}$",
            label_point,
            xytext=[0, 0],
            textcoords="offset points",
            ha=ha,
            va=label_va,
            zorder=zorder,
            **label_kwargs,
        )

    # Draw dashed slope labels
    if len(slopes) > 1:
        # Find the maximum slope value
        max_slope = max(slopes)

        # Create a new list excluding all occurrences of the maximum slope value
        dashed_slopes = [slope for slope in slopes if slope != max_slope]
        for slope in dashed_slopes:
            dashed_log_offset = y1 / (x1**slope)
            dashed_y2 = dashed_log_offset * ((x1 + width) ** slope)

            # Draw the (potentially) dashed slope lines
            if dashed_extra_slopes:
                linestyle = "--"
            else:
                linestyle = "-"
            ax.plot(
                [x1, x2],
                [y1, dashed_y2],
                linestyle=linestyle,
                linewidth=0.99,
                color=color,
                zorder=zorder - 2,
                alpha=alpha,
            )

            if inverted:
                dashed_label_point = [x2, dashed_y2]
                dashed_label_point[0] = label_point[0]

                label_va = "center"
                label_ha = "right"
            else:
                dashed_label_point = [x2, dashed_y2]
                dashed_label_point[0] = label_point[0]
                label_va = "center"

                label_ha = "left"

            # Draw slope labels
            ax.annotate(
                f"${slope}$",
                dashed_label_point,
                xytext=[0, 0],
                textcoords="offset points",
                ha=label_ha,
                va=label_va,
                zorder=zorder - 2,  # Lower zorder value
                **label_kwargs,
            )

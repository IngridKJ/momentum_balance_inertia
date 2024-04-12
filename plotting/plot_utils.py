import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

sys.path.append("../")


# -------------------------------------------------------------------------------------
def RGB(r, g, b):
    return (r / 255, g / 255, b / 255)


# -------------------------------------------------------------------------------------
def convergence_ratio(error_values):
    ratios = np.zeros(len(error_values) - 1)
    for i in range(len(error_values) - 1):
        ratios[i] = error_values[i] / error_values[i + 1]
    return ratios


# -------------------------------------------------------------------------------------
def read_float_values(filename) -> np.ndarray:
    with open(filename, "r") as file:
        content = file.read()
        numbers = np.array([float(num) for num in content.split(",")[:-1]])
        return numbers


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


# -------------------------------------------------------------------------------------


def create_and_save_animation(
    file_path: str,
    main_color: tuple,
    y_ax: tuple,
    additional_lines_info: list[tuple] = None,
    logarithmic: bool = True,
    output_file: str = None,
    save_as_mp4: bool = False,
    duration_seconds: int = None,
    framerate: int = None,
    relative: bool = True,
) -> None:
    """Function for making animation from values from a text file.

    Parameters:
        file_path: These are the lead characters of the plot. These values will have a
            dot following them as time passes.
        main_color: Tuple representing RGB colors for the main line.
        additional_lines_info: Additional files' values to be plotted. They will not
            have any animation and are thus just lineplots. It is a list consisting of
            tuples. The tuples must contain:
                * a string file path
                * a tuple with three RGB-values
                * a string with the line label name
        y_ax: Tuple with minimum and maximum value of the y-axis.
        logarithmic: Whether the y-axis should be plotted with log scale.
        output_file: Name of the .mp4 that might be saved.
        duration_seconds: Duration of the .mp4 (not sure if this actually works)
        framerate: framerate of the .mp4
        save_as_mp4: Whether an .mp4 should be saved or not.
        relative: Whether the values should be relative to the first energy value.

    Example:
        create_and_save_animation(
            file_path="energy_vals_404030_anis_long.txt",
            additional_lines_info=[
                ("energy_vals_404030_iso_long.txt", (200, 16, 0), "Iso long"),
                ("energy_vals_2d.txt", (115, 75, 210), "2d vals"),
            ],
            duration_seconds=83,
            framerate=20,
        )

    """
    # Read float values from the main file
    float_values = read_float_values(file_path)
    if relative and additional_lines_info is not None:
        float_values = float_values / max(float_values)

        # Extract information for additional lines
        additional_lines = [
            (
                (read_float_values(file)) / max(read_float_values(file)),
                RGB(red, green, blue),
                label,
            )
            for file, (red, green, blue), label in additional_lines_info
        ]
    elif relative and additional_lines_info is None:
        float_values = float_values / max(float_values)

    elif not relative and additional_lines_info is not None:
        # Extract information for additional lines
        additional_lines = [
            (read_float_values(file), RGB(red, green, blue), label)
            for file, (red, green, blue), label in additional_lines_info
        ]

    # Set up the plot
    fig, ax = plt.subplots()
    ax.grid(True, which="both", color=(0.87, 0.87, 0.87))

    if logarithmic:
        ax.set_yscale("log")

    ax.set_xlim(0, len(float_values) + 10)
    ax.set_ylim(y_ax[0], y_ax[1])
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Energy")
    ax.set_title("Energy evolution of a rotated wave")

    # Create additional persistent lines
    # First persistent vertical dashed lines:
    xmin = 10 / np.sqrt(3) * 100
    xmax = np.sqrt(200) / np.sqrt(3) * 100

    ax.axhline(
        y=0,
        xmin=0,
        xmax=12,
        color=(0, 0, 0),
        linewidth=0.5,
    )

    ax.axvline(
        x=xmin,
        ymin=0,
        ymax=5,
        color=(0.65, 0.65, 0.65),
        linestyle="--",
        linewidth=1,
    )
    ax.axvline(
        x=xmax,
        ymin=0,
        ymax=5,
        linestyle="--",
        color=(0.65, 0.65, 0.65),
        linewidth=1,
    )

    if additional_lines_info is not None:
        # Then the other energy values:
        additional_line_objects = [
            ax.plot([], [], "--", label=label, color=color)[0]
            for _, color, label in additional_lines
        ]

    # Create the initial red dashed line
    (red_line,) = ax.plot(
        [],
        [],
        color=RGB(main_color[0], main_color[1], main_color[2]),
        label="$\\theta = \pi/4$",
    )

    # Create the dynamic red dot
    (red_dot,) = ax.plot(
        [], [], "o", color=RGB(main_color[0], main_color[1], main_color[2])
    )

    # Create legend
    ax.legend()

    # Function to initialize the plot
    def init():
        red_line.set_data([], [])
        red_dot.set_data([], [])

        if additional_lines_info is not None:
            for line in additional_line_objects:
                line.set_data([], [])

            return tuple([red_line, red_dot] + additional_line_objects)
        else:
            return tuple([red_line, red_dot])

    # Function to update the plot during animation
    def update(frame, red_line, red_dot, *additional_lines):
        x = np.arange(0, frame + 1)
        y = float_values[: frame + 1]

        # Update the red dashed line with the full history
        red_line.set_data(np.arange(len(float_values)), float_values)

        # Update the red dot for the current value
        red_dot.set_data(frame, float_values[frame])

        if additional_lines_info is not None:
            # Update additional persistent lines with their values
            for line, (values, _, _) in zip(additional_line_objects, additional_lines):
                line.set_data(np.arange(len(values)), values)

            return tuple([red_line, red_dot] + additional_line_objects)
        else:
            return tuple([red_line, red_dot])

    # Calculate the number of frames based on the duration and framerate
    num_frames = int(duration_seconds * framerate)

    if additional_lines_info is not None:
        # Create the animation
        animation = FuncAnimation(
            fig,
            update,
            init_func=init,
            fargs=(red_line, red_dot) + tuple(additional_lines),
            frames=num_frames,
            interval=1000 / framerate,
        )
    else:
        # Create the animation
        animation = FuncAnimation(
            fig,
            update,
            init_func=init,
            fargs=(red_line, red_dot),
            frames=num_frames,
            interval=1000 / framerate,
        )

    # Calculate the correct fps for the desired duration
    correct_fps = num_frames / duration_seconds

    # Save the animation as an .mp4 file if specified
    if save_as_mp4:
        animation.save(output_file, writer="ffmpeg", fps=correct_fps)

    # Show the plot
    plt.show()


# -------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import sys

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


def create_and_save_animation(
    file_path: str,
    main_color: tuple,
    additional_lines_info: list[tuple],
    y_ax: tuple,
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
    if relative:
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
    else:
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

    # Then the other energy values:
    additional_line_objects = [
        ax.plot([], [], "--", label=label, color=color)[0]
        for _, color, label in additional_lines
    ]

    # Create the initial red dashed line
    (red_line,) = ax.plot(
        [], [], color=RGB(main_color[0], main_color[1], main_color[2]), label="Energy"
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

        for line in additional_line_objects:
            line.set_data([], [])

        return tuple([red_line, red_dot] + additional_line_objects)

    # Function to update the plot during animation
    def update(frame, red_line, red_dot, *additional_lines):
        x = np.arange(0, frame + 1)
        y = float_values[: frame + 1]

        # Update the red dashed line with the full history
        red_line.set_data(np.arange(len(float_values)), float_values)

        # Update the red dot for the current value
        red_dot.set_data(frame, float_values[frame])

        # Update additional persistent lines with their values
        for line, (values, _, _) in zip(additional_line_objects, additional_lines):
            line.set_data(np.arange(len(values)), values)

        return tuple([red_line, red_dot] + additional_line_objects)

    # Calculate the number of frames based on the duration and framerate
    num_frames = int(duration_seconds * framerate)

    # Create the animation
    animation = FuncAnimation(
        fig,
        update,
        init_func=init,
        fargs=(red_line, red_dot) + tuple(additional_lines),
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

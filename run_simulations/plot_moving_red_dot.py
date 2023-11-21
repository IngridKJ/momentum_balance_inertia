"""File for plotting animation of total energy in the system.

Is also possible to export the animation as an mp4. Some warnings appear, but everything
seems fine enough.

"""
import porepy as pp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def read_float_values(filename) -> np.ndarray:
    """Extracts numbers from a text file and puts them in a list.

    The file should have numbers separated by ",". It also ends with a ",", therefore we
    do not include the last element of the file.

    """
    with open(filename, "r") as file:
        content = file.read()
        numbers = np.array([float(num) for num in content.split(",")[:-1]])
        return numbers / numbers[0]


# Function to update the plot during animation
def update(frame):
    x = np.arange(0, frame + 1)
    y = float_values[: frame + 1]

    # Update the dashed line
    line.set_data(x, y)

    # Update the red dot for the current value
    dot.set_data(frame, float_values[frame])

    return line, dot


# File path to the text file with float values
file_path = "energy_vals.txt"

# Read float values from the file
float_values = read_float_values(file_path)

# Set up the plot
fig, ax = plt.subplots()
ax.grid(True, which="both", color=(0.87, 0.87, 0.87))
ax.set_yscale("log")
ax.set_xlim(0, len(float_values))
ax.set_ylim(min(float_values), max(float_values))
ax.set_xlabel("Frame")
ax.set_ylabel("Float Value")

# Create the initial dashed line
(line,) = ax.plot([], [], "r--", label="Dashed Line")  # 'r--' means red dashed line

# Create the initial red dot
(dot,) = ax.plot(0, float_values[0], "ro", label="Red Dot")  # 'ro' means red dot

# Create legend
ax.legend()


# Function to initialize the plot
def init():
    line.set_data([], [])
    dot.set_data([], [])
    return line, dot


# Duration parameter in seconds: Don't think this is necessary. Only focus on framerate.
duration_seconds = 83

# Calculate the number of frames based on the duration and framerate
framerate = 20
num_frames = int(duration_seconds * framerate)

# Create the animation
animation = FuncAnimation(
    fig, update, init_func=init, frames=num_frames, interval=1000 / framerate
)

# Calculate the correct fps for the desired duration
correct_fps = num_frames / duration_seconds

# # Save the animation as an .mp4 file with the correct fps
# output_file = "animation_output.mp4"
# animation.save(output_file, writer="ffmpeg", fps=correct_fps)

# Show the plot
plt.show()

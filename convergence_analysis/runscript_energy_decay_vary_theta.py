import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp

sys.path.append("../")

import plotting.plot_utils as pu
import run_models.run_linear_model as rlm
from analysis_models.model_energy_decay_analysis import ModelEnergyDecay

# Prepare path for generated output files
folder_name = "energy_values"
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

# Coarse/Fine variables and plotting (save figure)
coarse = True
save_figure = True


class MeshingAndExport:
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.1 if coarse else 0.015625, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def data_to_export(self):
        """Define the data to export to vtu.

        Returns:
            list: List of tuples containing the subdomain, variable name,
            and values to export.

        """
        data = super().data_to_export()
        sd = self.mdg.subdomains(dim=self.nd)[0]
        vel_op = self.velocity_time_dep_array([sd]) * self.velocity_time_dep_array([sd])
        vel_op_int = self.volume_integral(integrand=vel_op, grids=[sd], dim=2)
        vel_op_int_val = self.equation_system.evaluate(vel_op_int)

        vel = self.equation_system.evaluate(self.velocity_time_dep_array([sd]))

        data.append((sd, "energy", vel_op_int_val))
        data.append((sd, "velocity", vel))

        with open(os.path.join(output_dir, f"energy_values_{i}.txt"), "a") as file:
            file.write(f"{np.sum(vel_op_int_val)},")
        return data

    def write_pvd_and_vtu(self) -> None:
        """Override method such that pvd and vtu files are not created."""
        self.data_to_export()


class RotationAngle:
    @property
    def rotation_angle(self) -> float:
        return self.rotation_angle_from_list


class ModelSetupEnergyDecayAnalysis(
    MeshingAndExport,
    RotationAngle,
    ModelEnergyDecay,
):
    """Model class setup for the energy decay analysis with varying theta."""


# This is where the simulation actually is run. We loop through different wave rotation
# angles and run the model class once per angle.
rotation_angles = np.array([np.pi / 6, np.pi / 3, np.pi / 4, np.pi / 8])
i = 1
for rotation_angle in rotation_angles:
    tf = 15.0
    time_steps = 300
    dt = tf / time_steps

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
    )

    solid_constants = pp.SolidConstants(lame_lambda=0.01, shear_modulus=0.01)
    material_constants = {"solid": solid_constants}

    params = {
        "time_manager": time_manager,
        "grid_type": "simplex",
        "manufactured_solution": "diagonal_wave",
        "progressbars": True,
        "material_constants": material_constants,
    }

    model = ModelSetupEnergyDecayAnalysis(params)
    model.rotation_angle_from_list = rotation_angle
    model.angle_index = i
    with open(os.path.join(output_dir, f"energy_values_{i}.txt"), "w") as file:
        pass

    rlm.run_linear_model(model, params)
    i += 1

# Plotting from here and down
if save_figure:
    plt.figure(figsize=(7, 5))
    # Tuple value in dictionary:
    #   * Legend text
    #   * Color
    #   * Dashed/not dashed line
    #   * Logarithmic y scale/not logarithmic y scale.
    index_angle_dict = {
        1: (r"$\theta = \pi/6$", "#FF9E57", False, True),
        2: (r"$\theta = \pi/3$", "#A45892", True, True),
        3: (r"$\theta = \pi/4$", pu.RGB(0, 0, 0), True, True),
        4: (r"$\theta = \pi/8$", "#55A1FF", False, True),
    }

    for key, value in index_angle_dict.items():
        filename = os.path.join(output_dir, f"energy_values_{key}.txt")
        energy_values = (
            pu.read_float_values(filename=filename)
            / pu.read_float_values(filename=filename)[0]
        )
        final_time = 15
        time_values = np.linspace(0, final_time, len(energy_values))

        plt.yscale("log" if value[3] else "linear")
        plt.plot(
            time_values,
            energy_values,
            label=value[0],
            color=value[1],
            linestyle="-" if not value[2] else "--",
            linewidth=2,
        )

    plt.axvline(
        x=10 / np.sqrt(3),
        ymin=0,
        ymax=5,
        color=(0.65, 0.65, 0.65),
        linestyle="--",
        linewidth=1,
    )
    plt.axvline(
        x=10 * np.sqrt(6) / 3,
        ymin=0,
        ymax=5,
        color=(0.65, 0.65, 0.65),
        linestyle="--",
        linewidth=1,
    )

    plt.axhline(
        y=0,
        xmin=0,
        xmax=12,
        color=(0, 0, 0),
        linewidth=0.5,
    )

    plt.xlabel("Time [s]", fontsize=14)
    plt.ylabel("$\\frac{E}{E_0}$", fontsize=16)
    plt.title("Energy evolution with respect to time")
    plt.legend(fontsize=12)

    folder_name = "figures"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "energy_decay_vary_theta.png"))

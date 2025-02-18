import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp

sys.path.append("../")
import plotting.plot_utils as pu
import run_models.run_linear_model as rlm
from models import DynamicMomentumBalanceABCLinear
from utils import u_v_a_wrap

# Prepare path for generated output files
folder_name = "energy_values"
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, folder_name)
os.makedirs(output_dir, exist_ok=True)

# Coarse/Fine variables and plotting (save figure)
coarse = True
save_figure = True


# Model class for setting up and running the simulation from here and onwards.
class BoundaryConditionsEnergyDecayAnalysis:
    def initial_condition_bc(self, bg: pp.BoundaryGrid) -> np.ndarray:
        dt = self.time_manager.dt
        vals_0 = self.initial_condition_value_function(bg=bg, t=0)
        vals_1 = self.initial_condition_value_function(bg=bg, t=0 - dt)

        data = self.mdg.boundary_grid_data(bg)

        # The values for the 0th and -1th time step are to be stored
        pp.set_solution_values(
            name="boundary_displacement_values",
            values=vals_1,
            data=data,
            time_step_index=1,
        )
        pp.set_solution_values(
            name="boundary_displacement_values",
            values=vals_0,
            data=data,
            time_step_index=0,
        )
        return vals_0

    def initial_condition_value_function(self, bg, t):
        """Assigning initial bc values."""
        sd = bg.parent

        x = sd.face_centers[0, :]
        y = sd.face_centers[1, :]

        boundary_sides = self.domain_boundary_sides(sd)

        inds_north = np.where(boundary_sides.north)[0]
        inds_south = np.where(boundary_sides.south)[0]
        inds_west = np.where(boundary_sides.west)[0]
        inds_east = np.where(boundary_sides.east)[0]

        bc_vals = np.zeros((sd.dim, sd.num_faces))

        displacement_function = u_v_a_wrap(model=self)

        # North
        bc_vals[0, :][inds_north] = displacement_function[0](
            x[inds_north], y[inds_north], t
        )
        bc_vals[1, :][inds_north] = displacement_function[1](
            x[inds_north], y[inds_north], t
        )

        # East
        bc_vals[0, :][inds_east] = displacement_function[0](
            x[inds_east], y[inds_east], t
        )
        bc_vals[1, :][inds_east] = displacement_function[1](
            x[inds_east], y[inds_east], t
        )

        # West
        bc_vals[0, :][inds_west] = displacement_function[0](
            x[inds_west], y[inds_west], t
        )
        bc_vals[1, :][inds_west] = displacement_function[1](
            x[inds_west], y[inds_west], t
        )

        # South
        bc_vals[0, :][inds_south] = displacement_function[0](
            x[inds_south], y[inds_south], t
        )
        bc_vals[1, :][inds_south] = displacement_function[1](
            x[inds_south], y[inds_south], t
        )

        bc_vals = bc_vals.ravel("F")

        bc_vals = bg.projection(self.nd) @ bc_vals.ravel("F")
        return bc_vals


class SourceValuesEnergyDecayAnalysis:
    def evaluate_mechanics_source(self, f: list, sd: pp.Grid, t: float) -> np.ndarray:
        vals = np.zeros((self.nd, sd.num_cells))
        return vals.ravel("F")


class Geometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(self.cell_size_value, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class ExportEnergy:
    def data_to_export(self):
        data = super().data_to_export()
        for sd in self.mdg.subdomains(dim=self.nd):
            vel_op = self.velocity_time_dep_array([sd]) * self.velocity_time_dep_array(
                [sd]
            )
            vel_op_int = self.volume_integral(integrand=vel_op, grids=[sd], dim=self.nd)
            vel_op_int_val = vel_op_int.value(self.equation_system)

            vel = self.velocity_time_dep_array([sd]).value(self.equation_system)

            data.append((sd, "energy", vel_op_int_val))
            data.append((sd, "velocity", vel))

            with open(os.path.join(output_dir, f"energy_values_{i}.txt"), "a") as file:
                file.write(f"{np.sum(vel_op_int_val)},")

        return data


class RotationAngle:
    @property
    def rotation_angle(self) -> float:
        return np.pi / 4


class ModelSetupEnergyDecayAnalysis(
    BoundaryConditionsEnergyDecayAnalysis,
    SourceValuesEnergyDecayAnalysis,
    Geometry,
    ExportEnergy,
    RotationAngle,
    DynamicMomentumBalanceABCLinear,
):
    def write_pvd_and_vtu(self) -> None:
        """Override method such that pvd and vtu files are not created."""
        self.data_to_export()


# This is where the simulation actually is run. We loop through different space
# refinements and run the model class once per refinement.
if coarse:
    dxs = np.array([1 / 2**i for i in range(5, 7)])
else:
    dxs = np.array([1 / 2**i for i in range(5, 10)])
i = 8
for dx in dxs:
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
    model.cell_size_value = dx
    model.index = i

    with open(os.path.join(output_dir, f"energy_values_{i}.txt"), "w") as file:
        pass
    rlm.run_linear_model(model, params)
    i += 1


# Plotting from here and down

# Tuple value in dictionary:
#   * Legend text
#   * Color
#   * Dashed/not dashed line
#   * Logarithmic y scale/not logarithmic y scale.
if save_figure:
    plt.figure(figsize=(7, 5))
    if coarse:
        index_dx_dict = {
            9: ("$\Delta x = 1/64$", pu.RGB(255, 193, 7), True, True),
            8: ("$\Delta x = 1/32$", pu.RGB(0, 0, 0), True, True),
        }
    else:
        index_dx_dict = {
            12: ("$\Delta x = 1/512$", pu.RGB(0, 0, 0), False, True),
            11: ("$\Delta x = 1/256$", pu.RGB(216, 27, 96), False, True),
            10: ("$\Delta x = 1/128$", pu.RGB(30, 136, 229), True, True),
            9: ("$\Delta x = 1/64$", pu.RGB(255, 193, 7), True, True),
            8: ("$\Delta x = 1/32$", pu.RGB(0, 0, 0), True, True),
        }

    for key, value in index_dx_dict.items():
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

    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("$\\frac{E}{E_0}$", fontsize=16)
    plt.title("Energy evolution with respect to time")
    plt.legend()

    folder_name = "figures"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "energy_decay_space_refinement.png"))

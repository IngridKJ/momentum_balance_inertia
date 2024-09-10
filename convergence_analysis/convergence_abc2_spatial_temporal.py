import sys

sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
import utils
from plotting.plot_utils import draw_multiple_loglog_slopes
from porepy.applications.convergence_analysis import ConvergenceAnalysis

from model_convergence_ABC2 import ABC2Model


class MyUnitGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.25 / 2 ** (self.refinement), "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class SpatialRefinementModel(MyUnitGeometry, ABC2Model):
    def data_to_export(self):
        data = super().data_to_export()

        sd = self.mdg.subdomains(dim=self.nd)[0]
        x_cc = sd.cell_centers[0, :]
        time = self.time_manager.time
        cp = self.primary_wave_speed(is_scalar=True)

        # Exact displacement and traction
        u_exact = np.array([np.sin(time - x_cc / cp), np.zeros(len(x_cc))]).ravel("F")

        u, x, y, t = utils.symbolic_representation(model=self)
        _, sigma, _ = utils.symbolic_equation_terms(model=self, u=u, x=x, y=y, t=t)
        T_exact = self.elastic_force(
            sd=sd, sigma_total=sigma, time=self.time_manager.time
        )

        # Approximated displacement and traction
        displacement_ad = self.displacement([sd])
        u_approximate = displacement_ad.value(self.equation_system)
        traction_ad = self.stress([sd])
        T_approximate = traction_ad.value(self.equation_system)

        # Compute error for displacement and traction
        error_displacement = ConvergenceAnalysis.l2_error(
            grid=sd,
            true_array=u_exact,
            approx_array=u_approximate,
            is_scalar=False,
            is_cc=True,
            relative=True,
        )
        error_traction = ConvergenceAnalysis.l2_error(
            grid=sd,
            true_array=T_exact,
            approx_array=T_approximate,
            is_scalar=False,
            is_cc=False,
            relative=True,
        )

        if self.time_manager.final_time_reached():
            with open("displacement_and_traction_errors_abc.txt", "a") as file:
                file.write(f"{sd.num_cells}, {error_displacement}, {error_traction}\n")
        return data


with open(f"displacement_and_traction_errors_abc.txt", "w") as file:
    file.write("num_cells, displacement_error, traction_error\n")

refinements = np.array([0, 1, 2, 3, 4])
for refinement_coefficient in refinements:
    tf = 15.0
    time_steps = 15 * (2**refinement_coefficient)
    dt = tf / time_steps

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
    )
    # Unit square:
    solid_constants = pp.SolidConstants({"lame_lambda": 0.01, "shear_modulus": 0.01})
    material_constants = {"solid": solid_constants}

    params = {
        "time_manager": time_manager,
        "grid_type": "simplex",
        "folder_name": "testing_diag_wave",
        "manufactured_solution": "unit_test",
        "progressbars": True,
        "material_constants": material_constants,
    }

    model = SpatialRefinementModel(params)
    model.refinement = refinement_coefficient
    pp.run_time_dependent_model(model, params)


# Read the file and extract data into numpy arrays
num_cells, displacement_errors, traction_errors = np.loadtxt(
    "displacement_and_traction_errors_abc.txt",
    delimiter=",",
    skiprows=1,
    unpack=True,
    dtype=float,
)


num_cells_exp_1_over_dim = num_cells ** (1 / 2)

# Plot the sample data
fig, ax = plt.subplots()
ax.loglog(
    num_cells_exp_1_over_dim,
    displacement_errors,
    "o--",
    color="firebrick",
    label="Displacement",
)
ax.loglog(
    num_cells_exp_1_over_dim,
    traction_errors,
    "o--",
    color="royalblue",
    label="Traction",
)

ax.set_title("Combined temporal and spatial convergence, orthogonal wave")
ax.set_ylabel("Relative $L^2$ error")
ax.set_xlabel("$(Number\ of\ cells)^{1/2}$")
ax.legend()

# Draw the convergence triangle with multiple slopes
draw_multiple_loglog_slopes(
    fig,
    ax,
    origin=(1.1 * num_cells_exp_1_over_dim[-2], traction_errors[-2]),
    triangle_width=1.0,
    slopes=[-2],
    inverted=False,
    labelcolor=(0.33, 0.33, 0.33),
)

ax.grid(True, which="both", color=(0.87, 0.87, 0.87))
plt.show()

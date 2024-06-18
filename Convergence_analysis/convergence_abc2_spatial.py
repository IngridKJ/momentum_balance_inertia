import numpy as np
import porepy as pp
from model_convergence_ABC2 import ABC2Model
from porepy.applications.convergence_analysis import ConvergenceAnalysis


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
        mesh_args: dict[str, float] = {
            "cell_size": 0.25 / 2 ** (self.refinement) / self.units.m
        }
        return mesh_args


class SpatialRefinementModel(MyUnitGeometry, ABC2Model):
    def data_to_export(self):
        """Define the data to export to vtu.

        Returns:
            list: List of tuples containing the subdomain, variable name,
            and values to export.

        """
        data = super().data_to_export()
        for sd in self.mdg.subdomains(dim=self.nd):
            x = sd.cell_centers[0, :]
            t = self.time_manager.time
            cp = self.primary_wave_speed(is_scalar=True)

            d = self.mdg.subdomain_data(sd)

            u_e = np.sin(t - x / cp)
            u_h = pp.get_solution_values(name="u", data=d, time_step_index=0)

            u_h = np.reshape(u_h, (self.nd, sd.num_cells), "F")[0]

            error = ConvergenceAnalysis.l2_error(
                grid=sd,
                true_array=u_e,
                approx_array=u_h,
                is_scalar=True,
                is_cc=True,
                relative=True,
            )

            data.append((sd, "analytical", u_e))
            data.append((sd, "diff", u_h - u_e))
            with open(f"error_{self.refinement}_spatial.txt", "a") as file:
                file.write(f"{error},")
        return data


refinements = np.array([0, 1, 2, 3, 4])


def read_float_values(filename) -> np.ndarray:
    with open(filename, "r") as file:
        content = file.read()
        numbers = np.array([float(num) for num in content.split(",")[:-1]])
        return numbers


with open(f"spatial_refinement.txt", "w") as file:
    pass

for refinement_coefficient in refinements:
    tf = 15.0
    time_steps = 900
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
        "manufactured_solution": "unit_test",
        "progressbars": True,
        "material_constants": material_constants,
    }
    model = SpatialRefinementModel(params)
    model.refinement = refinement_coefficient
    with open(f"error_{model.refinement}_spatial.txt", "w") as file:
        pass

    pp.run_time_dependent_model(model, params)
    error_value = read_float_values(filename=f"error_{model.refinement}_spatial.txt")[
        -1
    ]
    with open(f"spatial_refinement.txt", "a") as file:
        file.write(
            f"Refinement coefficient: {refinement_coefficient}. Error value: {error_value}\n"
        )

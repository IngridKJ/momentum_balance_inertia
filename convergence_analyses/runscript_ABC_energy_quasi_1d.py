import sys

import numpy as np
import porepy as pp

sys.path.append("../")

from model_convergence_ABC2 import ABC2Model


class BoundaryConditionAndSourceValuesEnergyTest:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for the absorbing boundary condition model class.

        Assigns Robin boundaries to all subdomain boundaries. This includes setting the
        Robin weight by a helper function.

        """
        # Fetch boundary sides and assign type of boundary condition for the different
        # sides
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.east + bounds.west, "rob")

        # Calling helper function for assigning the Robin weight
        self.assign_robin_weight(sd=sd, bc=bc)
        return bc

    def source_values(self, f, sd, t) -> np.ndarray:
        vals = np.zeros((self.nd, sd.num_cells))
        return vals.ravel("F")


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}
        box.update({"ymin": 0, "ymax": y})
        return pp.Domain(box)

    def set_domain(self) -> None:
        x = self.solid.convert_units(1.0, "m")
        y = self.solid.convert_units(1.0, "m")
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.015625, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class ExportEnergy:
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
        vel_op_int_val = vel_op_int.value(self.equation_system)

        vel = self.velocity_time_dep_array([sd]).value(self.equation_system)

        data.append((sd, "energy", vel_op_int_val))
        data.append((sd, "velocity", vel))

        with open(f"energy_values_5.txt", "a") as file:
            file.write(f"{np.sum(vel_op_int_val)},")

        return data


class EnergyTestModel(
    BoundaryConditionAndSourceValuesEnergyTest,
    MyGeometry,
    ExportEnergy,
    ABC2Model,
): ...


tf = 15.0
time_steps = 300
dt = tf / time_steps

time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
)

solid_constants = pp.SolidConstants({"lame_lambda": 0.01, "shear_modulus": 0.01})
material_constants = {"solid": solid_constants}

params = {
    "time_manager": time_manager,
    "grid_type": "simplex",
    "manufactured_solution": "unit_test",
    "progressbars": True,
    "material_constants": material_constants,
}

model = EnergyTestModel(params)

with open(f"energy_values_5.txt", "w") as file:
    pass
pp.run_time_dependent_model(model, params)

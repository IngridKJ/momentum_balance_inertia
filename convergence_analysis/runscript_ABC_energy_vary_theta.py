import sys

import numpy as np
import porepy as pp

sys.path.append("../")

from models import DynamicMomentumBalanceABC2
from utils import u_v_a_wrap


class BoundaryConditionsEnergyTest:
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


class SourceValuesEnergyTest:
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

        with open(f"energy_values_{self.rotation_angle_index}.txt", "a") as file:
            file.write(f"{np.sum(vel_op_int_val)},")

        return data


class RotationAngle:
    @property
    def rotation_angle(self) -> float:
        return self.rotation_angle_from_list


class EnergyTestModel(
    BoundaryConditionsEnergyTest,
    SourceValuesEnergyTest,
    MyGeometry,
    ExportEnergy,
    RotationAngle,
    DynamicMomentumBalanceABC2,
): ...


rotation_angles = np.array([0, np.pi / 6, np.pi / 3, np.pi / 4, np.pi / 8])
i = 0
for rotation_angle in rotation_angles:
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
        "manufactured_solution": "diagonal_wave",
        "progressbars": True,
        "material_constants": material_constants,
    }

    model = EnergyTestModel(params)
    model.rotation_angle_from_list = rotation_angle
    model.rotation_angle_index = i
    with open(f"energy_values_{i}.txt", "w") as file:
        pass
    pp.run_time_dependent_model(model, params)
    i += 1

"""We have a setup of a sine-wave travelling at an angle from the lower left corner to
the upper left corner. 

Boundary conditions are: 
* all absorbing

What drives the wave: 
* Initial displacement, velocity and acceleration corresponding to some analytical wave.

"""

import porepy as pp
import numpy as np


import sys

sys.path.append("../")

from models import MomentumBalanceABC


from utils import get_boundary_cells


class RotationAngle:
    @property
    def rotation_angle(self) -> float:
        return np.pi / 4


class InitialConditionSourceTermUnitTest:
    def initial_condition(self):
        """Assigning initial bc values."""
        super().initial_condition()

        sd = self.mdg.subdomains(dim=self.nd)[0]
        data = self.mdg.subdomain_data(sd)

        cp = self.primary_wave_speed
        t = 0
        x = sd.face_centers[0, :]
        y = sd.face_centers[1, :]
        alpha = self.rotation_angle

        inds_north = get_boundary_cells(
            self=self, sd=sd, side="north", return_faces=True
        )
        inds_east = get_boundary_cells(self=self, sd=sd, side="east", return_faces=True)
        inds_west = get_boundary_cells(self=self, sd=sd, side="west", return_faces=True)
        inds_south = get_boundary_cells(
            self=self, sd=sd, side="south", return_faces=True
        )

        bc_vals = np.zeros((sd.dim, sd.num_faces))

        # North
        bc_vals[0, :][inds_north] = np.sin(
            t - (x[inds_north] * np.cos(alpha) + y[inds_north] * np.sin(alpha)) / (cp)
        )
        bc_vals[1, :][inds_north] = np.sin(
            t - (x[inds_north] * np.cos(alpha) + y[inds_north] * np.sin(alpha)) / (cp)
        )

        # East
        bc_vals[0, :][inds_east] = np.sin(
            t - (x[inds_east] * np.cos(alpha) + y[inds_east] * np.sin(alpha)) / (cp)
        )
        bc_vals[1, :][inds_east] = np.sin(
            t - (x[inds_east] * np.cos(alpha) + y[inds_east] * np.sin(alpha)) / (cp)
        )

        # West
        bc_vals[0, :][inds_west] = np.sin(
            t - (x[inds_west] * np.cos(alpha) + y[inds_west] * np.sin(alpha)) / (cp)
        )
        bc_vals[1, :][inds_west] = np.sin(
            t - (x[inds_west] * np.cos(alpha) + y[inds_west] * np.sin(alpha)) / (cp)
        )

        # South
        bc_vals[0, :][inds_south] = np.sin(
            t - (x[inds_south] * np.cos(alpha) + y[inds_south] * np.sin(alpha)) / (cp)
        )
        bc_vals[1, :][inds_west] = np.sin(
            t - (x[inds_south] * np.cos(alpha) + y[inds_south] * np.sin(alpha)) / (cp)
        )

        bc_vals = bc_vals.ravel("F")

        pp.set_solution_values(
            name=self.bc_values_mechanics_key,
            values=bc_vals,
            data=data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name=self.bc_values_mechanics_key,
            values=bc_vals,
            data=data,
            iterate_index=0,
        )

    def source_values(self, f, sd, t) -> np.ndarray:
        """Computes the integrated source values by the source function.

        Parameters:
            f: Function depending on time and space for the source term.
            sd: Subdomain where the source term is defined.
            t: Current time in the time-stepping.

        Returns:
            An array of source values.

        """
        vals = np.zeros((self.nd, sd.num_cells))
        return vals.ravel("F")


class ExportErrors:
    def data_to_export(self):
        """Define the data to export to vtu.

        Returns:
            list: List of tuples containing the subdomain, variable name,
            and values to export.

        """
        data = super().data_to_export()
        for sd in self.mdg.subdomains(dim=self.nd):
            vel_op = self.velocity_time_dep_array([sd]) * self.velocity_time_dep_array(
                [sd]
            )
            vel_op_int = self.volume_integral(integrand=vel_op, grids=[sd], dim=2)
            vel_op_int_val = vel_op_int.evaluate(self.equation_system)

            data.append((sd, "energy", vel_op_int_val))

            with open("energy_vals.txt", "a") as file:
                file.write(f"{np.sum(vel_op_int_val)},")

        return data


class BaseScriptModel(
    RotationAngle,
    InitialConditionSourceTermUnitTest,
    ExportErrors,
    MomentumBalanceABC,
):
    ...

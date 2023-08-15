import porepy as pp
import numpy as np

from models import DynamicMomentumBalance

import sys

sys.path.append("../utils")

from utils import body_force_function
from utils import u_v_a_wrap


class BoundaryAndInitialCond:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        # 2D hardcoded
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "dir",
        )
        return bc

    def initial_displacement(self, dofs: int) -> np.ndarray:
        """Initial displacement values."""
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        displacement_function = u_v_a_wrap(self)
        vals[0] = displacement_function[0](x, y, t)
        vals[1] = displacement_function[1](x, y, t)

        return vals.ravel("F")

    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial velocity values."""
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        velocity_function = u_v_a_wrap(self, return_dt=True)
        vals[0] = velocity_function[0](x, y, t)
        vals[1] = velocity_function[1](x, y, t)

        return vals.ravel("F")

    def initial_acceleration(self, dofs: int) -> np.ndarray:
        """Initial acceleration values."""
        sd = self.mdg.subdomains()[0]

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        acceleration_function = u_v_a_wrap(self, return_ddt=True)
        vals[0] = acceleration_function[0](x, y, t)
        vals[1] = acceleration_function[1](x, y, t)

        return vals.ravel("F")


class Source:
    def before_nonlinear_loop(self) -> None:
        """Update values of external sources.

        Currently also used for setting exact values for displacement, velocity and
        acceleration for use in debugging/verification/convergence analysis.

        """
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        source_func = body_force_function(self)
        mech_source = self.source_values(source_func, sd, t)
        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, iterate_index=0
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

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        x_val = f[0](x, y, t)
        y_val = f[1](x, y, t)

        cell_volume = sd.cell_volumes

        vals[0] = x_val * cell_volume
        vals[1] = y_val * cell_volume

        return vals.ravel("F")


class MomentumBalanceTimeDepSource(
    BoundaryAndInitialCond,
    Source,
    DynamicMomentumBalance,
):
    ...

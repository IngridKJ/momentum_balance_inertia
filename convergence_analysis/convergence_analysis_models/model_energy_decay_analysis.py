import sys

import numpy as np
import porepy as pp

sys.path.append("../")

from models import DynamicMomentumBalanceABCLinear
from utils import u_v_a_wrap


class BoundaryConditionsEnergyDecayAnalysis:
    def initial_condition_bc(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Method for setting initial boundary values for 0th and -1st time step.

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the initial displacement boundary values.

        """
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
        """Initial values for the absorbing boundary.

        Parameters:
            bg: The boundary grid where the initial values are to be defined.
            t: The time which the values are to be defined for. Typically t = 0 or t =
                -dt, as we set initial values for the boundary condition both at initial
                time and one time-step back in time.

        Returns:
            An array of the initial boundary values.

        """

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
        """Setting a rectangular domain."""
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}
        box.update({"ymin": 0, "ymax": y})
        return pp.Domain(box)

    def set_domain(self) -> None:
        """Defining the dimensions of the rectangular domain."""
        x = self.units.convert_units(1.0, "m")
        y = self.units.convert_units(1.0, "m")
        self._domain = self.nd_rect_domain(x, y)


class ModelEnergyDecay(
    BoundaryConditionsEnergyDecayAnalysis,
    SourceValuesEnergyDecayAnalysis,
    Geometry,
    DynamicMomentumBalanceABCLinear,
):
    """Model class setup for the energy decay analysis."""

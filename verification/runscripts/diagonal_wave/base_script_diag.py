"""This file contains a unit test of the absorbing boundary conditions implemented.

We have a setup of a sine-wave travelling at an angle from the lower left corner to the
upper left corner. 

Boundary conditions are: 
* West and south: Dirichlet. Time dependent, diagonal sine wave. 
* North and east: Absorbing. 

"""

import porepy as pp
import numpy as np

import sys

sys.path.append("../../../")

from models import MomentumBalanceABC

from utils import get_boundary_cells


class RotationAngle:
    @property
    def rotation_angle(self) -> float:
        return np.pi / 4


class BoundaryConditionsUnitTest:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        # Approximating the time derivative in the BCs and rewriting the BCs on "Robin
        # form" gives need for Robin weights.

        # These two lines provide an array with 2 components. The first component is a
        # 2d array with ones in the first row and zeros in the second. The second
        # component is also a 2d array, but now the first row is zeros and the second
        # row is ones.
        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        value = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F")

        bounds = self.domain_boundary_sides(sd)

        value[1][1][bounds.east] *= self.robin_weight_value(
            direction="shear", side="east"
        )
        value[0][0][bounds.east] *= self.robin_weight_value(
            direction="tensile", side="east"
        )

        value[0][0][bounds.north] *= self.robin_weight_value(
            direction="shear", side="north"
        )

        value[1][1][bounds.north] *= self.robin_weight_value(
            direction="tensile", side="north"
        )

        # Choosing type of boundary condition for the different domain sides.
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "rob",
        )
        bc.is_rob[:, bounds.west + bounds.south] = False

        bc.is_dir[:, bounds.west + bounds.south] = True

        bc.robin_weight = value
        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        x = bg.cell_centers[0, :]
        y = bg.cell_centers[1, :]
        t = self.time_manager.time

        alpha = self.rotation_angle
        cp = self.primary_wave_speed

        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        values = np.reshape(values, (self.nd, bg.num_cells), "F")

        values[0][bounds.west] += np.sin(
            np.ones(len(x[bounds.west])) * t - 1 / (cp) * y[bounds.west] * np.sin(alpha)
        )
        values[1][bounds.west] += np.sin(
            np.ones(len(y[bounds.west])) * t - 1 / (cp) * y[bounds.west] * np.sin(alpha)
        )
        values[0][bounds.south] += np.sin(
            np.ones(len(x[bounds.south])) * t
            - 1 / (cp) * x[bounds.south] * np.cos(alpha)
        )
        values[1][bounds.south] += np.sin(
            np.ones(len(y[bounds.south])) * t
            - 1 / (cp) * x[bounds.south] * np.cos(alpha)
        )

        return values.ravel("F")

    def bc_values_robin(self, bg: pp.BoundaryGrid) -> np.ndarray:
        face_areas = bg.cell_volumes
        data = self.mdg.boundary_grid_data(bg)

        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)
        if self.time_manager.time_index > 1:
            sd = bg.parent
            displacement_boundary_operator = self.boundary_displacement([sd])
            displacement_values = displacement_boundary_operator.evaluate(
                self.equation_system
            ).val

            displacement_values = bg.projection(self.nd) @ displacement_values

        elif self.time_manager.time_index == 1:
            displacement_values = pp.get_solution_values(
                name="bc_robin", data=data, time_step_index=0
            )

        elif self.time_manager.time_index == 0:
            return self.initial_condition_bc(bg)

        displacement_values = np.reshape(
            displacement_values, (self.nd, bg.num_cells), "F"
        )

        # Values for the absorbing boundaries
        # Scaling with face area is crucial for the ABCs.This is due to the way we
        # handle the time derivative.
        # East:
        values[1][bounds.east] += (
            self.robin_weight_value(direction="shear", side="east")
            * displacement_values[1][bounds.east]
        ) * face_areas[bounds.east]
        values[0][bounds.east] += (
            self.robin_weight_value(direction="tensile", side="east")
            * displacement_values[0][bounds.east]
        ) * face_areas[bounds.east]

        # North:
        values[0][bounds.north] += (
            self.robin_weight_value(direction="shear", side="north")
            * displacement_values[0][bounds.north]
        ) * face_areas[bounds.north]
        values[1][bounds.north] += (
            self.robin_weight_value(direction="tensile", side="north")
            * displacement_values[1][bounds.north]
        ) * face_areas[bounds.north]
        return values.ravel("F")


class InitialConditionSourceTermUnitTest:
    def initial_condition_bc(self, bg):
        """Assigning initial bc values."""
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

        bc_vals = bc_vals.ravel("F")

        bc_vals = bg.projection(self.nd) @ bc_vals.ravel("F")
        return bc_vals

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


class BaseScriptModel(
    RotationAngle,
    BoundaryConditionsUnitTest,
    InitialConditionSourceTermUnitTest,
    MomentumBalanceABC,
):
    ...

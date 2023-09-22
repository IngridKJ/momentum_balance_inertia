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

    def time_dependent_bc_values_mechanics(
        self, subdomains: list[pp.Grid]
    ) -> np.ndarray:
        """Method for assigning the time dependent bc values.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        assert len(subdomains) == 1
        sd = subdomains[0]
        face_areas = sd.face_areas
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time
        cp = self.primary_wave_speed

        values = np.zeros((self.nd, sd.num_faces))
        bounds = self.domain_boundary_sides(sd)

        if self.time_manager.time_index > 1:
            # "Get face displacement": Create them using bound_displacement_face/
            # bound_displacement_cell from second timestep and ongoing.
            displacement_boundary_operator = self.boundary_displacement([sd])
            displacement_values = displacement_boundary_operator.evaluate(
                self.equation_system
            ).val

        else:
            # On first timestep, initial values are fetched from the data dictionary.
            # These initial values are assigned in the initial_condition function.
            displacement_values = pp.get_solution_values(
                name=self.bc_values_mechanics_key, data=data, time_step_index=0
            )

        # Note to self: The "F" here is crucial.
        displacement_values = np.reshape(
            displacement_values, (self.nd, sd.num_faces), "F"
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

        # Values for the western and southern side sine wave
        x = sd.face_centers[0, :]
        y = sd.face_centers[1, :]
        alpha = self.rotation_angle

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


class InitialConditionUnitTest:
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


class ExportErrors:
    def data_to_export(self):
        """Define the data to export to vtu.

        Returns:
            list: List of tuples containing the subdomain, variable name,
            and values to export.

        """
        data = super().data_to_export()
        for sd in self.mdg.subdomains(dim=self.nd):
            x = sd.cell_centers[0, :]
            y = sd.cell_centers[1, :]
            t = self.time_manager.time
            cp = self.primary_wave_speed
            alpha = self.rotation_angle

            d = self.mdg.subdomain_data(sd)
            disp_vals = pp.get_solution_values(name="u", data=d, time_step_index=0)

            u_h = np.reshape(disp_vals, (self.nd, sd.num_cells), "F")

            u_e = np.array(
                [
                    np.sin(t - (x * np.cos(alpha) + y * np.sin(alpha)) / (cp)),
                    np.sin(t - (x * np.cos(alpha) + y * np.sin(alpha)) / (cp)),
                ]
            )

            du = u_e - u_h

            relative_l2_error = pp.error_computation.l2_error(
                grid=sd,
                true_array=u_e,
                approx_array=u_h,
                is_scalar=True,
                is_cc=True,
                relative=True,
            )

            data.append((sd, "absolute_error", du))
            data.append((sd, "analytical_solution", u_e))

            with open(self.filename, "a") as file:
                file.write(str(relative_l2_error))
        return data


class BaseScriptModel(
    RotationAngle,
    BoundaryConditionsUnitTest,
    InitialConditionUnitTest,
    ExportErrors,
    MomentumBalanceABC,
):
    ...

"""This file contains a unit test of the absorbing boundary conditions implemented.

We have a quasi 1D-setup of a sine-wave travelling perpendicular to the east boundary.
The test shows that the eastern boundary absorbs the traveling sine wave. To mimic a 1D
setup, all off-diagonal terms of the fourth order stiffness tensor, C, are removed.

Boundary conditions are: 
* North and south: Zero Neumann. 
* East and west: Low order absorbing boundary condition.

"""

import porepy as pp
import numpy as np


import sys

sys.path.append("../")

from models import MomentumBalanceABC


class BoundaryConditionsUnitTest:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        value = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F")

        bounds = self.domain_boundary_sides(sd)

        value[1][1][bounds.east] *= self.robin_weight_value(
            direction="shear", side="east"
        )
        value[0][0][bounds.east] *= self.robin_weight_value(
            direction="tensile", side="east"
        )

        value[1][1][bounds.west] *= self.robin_weight_value(
            direction="shear", side="west"
        )
        value[0][0][bounds.west] *= self.robin_weight_value(
            direction="tensile", side="west"
        )

        # Choosing type of boundary condition for the different domain sides.
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "rob",
        )
        bc.is_rob[:, bounds.north + bounds.south] = False

        bc.is_neu[:, bounds.north + bounds.south] = True

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

        # Zero Neumann on top - Waves are just allowed to slide alongside the
        # boundaries.
        values[0][bounds.north] = np.zeros(len(displacement_values[0][bounds.north]))
        values[1][bounds.north] = np.zeros(len(displacement_values[1][bounds.north]))

        values[0][bounds.south] = np.zeros(len(displacement_values[0][bounds.south]))
        values[1][bounds.south] = np.zeros(len(displacement_values[1][bounds.south]))

        # Values for the absorbing boundary
        # Scaling with face area is crucial for the ABCs.This is due to the way we
        # handle the time derivative.
        values[1][bounds.east] += (
            self.robin_weight_value(direction="shear", side="east")
            * displacement_values[1][bounds.east]
        ) * face_areas[bounds.east]
        values[0][bounds.east] += (
            self.robin_weight_value(direction="tensile", side="east")
            * displacement_values[0][bounds.east]
        ) * face_areas[bounds.east]

        values[1][bounds.west] += (
            self.robin_weight_value(direction="shear", side="west")
            * displacement_values[1][bounds.west]
        ) * face_areas[bounds.west]
        values[0][bounds.west] += (
            self.robin_weight_value(direction="tensile", side="west")
            * displacement_values[0][bounds.west]
        ) * face_areas[bounds.west]

        return values.ravel("F")


class ConstitutiveLawUnitTest:
    def stiffness_tensor(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa].

        Parameters:
            subdomain: Subdomain where the stiffness tensor is defined.

        Returns:
            Cell-wise stiffness tensor in SI units.

        """
        lmbda = self.solid.lame_lambda() * np.ones(subdomain.num_cells)
        mu = self.solid.shear_modulus() * np.ones(subdomain.num_cells)
        return FourthOrderTensorUnitTest(mu, lmbda)


class FourthOrderTensorUnitTest(object):
    """Cell-wise representation of fourth order tensor, represented by (3^2, 3^2 ,Nc)-array, where Nc denotes the number of cells, i.e. the tensor values are stored discretely.

    For each cell, there are dim^4 degrees of freedom, stored in a 3^2 * 3^2 matrix.

    The only constructor available for the moment is based on the Lame parameters, e.g.
    using two degrees of freedom.

    Attributes:
        values: dimensions (3^2, 3^2, nc), cell-wise representation of
            the stiffness matrix.
        lmbda (np.ndarray): Nc array of first Lame parameter
        mu (np.ndarray): Nc array of second Lame parameter

    """

    def __init__(self, mu: np.ndarray, lmbda: np.ndarray):
        """Constructor for fourth order tensor on Lame-parameter form

        Parameters:
            mu: Nc array of shear modulus (second lame parameter).
            lmbda: Nc array of first lame parameter.

        Raises:
            ValueError if mu or lmbda are not 1d arrays.
            ValueError if the lengths of mu and lmbda are not matching.
        """

        if not isinstance(mu, np.ndarray):
            raise ValueError("Input mu should be a numpy array")
        if not isinstance(lmbda, np.ndarray):
            raise ValueError("Input lmbda should be a numpy array")
        if not mu.ndim == 1:
            raise ValueError("mu should be 1-D")
        if not lmbda.ndim == 1:
            raise ValueError("Lmbda should be 1-D")
        if mu.size != lmbda.size:
            raise ValueError("Mu and lmbda should have the same length")

        # Save lmbda and mu, can be useful to have in some cases
        self.lmbda = lmbda
        self.mu = mu

        # Basis for the contributions of mu, lmbda and phi is hard-coded
        mu_mat = np.array(
            [
                [2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2],
            ]
        )
        lmbda_mat = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Expand dimensions to prepare for cell-wise representation
        mu_mat = mu_mat[:, :, np.newaxis]
        lmbda_mat = lmbda_mat[:, :, np.newaxis]

        c = mu_mat * mu + lmbda_mat * lmbda
        self.values = c

    def copy(self) -> "FourthOrderTensorUnitTest":
        """`
        Define a deep copy of the tensor.

        Returns:
            FourthOrderTensor: New tensor with identical fields, but separate
                arrays (in the memory sense).
        """
        C = FourthOrderTensorUnitTest(mu=self.mu, lmbda=self.lmbda)
        C.values = self.values.copy()
        return C


class CustomEta:
    def set_discretization_parameters(self) -> None:
        """Set discretization parameters for the simulation.

        Sets eta = 1/3 on all faces if it is a simplex grid.

        """

        super().set_discretization_parameters()
        if self.params["grid_type"] == "simplex":
            num_subfaces = 0
            for sd, data in self.mdg.subdomains(return_data=True):
                subcell_topology = pp.fvutils.SubcellTopology(sd)
                num_subfaces += subcell_topology.num_subfno
                eta_values = np.ones(num_subfaces) * 1 / 3
                if sd.dim == self.nd:
                    pp.initialize_data(
                        sd,
                        data,
                        self.stress_keyword,
                        {
                            "mpsa_eta": eta_values,
                        },
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
    BoundaryConditionsUnitTest,
    ConstitutiveLawUnitTest,
    CustomEta,
    ExportErrors,
    MomentumBalanceABC,
):
    ...

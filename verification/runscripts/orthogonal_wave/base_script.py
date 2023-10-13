"""This file contains a unit test of the absorbing boundary conditions implemented.

We have a quasi 1D-setup of a sine-wave travelling perpendicular to the east boundary.
The test shows that the eastern boundary absorbs the traveling sine wave. To mimic a 1D
setup, all off-diagonal terms of the fourth order stiffness tensor, C, are removed.

Boundary conditions are: 
* West: Dirichlet. Time dependent towards-right-travelling sine wave. 
* North and south: Zero Neumann. 
* East: Low order absorbing boundary condition.

"""

import porepy as pp
import numpy as np

from plot_l2_error import plot_the_error

import sys

sys.path.append("../../../")

from models import MomentumBalanceABC


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

        # Choosing type of boundary condition for the different domain sides.
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "rob",
        )
        bc.is_rob[:, bounds.west + bounds.north + bounds.south] = False

        bc.is_dir[:, bounds.west] = True
        bc.is_neu[:, bounds.north + bounds.south] = True

        bc.robin_weight = value
        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        t = self.time_manager.time

        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        displacement_values = np.reshape(values, (self.nd, bg.num_cells), "F")

        values[0][bounds.west] += np.ones(
            len(displacement_values[0][bounds.west])
        ) * np.sin(t)
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

        values[1][bounds.east] += (
            self.robin_weight_value(direction="shear", side="east")
            * displacement_values[1][bounds.east]
        ) * face_areas[bounds.east]

        values[0][bounds.east] += (
            self.robin_weight_value(direction="tensile", side="east")
            * displacement_values[0][bounds.east]
        ) * face_areas[bounds.east]
        return values.ravel("F")

    def initial_condition_bc(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros((self.nd, bg.num_cells))


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
            t = self.time_manager.time
            cp = self.primary_wave_speed

            d = self.mdg.subdomain_data(sd)
            disp_vals = pp.get_solution_values(name="u", data=d, time_step_index=0)

            u_h = np.reshape(disp_vals, (self.nd, sd.num_cells), "F")[0]

            u_e = np.sin(t - x / cp)

            du = u_e - u_h

            relative_l2_error = pp.error_computation.l2_error(
                grid=sd,
                true_array=u_e,
                approx_array=u_h,
                is_scalar=True,
                is_cc=True,
                relative=True,
            )

            data.append((sd, "diff_anal_num", du))
            data.append((sd, "analytical_solution", u_e))

            x_max = self.domain.bounding_box["xmax"]
            # Don't bother collecting the errors before the wave has traveled the
            # entire domain.
            if self.params.get("write_errors", False):
                if self.time_manager.time >= x_max / cp:
                    with open("errors.txt", "a") as file:
                        file.write("," + str(relative_l2_error))
                if (
                    int(self.time_manager.time_final / self.time_manager.dt)
                ) == self.time_manager.time_index:
                    plot_the_error("errors.txt", write_stats=True)

        return data


class BaseScriptModel(
    BoundaryConditionsUnitTest,
    ConstitutiveLawUnitTest,
    # ExportErrors,
    MomentumBalanceABC,
):
    ...

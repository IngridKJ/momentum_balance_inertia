import sys

import numpy as np
import porepy as pp
import sympy as sym

sys.path.append("../")

from models import DynamicMomentumBalanceABC2
from utils import u_v_a_wrap


class BoundaryConditionsUnitTest:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for the absorbing boundary condition model class.

        Assigns the following boundary condition types:
            * North and south: Neumann
            * West: Dirichlet
            * East: Robin

        Parameters:
            sd: The subdomain whose bc type is assigned.

        Return:
            The boundary condition object.

        """
        # Fetch boundary sides and assign type of boundary condition for the different
        # sides
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.east + bounds.west, "rob")

        # Only the eastern boundary will be Robin (absorbing)
        bc.is_rob[:, bounds.west] = False

        # Western boundary is Dirichlet
        bc.is_dir[:, bounds.west] = True

        # Calling helper function for assigning the Robin weight
        self.assign_robin_weight(sd=sd, bc=bc)
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Method for assigning Neumann and Robin boundary condition values.

        Specifically, for the Robin values, this method assigns the values corresponding
        to ABC_2, namely a second order approximation to u_t in:

            sigma * n + alpha * u_t = G

        Robin/ABC_2 is employed for the east and west boundary. Zero Neumann conditions
        are assigned for the north and south boundary.

        Parameters:
            boundary_grid: The boundary grids on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        # !!! Check what is the deal with this guy. I think it is a thing for fractured
        # media.
        if boundary_grid.dim != (self.nd - 1):
            return np.array([])

        data = self.mdg.boundary_grid_data(boundary_grid)
        sd = boundary_grid.parent
        boundary_faces = sd.get_boundary_faces()
        name = "boundary_displacement_values"

        if self.time_manager.time_index == 0:
            return self.initial_condition_bc(boundary_grid)

        displacement_values_0 = pp.get_solution_values(
            name=name, data=data, time_step_index=0
        )
        displacement_values_1 = pp.get_solution_values(
            name=name, data=data, time_step_index=1
        )

        # According to the expression for ABC_2 we have a coefficient 2 in front of the
        # values u_(n-1) and -0.5 in front of u_(n-2):
        displacement_values = 2 * displacement_values_0 - 0.5 * displacement_values_1

        # Transposing and reshaping displacement values to prepare for broadcasting
        displacement_values = displacement_values.reshape(
            boundary_grid.num_cells, self.nd, 1
        )

        # Assembling the vector representing the RHS of the Robin conditions
        total_coefficient_matrix = self.total_coefficient_matrix(sd=sd)
        robin_rhs = np.matmul(total_coefficient_matrix, displacement_values).squeeze(-1)
        robin_rhs *= sd.face_areas[boundary_faces][:, None]
        robin_rhs = robin_rhs.T

        boundary_sides = self.domain_boundary_sides(sd)
        inds_north = np.where(boundary_sides.north)[0]
        inds_south = np.where(boundary_sides.south)[0]

        inds_north = np.where(np.isin(boundary_faces, inds_north))[0]
        inds_south = np.where(np.isin(boundary_faces, inds_south))[0]

        robin_rhs[:, inds_north] *= 0
        robin_rhs[:, inds_south] *= 0

        return robin_rhs.ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Method for setting Dirichlet boundary values.

        Sets a time dependent sine condition in the x-direction of the western boundary.

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the displacement boundary values at time t.

        """
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)
        t = self.time_manager.time

        # Time dependent sine Dirichlet condition
        values[0][bounds.west] += np.ones(len(values[0][bounds.west])) * np.sin(t)

        return values.ravel("F")

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
            values=vals_0,
            data=data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="boundary_displacement_values",
            values=vals_1,
            data=data,
            time_step_index=1,
        )
        return vals_0

    def initial_condition_value_function(
        self, bg: pp.BoundaryGrid, t: float
    ) -> np.ndarray:
        """Initial values for the ABC boundaries.

        In the quasi-1d test we have to assign initial values to the east and west
        boundaries (ABC boundaries).

        """
        sd = bg.parent

        x = sd.face_centers[0, :]
        y = sd.face_centers[1, :]

        boundary_sides = self.domain_boundary_sides(sd)
        inds_east = np.where(boundary_sides.east)[0]
        inds_west = np.where(boundary_sides.west)[0]

        bc_vals = np.zeros((sd.dim, sd.num_faces))

        displacement_function = u_v_a_wrap(model=self)

        # East
        bc_vals[0, :][inds_east] = displacement_function[0](
            x[inds_east], y[inds_east], t
        )

        # West
        bc_vals[0, :][inds_west] = displacement_function[0](
            x[inds_west], y[inds_west], t
        )

        bc_vals = bg.projection(self.nd) @ bc_vals.ravel("F")

        return bc_vals


class ConstitutiveLawsAndSource:
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

    def elastic_force(self, sd, sigma_total, time: float) -> np.ndarray:
        """Evaluate exact elastic force [N] at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_faces, )`` containing the exact elastic force
            at the face centers for the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get cell centers and face normals
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression
        sigma_total_fun = [
            [
                sym.lambdify((x, y, t), sigma_total[0][0], "numpy"),
                sym.lambdify((x, y, t), sigma_total[0][1], "numpy"),
            ],
            [
                sym.lambdify((x, y, t), sigma_total[1][0], "numpy"),
                sym.lambdify((x, y, t), sigma_total[1][1], "numpy"),
            ],
        ]

        # Face-centered elastic force
        force_total_fc: list[np.ndarray] = [
            # (sigma_xx * n_x + sigma_xy * n_y) * face_area
            sigma_total_fun[0][0](fc[0], fc[1], time) * fn[0]
            # Eliminate y-direction, as the analytical force is only in x-direction.
            + 0 * sigma_total_fun[0][1](fc[0], fc[1], time) * fn[1],
            # (sigma_yx * n_x + sigma_yy * n_y) * face_area
            sigma_total_fun[1][0](fc[0], fc[1], time) * fn[0]
            # Eliminate y-direction, as the analytical force is only in x-direction.
            + 0 * sigma_total_fun[1][1](fc[0], fc[1], time) * fn[1],
        ]

        # Flatten array
        force_total_flat: np.ndarray = np.asarray(force_total_fc).ravel("F")
        return force_total_flat

    def source_values(self, f, sd, t) -> np.ndarray:
        vals = np.zeros((self.nd, sd.num_cells))
        return vals.ravel("F")


class FourthOrderTensorUnitTest(object):
    """Adapted from the FourthOrderTensor in PorePy.

    Cell-wise representation of a fourth order tensor whose off-diagonal components
    are discarded. It is represented by (3^2, 3^2 ,Nc) -array, where Nc denotes the
    number of cells, i.e. the tensor values are stored discretely.

    Attributes:
        values: dimensions (3^2, 3^2, nc), cell-wise representation of
            the stiffness matrix.
        lmbda (np.ndarray): Nc array of first Lame parameter
        mu (np.ndarray): Nc array of second Lame parameter

    """

    def __init__(self, mu: np.ndarray, lmbda: np.ndarray):
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
        C = FourthOrderTensorUnitTest(mu=self.mu, lmbda=self.lmbda)
        C.values = self.values.copy()
        return C


class ABC2Model(
    BoundaryConditionsUnitTest,
    ConstitutiveLawsAndSource,
    DynamicMomentumBalanceABC2,
): ...

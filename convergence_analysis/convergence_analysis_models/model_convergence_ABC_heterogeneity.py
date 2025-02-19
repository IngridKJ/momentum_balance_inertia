"""Model class setup for the convergence analysis of MPSA-Newmark with absorbing
boundaries.

All off-diagonal/shear components of the stiffness tensor are discarded.

"""

import sys

import numpy as np
import porepy as pp

sys.path.append("../../")

from models import DynamicMomentumBalanceABCLinear
from utils import u_v_a_wrap
import sympy as sym


class BoundaryConditionsUnitTest:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Method for assigning boundary condition type.

        Assigns the following boundary condition types:
            * North and south: Roller boundary (Neumann in x-direction and Dirichlet in
              y-direction)
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
        bc = pp.BoundaryConditionVectorial(sd, bounds.all_bf, "dir")

        # # East side: Absorbing
        # bc.is_dir[:, bounds.east] = False
        # bc.is_rob[:, bounds.east] = True

        # North side: Roller
        bc.is_dir[0, bounds.north + bounds.south] = False
        bc.is_neu[0, bounds.north + bounds.south] = True

        # Calling helper function for assigning the Robin weight
        self.assign_robin_weight(sd=sd, bc=bc)
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Method for assigning Neumann and Robin boundary condition values.

        Specifically, for the Robin values, this method assigns the values corresponding
        to absorbing boundary conditions with a second order approximation of u_t in:

            sigma * n + alpha * u_t = G

        Robin/Absorbing boundaries are employed for the east and west boundary. Zero
        Neumann conditions are assigned for the north and south boundary.

        Parameters:
            boundary_grid: The boundary grids on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
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

        # According to the expression for the absorbing boundaries we have a coefficient
        # 2 in front of the values u_(n-1) and -0.5 in front of u_(n-2):
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
        Zero elsewhere.

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the displacement boundary values at time t.

        """
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)
        t = self.time_manager.time

        # Time dependent sine Dirichlet condition
        bc_left, bc_right = self.heterogeneous_analytical_solution()
        values[0][bounds.west] += np.ones(len(values[0][bounds.west])) * bc_left[0](
            0.0, t
        )
        values[0][bounds.east] += np.ones(len(values[0][bounds.west])) * bc_right[0](
            1.0, t
        )

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
        """Initial values for the absorbing boundaries.

        In the quasi-1d test we have to assign initial values to the east and west
        boundaries (absorbing boundaries).

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


class InitialConditions:
    def heterogeneous_analytical_solution(self, return_dt=False, return_ddt=False):
        """Compute the analytical solution and its time derivatives."""
        x, t = sym.symbols("x t")

        L = self.params.get("L", 0.5)
        cp = self.primary_wave_speed(is_scalar=False)
        c_A, c_C = cp[0], cp[-1]

        u_left = sym.sin(t - (x - L) / c_A) + (c_C - c_A) / (c_C + c_A) * sym.sin(
            t - (x - L) / c_A
        )

        u_right = (2 * c_C) / (c_C + c_A) * sym.sin(t - (x - L) / c_C)

        # Compute derivatives based on function arguments
        if return_dt:
            u_left, u_right = sym.diff(u_left, t), sym.diff(u_right, t)
        elif return_ddt:
            u_left, u_right = sym.diff(u_left, (t, 2)), sym.diff(u_right, (t, 2))

        # Lambdify the results for numerical evaluation
        return [sym.lambdify((x, t), u_left, "numpy"), 0], [
            sym.lambdify((x, t), u_right, "numpy"),
            0,
        ]

    def _compute_initial_condition(self, return_dt=False, return_ddt=False):
        """Helper function to compute displacement, velocity, or acceleration."""
        sd = self.mdg.subdomains()[0]
        x = sd.cell_centers[0, :]
        t = self.time_manager.time

        L = self.params.get("L", 0.5)
        layer_A = x < L
        layer_C = x > L

        vals = np.zeros((self.nd, sd.num_cells))

        left_solution, right_solution = self.heterogeneous_analytical_solution(
            return_dt=return_dt, return_ddt=return_ddt
        )

        vals[0, layer_A] = left_solution[0](x[layer_A], t)
        vals[0, layer_C] = right_solution[0](x[layer_C], t)
        return vals.ravel("F")

    def initial_displacement(self, dofs):
        """Compute the initial displacement."""
        return self._compute_initial_condition()

    def initial_velocity(self, dofs):
        """Compute the initial velocity."""
        return self._compute_initial_condition(return_dt=True)

    def initial_acceleration(self, dofs):
        """Compute the initial acceleration."""
        return self._compute_initial_condition(return_ddt=True)


class ConstitutiveLawsAndSource:
    def elastic_force_(self, sd, sigma_total, time: float) -> np.ndarray:
        """"""

    def evaluate_mechanics_source(self, f: list, sd: pp.Grid, t: float) -> np.ndarray:
        vals = np.zeros((self.nd, sd.num_cells))
        return vals.ravel("F")

    def vector_valued_mu_lambda(self):
        """Setting a layered medium."""
        subdomain = self.mdg.subdomains(dim=self.nd)[0]
        x = subdomain.cell_centers[0, :]

        lmbda1 = self.solid.lame_lambda
        mu1 = self.solid.shear_modulus

        lmbda2 = self.solid.lame_lambda * self.params.get("heterogeneity_factor", 0.5)
        mu2 = self.solid.shear_modulus * self.params.get("heterogeneity_factor", 0.5)

        lmbda_vec = np.ones(subdomain.num_cells)
        mu_vec = np.ones(subdomain.num_cells)

        layer_A = x < self.params.get("L", 0.5)
        layer_C = x > self.params.get("L", 0.5)

        lmbda_vec[layer_A] *= lmbda1
        mu_vec[layer_A] *= mu1

        lmbda_vec[layer_C] *= lmbda2
        mu_vec[layer_C] *= mu2

        self.mu_vector = mu_vec
        self.lambda_vector = lmbda_vec


class ABCModel(
    BoundaryConditionsUnitTest,
    ConstitutiveLawsAndSource,
    InitialConditions,
    DynamicMomentumBalanceABCLinear,
): ...

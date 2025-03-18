"""Model class setup for the convergence analysis of MPSA-Newmark with absorbing
boundaries."""

import sys

import numpy as np
import porepy as pp
import sympy as sym

sys.path.append("../../")

from typing import Callable, List, Tuple, Union

from porepy.applications.convergence_analysis import ConvergenceAnalysis

from models import DynamicMomentumBalanceABCLinear
from utils.utility_functions import (
    create_stiffness_tensor_basis,
    use_constraints_for_inner_domain_cells,
)


class HeterogeneityProperties:
    @property
    def heterogeneity_location(self) -> float:
        """Location on the x-axis of the vertical split of the simulation domain."""
        return self.params.get("heterogeneity_location", 0.5)

    @property
    def heterogeneity_factor(self) -> float:
        """Factor determining how strong heterogeneity we have in a domain.

        Determines the factor that the material parameters to the left of
        self.heterogeneity_location should differ from the material parameters to the
        right of self.heterogeneity_location.

        """
        return self.params.get("heterogeneity_factor", 1.0)


class Geometry:
    """Mixin for setting the geometry used in the convergene analysis.

    Here we set a square unit domain. Additionally, the methods set_polygons() and
    set_fractures() is set. Those methods are used (in other methods and classes) to
    define different stiffness tensors in different regions in the simulation domain.

    """

    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}
        box.update({"ymin": 0, "ymax": y})
        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def set_polygons(self):
        """Defining polygons around the region where a different stiffness tensor is
        to be defined."""
        if type(self.heterogeneity_location) is list:
            L = self.heterogeneity_location[0]
            W = self.heterogeneity_location[1]
        else:
            L = self.heterogeneity_location
            W = self.domain.bounding_box["xmax"]

        H = self.domain.bounding_box["ymax"]
        west = np.array([[L, L], [0.0, H]])
        north = np.array([[L, W], [H, H]])
        east = np.array([[W, W], [H, 0.0]])
        south = np.array([[W, L], [0.0, 0.0]])
        return west, north, east, south

    def set_fractures(self) -> None:
        """Setting constraints for meshing.

        The constraints, defined by help of the set_polygons()-method, are used for the
        meshing such that the grid conforms to the intersection between two regions in
        the simulation domain. This necessitates the usage of the meshing kwargs
        "constraints". This can be done e.g. in the following way via the params
        dictionary:

            params = {"meshing_kwargs": {"constraints": [0, 1, 2, 3]}}.

        If not, the objects being defined and set to self._fractures here will in fact
        be fractures, and not only meshing constraints.

        """
        west, north, east, south = self.set_polygons()

        self._fractures = [
            pp.LineFracture(west),
            pp.LineFracture(north),
            pp.LineFracture(east),
            pp.LineFracture(south),
        ]


class BoundaryConditionsAndSource:
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

        # East side: Absorbing
        bc.is_dir[:, bounds.east] = False
        bc.is_rob[:, bounds.east] = True

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

        Robin/Absorbing boundaries are employed for the east boundary. Zero Neumann
        values are assigned for the north and south boundary.

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

        # Values corresponding to the right-hand side of the absorbing boundaries for
        # all boundary sides.
        robin_rhs = robin_rhs.T

        # As the same method (bc_values_stress) is used for assigning Neumann and Robin
        # (Absorbing) boundaries, we need to set zero-values where we have assigned the
        # Neumann boundary type.
        boundary_sides = self.domain_boundary_sides(sd)
        for direction in ["north", "south"]:
            inds = np.where(getattr(boundary_sides, direction))[0]
            inds = np.where(np.isin(boundary_faces, inds))[0]
            robin_rhs[:, inds] *= 0

        return robin_rhs.ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Method for setting Dirichlet boundary values.

        Sets a time dependent condition in the x-direction of the western boundary. Zero
        elsewhere. The boundary value function is determined by the method
        heterogeneous_analytical_solution(), which is the known analytical solution for
        a 1D wave travelling in an inhomogeneous domain.

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the displacement boundary values at time t.

        """
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)
        t = self.time_manager.time

        xmin = self.domain.bounding_box["xmin"]

        bc_left, _ = self.heterogeneous_analytical_solution()
        values[0][bounds.west] += np.ones(len(values[0][bounds.west])) * bc_left[0](
            xmin, t
        )

        return values.ravel("F")

    def initial_condition_value_function(
        self, bg: pp.BoundaryGrid, t: float
    ) -> np.ndarray:
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

        boundary_sides = self.domain_boundary_sides(sd)
        inds_east = np.where(boundary_sides.east)[0]

        bc_vals = np.zeros((sd.dim, sd.num_faces))

        _, displacement_function_left = self.heterogeneous_analytical_solution()

        # East
        bc_vals[0, :][inds_east] = displacement_function_left[0](x[inds_east], t)

        # Mapping the face-wise values of the parent grid onto the boundary grid.
        bc_vals = bg.projection(self.nd) @ bc_vals.ravel("F")
        return bc_vals

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

    def evaluate_mechanics_source(self, f: list, sd: pp.Grid, t: float) -> np.ndarray:
        vals = np.zeros((self.nd, sd.num_cells))
        return vals.ravel("F")


class InitialConditions:
    def heterogeneous_analytical_solution(
        self, return_dt: bool = False, return_ddt: bool = False, lambdify: bool = True
    ) -> Union[
        Tuple[List[Callable], List[Callable]],  # When lambdify=True
        Tuple[sym.Symbol, sym.Symbol, sym.Expr, sym.Expr],  # When lambdify=False
    ]:
        """Analytical solutions for wave propagation in inhomogeneous media.

        Representation of the displacement (and, upon request, the velocity and
        acceleration) wave in an inhomogeneous media.

        Parameters:
            return_dt: If True, the first time-derivative (velocity) is returned.
            return_ddt: If True, the second time-derivative (acceleration) is returned.
            lambdify: If True, the lambdified expression is returned. The lambdified
                expression can be used to evaluate the function value. This method is
                only used with lambdify=False when the displacement expressions here are
                used for constructing the exact heterogeneous force.

        Returns:
            Either the callable functions for displacement/velocity/acceleration, or the
            sympy expression of the displacement.

        """
        x, t = sym.symbols("x t")

        L = self.heterogeneity_location
        cp = self.primary_wave_speed(is_scalar=False)

        heterogeneity_factor = self.heterogeneity_factor
        if heterogeneity_factor >= 1.0:
            left_speed = min(cp)
            right_speed = max(cp)
        else:
            left_speed = max(cp)
            right_speed = min(cp)

        u_left = sym.sin(t - (x - L) / left_speed) - (right_speed - left_speed) / (
            right_speed + left_speed
        ) * sym.sin(t + (x - L) / left_speed)
        u_right = (
            (2 * left_speed)
            / (right_speed + left_speed)
            * sym.sin(t - (x - L) / right_speed)
        )

        # Compute derivatives based on function arguments
        if return_dt:
            u_left, u_right = sym.diff(u_left, t), sym.diff(u_right, t)
        elif return_ddt:
            u_left, u_right = sym.diff(u_left, t, 2), sym.diff(u_right, t, 2)

        if lambdify:
            return [sym.lambdify((x, t), u_left, "numpy"), 0], [
                sym.lambdify((x, t), u_right, "numpy"),
                0,
            ]
        else:
            return x, t, u_left, u_right

    def _compute_initial_condition(
        self, return_dt: bool = False, return_ddt: bool = False
    ) -> np.ndarray:
        """Helper function to compute displacement, velocity or acceleration."""
        sd = self.mdg.subdomains()[0]
        x = sd.cell_centers[0, :]
        t = self.time_manager.time

        L = self.heterogeneity_location
        left_layer = x <= L
        right_layer = x > L

        vals = np.zeros((self.nd, sd.num_cells))

        left_solution, right_solution = self.heterogeneous_analytical_solution(
            return_dt=return_dt, return_ddt=return_ddt
        )

        vals[0, left_layer] = left_solution[0](x[left_layer], t)
        vals[0, right_layer] = right_solution[0](x[right_layer], t)
        return vals.ravel("F")

    def initial_displacement(self, dofs) -> np.ndarray:
        """Compute the initial displacement."""
        return self._compute_initial_condition()

    def initial_velocity(self, dofs) -> np.ndarray:
        """Compute the initial velocity."""
        return self._compute_initial_condition(return_dt=True)

    def initial_acceleration(self, dofs) -> np.ndarray:
        """Compute the initial acceleration."""
        return self._compute_initial_condition(return_ddt=True)


class ExactHeterogeneousSigmaAndForce:
    def exact_heterogeneous_sigma(
        self, u: sym.Expr, lam: float, mu: float, x: sym.Symbol
    ) -> list:
        """Representation of the exact stress tensor given a displacement field.

        Parameters:
            u: The sympy representation of the exact displacement.
            lam: The first Lamé parameter.
            mu: The second Lamé parameter, also called shear modulus.
            x: The sympy representation of the x-coordinate.

        Returns:
            A list which represents the sympy expression of the exact stress tensor.

        """
        y = sym.symbols("y")

        u = [u, 0]
        # Exact gradient of u and transpose of gradient of u
        grad_u = [
            [sym.diff(u[0], x), sym.diff(u[0], y)],
            [sym.diff(u[1], x), sym.diff(u[1], y)],
        ]

        grad_u_T = [[grad_u[0][0], grad_u[1][0]], [grad_u[0][1], grad_u[1][1]]]

        # Trace of gradient of u, in the linear algebra sense
        trace_grad_u = grad_u[0][0] + grad_u[1][1]

        # Exact strain (\epsilon(u))
        strain = 0.5 * np.array(
            [
                [grad_u[0][0] + grad_u_T[0][0], grad_u[0][1] + grad_u_T[0][1]],
                [grad_u[1][0] + grad_u_T[1][0], grad_u[1][1] + grad_u_T[1][1]],
            ]
        )

        # Exact stress tensor (\sigma(\epsilon(u)))
        sigma = [
            [2 * mu * strain[0][0] + lam * trace_grad_u, 2 * mu * strain[0][1]],
            [2 * mu * strain[1][0], 2 * mu * strain[1][1] + lam * trace_grad_u],
        ]
        return sigma

    def evaluate_exact_force(
        self,
        sd: pp.Grid,
        time: float,
        sigma: sym.Expr,
        inds: np.ndarray,
        force_array: np.ndarray,
    ) -> np.ndarray:
        """Evaluate exact elastic force at the face centers for certain face indices.

        This method is called more than once in the same simulation. This is because the
        exact elastic force values may be different in two or more parts of the
        subdomain. In the case of two parts of the subdomain, the method fills half the
        `force_array` in the first call, and then the other half in the second call. The
        filling of the array is done index wise, determined by `inds`.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.
            sigma: Exact stress tensor.
            inds: The face indices which we are computing the force at.
            force_array: Either empty or semi empty array of shape (self.nd, sd.
                num_faces) which we are filling with the force values. This is done by
                indices in `inds`.

        Returns:
            Array of `shape=(self.nd, sd.num_faces)` containing the exact elastic force
            at the face centers of `inds` for the given `time`.

        Notes:
            - The returned elastic force is _not_ given in PorePy's flattened vector
              format. Thus, it may be necessary to flatten it at a later point.
            - Recall that force = (stress dot_prod unit_normal) * face_area.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        fc = sd.face_centers[:, inds].squeeze()
        fn = sd.face_normals[:, inds].squeeze()

        # Lambdify expression
        sigma_total_fun = [
            [
                sym.lambdify((x, y, t), sigma[0][0], "numpy"),
                sym.lambdify((x, y, t), sigma[0][1], "numpy"),
            ],
            [
                sym.lambdify((x, y, t), sigma[1][0], "numpy"),
                sym.lambdify((x, y, t), sigma[1][1], "numpy"),
            ],
        ]

        # Face-centered elastic force
        force_total_fc: list[np.ndarray] = [
            # (sigma_xx * n_x + sigma_xy * n_y) * face_area
            sigma_total_fun[0][0](fc[0], fc[1], time) * fn[0]
            + sigma_total_fun[0][1](fc[0], fc[1], time) * fn[1],
            # (sigma_yx * n_x + sigma_yy * n_y) * face_area
            sigma_total_fun[1][0](fc[0], fc[1], time) * fn[0]
            + sigma_total_fun[1][1](fc[0], fc[1], time) * fn[1],
        ]

        # Insert values into force_array at the indices given by inds
        for i, force_component in enumerate(force_total_fc):
            # Update the appropriate row of force_array
            force_array[i, inds] = force_component

        return force_array

    def evaluate_exact_heterogeneous_force(self, sd: pp.Grid) -> np.ndarray:
        """Evaluate the exact heterogeneous force in the entire domain.

        The domain is split in 2: One left and one right part, where the exact force may
        be different in each part of the domain. This method handles computing the exact
        force values for the entire domain, one region at a time.

        Parameters:
            sd: The subdomain grid where the forces are to be evaluated.

        Returns:
            A flattened array of the exact force values in the entire domain.

        """
        x, _, u_left, u_right = self.heterogeneous_analytical_solution(lambdify=False)

        mu_lambda_values = {
            "left": (self.solid.lame_lambda, self.solid.shear_modulus),
            "right": (
                self.solid.lame_lambda * self.heterogeneity_factor,
                self.solid.shear_modulus * self.heterogeneity_factor,
            ),
        }

        sigma = {}
        for side, (lam, mu) in mu_lambda_values.items():
            u = u_left if side == "left" else u_right
            sigma[side] = self.exact_heterogeneous_sigma(u, lam, mu, x)

        fc_x = sd.face_centers[0, :]

        empty_force_array = np.zeros((self.nd, sd.num_faces))

        inds_left = np.where(fc_x < self.heterogeneity_location)
        inds_right = np.where(fc_x >= self.heterogeneity_location)

        semi_full_force_array = self.evaluate_exact_force(
            sd, self.time_manager.time, sigma["left"], inds_left, empty_force_array
        )
        full_force_array = self.evaluate_exact_force(
            sd,
            self.time_manager.time,
            sigma["right"],
            inds_right,
            semi_full_force_array,
        )
        full_force_array: np.ndarray = np.asarray(full_force_array).ravel("F")
        return full_force_array

    def vector_valued_mu_lambda(self) -> None:
        """Setting a vertically layered medium based on self.heterogeneity_location."""
        subdomain = self.mdg.subdomains(dim=self.nd)[0]
        x = subdomain.cell_centers[0, :]

        lmbda1 = self.solid.lame_lambda
        mu1 = self.solid.shear_modulus

        lmbda2 = self.solid.lame_lambda * self.heterogeneity_factor
        mu2 = self.solid.shear_modulus * self.heterogeneity_factor

        lmbda_vec = np.ones(subdomain.num_cells)
        mu_vec = np.ones(subdomain.num_cells)

        left_layer = x <= self.heterogeneity_location
        right_layer = x > self.heterogeneity_location

        lmbda_vec[left_layer] *= lmbda1
        mu_vec[left_layer] *= mu1

        lmbda_vec[right_layer] *= lmbda2
        mu_vec[right_layer] *= mu2

        self.mu_vector = mu_vec
        self.lambda_vector = lmbda_vec


class ExportData:
    def data_to_export(self):
        data = super().data_to_export()
        if self.time_manager.final_time_reached():
            self.compute_and_save_errors(filename=self.filename_path)
        return data

    def compute_and_save_errors(self, filename: str) -> None:
        sd = self.mdg.subdomains(dim=self.nd)[0]

        x = sd.cell_centers[0, :]
        L = self.heterogeneity_location

        left_solution, right_solution = self.heterogeneous_analytical_solution()
        left_layer = x <= L
        right_layer = x > L

        vals = np.zeros((self.nd, sd.num_cells))
        vals[0, left_layer] = left_solution[0](x[left_layer], self.time_manager.time)
        vals[0, right_layer] = right_solution[0](x[right_layer], self.time_manager.time)

        displacement_ad = self.displacement([sd])
        u_approximate = self.equation_system.evaluate(displacement_ad)
        exact_displacement = vals.ravel("F")

        exact_force = self.evaluate_exact_heterogeneous_force(sd=sd)
        force_ad = self.stress([sd])
        approx_force = self.equation_system.evaluate(force_ad)

        error_displacement = ConvergenceAnalysis.lp_error(
            grid=sd,
            true_array=exact_displacement,
            approx_array=u_approximate,
            is_scalar=False,
            is_cc=True,
            relative=True,
        )
        error_traction = ConvergenceAnalysis.lp_error(
            grid=sd,
            true_array=exact_force,
            approx_array=approx_force,
            is_scalar=False,
            is_cc=False,
            relative=True,
        )
        with open(filename, "a") as file:
            file.write(
                f"{sd.num_cells}, {self.time_manager.time_index}, {error_displacement}, {error_traction}\n"
            )


class TensorForConvergenceWithAbsorbingBoundaries:
    def stiffness_tensor(self, subdomain: pp.Grid):
        """Compute the stiffness tensor for a given subdomain.

        Takes into consideration that there is a heterogeneity factor for determining
        the heterogeneity in a domain. The tensor of a certain cell is multiplied by the
        heterogeneity factor if the cell is inside the constraints that are defined in
        set_fractures() and set_polygons().

        Parameters:
            subdomain: The subdomain where the tensor is to be defined.

        Returns:
            The stiffness tensor.

        """
        # Fetch inner domain indices
        inner_cell_indices = use_constraints_for_inner_domain_cells(
            model=self,
            sd=subdomain,
        )

        h = self.heterogeneity_factor

        # Preparing basis arrays for inner and outer domains
        inner = np.zeros(subdomain.num_cells)
        inner[inner_cell_indices] = 1

        outer = np.ones(subdomain.num_cells)
        outer = outer - inner

        # Compute the stiffness tensors for each term using the extracted constants
        stiffness_matrices = create_stiffness_tensor_basis(
            lambda_val=1,
            lambda_parallel=1,
            lambda_perpendicular=1,
            mu_parallel=1,
            mu_perpendicular=1,
            n=self.params.get("symmetry_axis", [0, 0, 1]),
        )

        # Extract stiffness matrices for each component
        lmbda_mat = stiffness_matrices["lambda"]
        lambda_parallel_mat = stiffness_matrices["lambda_parallel"]
        lambda_orthogonal_mat = stiffness_matrices["lambda_perpendicular"]
        mu_parallel_mat = stiffness_matrices["mu_parallel"]
        mu_orthogonal_mat = stiffness_matrices["mu_perpendicular"]

        # Extract individual constants from the anisotropy constants dictionary
        anisotropy_constants = self.params.get(
            "anisotropy_constants",
            {
                "mu_parallel": self.solid.shear_modulus,
                "mu_orthogonal": self.solid.shear_modulus,
                "lambda_parallel": 0.0,
                "lambda_orthogonal": 0.0,
                "volumetric_compr_lambda": self.solid.lame_lambda,
            },
        )

        volumetric_compr_lambda = anisotropy_constants["volumetric_compr_lambda"] * h
        mu_parallel = anisotropy_constants["mu_parallel"] * h
        mu_orthogonal = anisotropy_constants["mu_orthogonal"] * h
        lambda_parallel = anisotropy_constants["lambda_parallel"] * h
        lambda_orthogonal = anisotropy_constants["lambda_orthogonal"]

        # Standard material values: assigned to the outer domain
        lmbda = self.solid.lame_lambda * outer
        mu = self.solid.shear_modulus * outer

        # Values for inner domain with anisotropic constants
        mu_parallel_inner = mu_parallel * inner
        mu_orthogonal_inner = mu_orthogonal * inner
        volumetric_compr_lambda_inner = volumetric_compr_lambda * inner
        lambda_parallel_inner = lambda_parallel * inner
        lambda_orthogonal_inner = lambda_orthogonal * inner

        # Create the final stiffness tensor
        stiffness_tensor = pp.FourthOrderTensor(
            mu=mu,
            lmbda=lmbda,
            other_fields={
                "mu_parallel": (mu_parallel_mat, mu_parallel_inner),
                "mu_orthogonal": (mu_orthogonal_mat, mu_orthogonal_inner),
                "volumetric_compr_lambda": (lmbda_mat, volumetric_compr_lambda_inner),
                "lambda_parallel": (lambda_parallel_mat, lambda_parallel_inner),
                "lambda_orthogonal": (lambda_orthogonal_mat, lambda_orthogonal_inner),
            },
        )
        return stiffness_tensor


class ABCModel(
    HeterogeneityProperties,
    Geometry,
    BoundaryConditionsAndSource,
    InitialConditions,
    ExactHeterogeneousSigmaAndForce,
    ExportData,
    TensorForConvergenceWithAbsorbingBoundaries,
    DynamicMomentumBalanceABCLinear,
):
    """Model class setup for the convergence analysis with absorbing boundaries."""

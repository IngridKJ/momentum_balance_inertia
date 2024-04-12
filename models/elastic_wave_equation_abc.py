from functools import cached_property
from typing import Callable, Sequence, Union, cast

import numpy as np
import porepy as pp
import time_derivatives
from porepy.models.momentum_balance import MomentumBalance
from utils import (acceleration_velocity_displacement, body_force_function,
                   u_v_a_wrap)


class NamesAndConstants:
    def _is_nonlinear_problem(self) -> bool:
        return False

    @property
    def bc_robin_key(self):
        # Key for robin boundary conditions
        return "bc_robin"

    @property
    def beta(self) -> float:
        return 0.25

    @property
    def gamma(self) -> float:
        return 0.5

    @property
    def velocity_key(self) -> str:
        """Key for velocity in the time step and iterate dictionaries."""
        return "velocity"

    @property
    def acceleration_key(self) -> str:
        """Key for acceleration in the time step and iterate dictionaries."""
        return "acceleration"

    @property
    def bc_values_mechanics_key(self) -> str:
        """Key for mechanical boundary conditions in the time step and iterate
        dictionaries.

        """
        return "bc_values_mechanics"

    def primary_wave_speed(self, is_scalar: bool = False) -> Union[float, np.ndarray]:
        """Primary wave speed (c_p).

        Speed of the compressive elastic waves.

        Parameters:
            is_scalar: Whether the primary wavespeed should be scalar or not. Relevant
                for use with manufactured solutions, where the vector valued lambda and
                mu are not available yet when a call to this function is made.

        Returns:
            The value of the compressive elastic waves. Either scalar valued or the
            cell-center values, depending on the input parameter.

        """
        rho = self.solid.density()
        if not is_scalar:
            return np.sqrt((self.lambda_vector + 2 * self.mu_vector) / rho)
        else:
            return np.sqrt(
                (self.solid.lame_lambda() + 2 * self.solid.shear_modulus() / rho)
            )

    def secondary_wave_speed(self, is_scalar: bool = False) -> Union[float, np.ndarray]:
        """Secondary wave speed (c_s).

        Speed of the shear elastic waves.

        Parameters:
            is_scalar: Whether the secondary wavespeed should be scalar or not. Relevant
                for use with manufactured solutions, where the vector valued mu is not
                available yet when a call to this function is made.

        Returns:
            The value of the shear elastic waves. Either scalar valued or the
            cell-center values, depending on the input parameter.

        """
        rho = self.solid.density()
        if not is_scalar:
            return np.sqrt(self.mu_vector / rho)
        else:
            return np.sqrt(self.solid.shear_modulus() / rho)


class BoundaryAndInitialConditions:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for the absorbing boundary condition model class.

        Assigns Robin boundaries to all subdomain boundaries. This includes setting the
        Robin weight.

        """
        # Fetch boundary sides and assign type of boundary condition for the different
        # sides
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north
            + bounds.south
            + bounds.east
            + bounds.west
            + bounds.bottom
            + bounds.top,
            "rob",
        )

        # Calling helper function for assigning the Robin weight
        self.assign_robin_weight(sd=sd, bc=bc)
        return bc

    def assign_robin_weight(
        self, sd: pp.Grid, bc: pp.BoundaryConditionVectorial
    ) -> None:
        """Method for assigning the Robin weight.

        Using Robin boundary conditions we need to assign the robin weight. That is,
        alpha in the following, general, Robin boundary condition expression:

            sigma * n + alpha * u = G

        This method assigns the weigt values corresponding to a first order absorbing
        boundary condition.

        Parameters:
            sd: The subdomain whose boundary conditions are to be defined.
            bc: The vectorial boundary condition object.

        """
        # Initiating the arrays for the Robin weight
        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        value = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F")

        # Problems with fractures is solved by this
        if sd.dim != self.nd:
            bc.robin_weight = value
        else:
            total_coefficient_matrix = self.total_coefficient_matrix(
                sd=sd,
            )
            total_coefficient_matrix *= self.discrete_robin_weight_coefficient

            # Fethcing all boundary faces for the domain
            boundary_faces = sd.get_boundary_faces()

            value[:, :, boundary_faces] *= total_coefficient_matrix.T

            # Finally setting the actual Robin weight.
            bc.robin_weight = value

    def total_coefficient_matrix(self, sd: pp.Grid) -> np.ndarray:
        """Coefficient matrix for the absorbing boundary conditions.

        Used together with Robin boundary condition assignment.

        The absorbing boundary conditions look like the following:
            \sigma * n + D * u_t = 0,

            where D is a matrix containing material parameters and wave velocities.

        Approximate the time derivative by a first or second order backward difference:
            1: \sigma * n + D_h * u_n = D_h * u_(n-1),
            2: \sigma * n + D_h * 3/2 * u_n = D_h * (2 * u_(n-1) - 0.5 * u_(n-2)).

            D_h = 1/dt * D.

        Parameters:
            grid: The grid where the boundary conditions are assigned.


        Returns:
            A block array which is the sum of the normal and tangential component of the
            coefficient matrices.

        """
        # Fetching necessary grid related quantities
        boundary_cells = self.boundary_cells_of_grid
        boundary_faces = sd.get_boundary_faces()
        face_normals = sd.face_normals[:, boundary_faces][: self.nd]
        unitary_face_normals = face_normals / np.linalg.norm(
            face_normals, axis=0, keepdims=True
        )

        # This cursed line does the same as the line that is commented out below. Thank
        # you Yura for helping with this!!
        tensile_matrices = np.einsum(
            "ik,jk->kij", unitary_face_normals, unitary_face_normals
        )
        # tensile_matrices = np.array(
        #     [np.outer(column, column) for column in sd.face_normals.T]
        # )

        # Creating a block array of identity matrices. Subtracting the tensile matrix
        # block array from this provides us with the shear matrix block array.
        eye_block_array = np.tile(np.eye(self.nd), (len(tensile_matrices), 1, 1))
        shear_mat = eye_block_array - tensile_matrices

        # Scaling with Robin weight
        tensile_coeff = self.robin_weight_value(direction="tensile")[boundary_cells]
        shear_coeff = self.robin_weight_value(direction="shear")[boundary_cells]

        tensile_matrix_with_coeff = tensile_matrices * tensile_coeff[:, None, None]
        shear_matrix_with_coeff = shear_mat * shear_coeff[:, None, None]
        total_coefficient_matrix = tensile_matrix_with_coeff + shear_matrix_with_coeff
        return total_coefficient_matrix

    def robin_weight_value(self, direction: str) -> float:
        """Weight for Robin boundary conditions.

        Either shear or tensile Robin weight will be returned by this method. This
        depends on whether shear or tensile "direction" is chosen.

        Parameters:
            direction: Whether the boundary condition that uses the weight is the shear
                or tensile component of the displacement.

        Returns:
            The weight/coefficient for use in the Robin boundary conditions.

        """
        dt = self.time_manager.dt
        rho = self.solid.density()
        mu = self.mu_vector

        if direction == "shear":
            value = np.sqrt(rho * mu) * 1 / dt
        elif direction == "tensile":
            lmbda = self.lambda_vector
            value = np.sqrt(rho * (lmbda + 2 * mu)) * 1 / dt
        return value

    def boundary_displacement(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Method for reconstructing the boundary displacement.

        Usage within the "realm" of absorbing boundary conditions: we need the
        displacement values on the boundary at the previous time step.

        Note: This is for the pure mechanical problem - even without faults.
            Modifications are needed when a coupling to fluid flow is introduced at some
            later point.

        Parameters:
            subdomains: List of subdomains. Should be of co-dimension 0.

        Returns:
            Ad operator representing the displacement on grid faces of the subdomains.

        """
        # Discretization
        discr = self.stress_discretization(subdomains)

        # Boundary conditions on external boundaries
        bc = self._combine_boundary_operators(  # type: ignore [call-arg]
            subdomains=subdomains,
            dirichlet_operator=self.displacement,
            neumann_operator=self.mechanical_stress,
            bc_type=self.bc_type_mechanics,
            robin_operator=lambda bgs: self.create_boundary_operator(
                name=self.bc_robin_key, domains=bgs
            ),
            dim=self.nd,
            name="bc_values_mechanics",
        )
        # Displacement
        displacement = self.displacement(subdomains)

        boundary_displacement = (
            discr.bound_displacement_cell() @ displacement
            + discr.bound_displacement_face() @ bc
        )

        boundary_displacement.set_name("boundary_displacement")
        return boundary_displacement

    def initial_displacement(self, dofs: int) -> np.ndarray:
        """Initial displacement values."""
        sd = self.mdg.subdomains()[0]
        t = self.time_manager.time

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        vals = np.zeros((self.nd, sd.num_cells))

        if self.nd == 2:
            displacement_function = u_v_a_wrap(self)

            vals[0] = displacement_function[0](x, y, t)
            vals[1] = displacement_function[1](x, y, t)

        elif self.nd == 3:
            z = sd.cell_centers[2, :]

            displacement_function = u_v_a_wrap(self, is_2D=False)

            vals[0] = displacement_function[0](x, y, z, t)
            vals[1] = displacement_function[1](x, y, z, t)
            vals[2] = displacement_function[2](x, y, z, t)

        return vals.ravel("F")

    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial velocity values."""
        sd = self.mdg.subdomains()[0]
        t = self.time_manager.time

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        vals = np.zeros((self.nd, sd.num_cells))

        if self.nd == 2:
            velocity_function = u_v_a_wrap(self, return_dt=True)

            vals[0] = velocity_function[0](x, y, t)
            vals[1] = velocity_function[1](x, y, t)

        elif self.nd == 3:
            z = sd.cell_centers[2, :]

            velocity_function = u_v_a_wrap(self, is_2D=False, return_dt=True)

            vals[0] = velocity_function[0](x, y, z, t)
            vals[1] = velocity_function[1](x, y, z, t)
            vals[2] = velocity_function[2](x, y, z, t)

        return vals.ravel("F")

    def initial_acceleration(self, dofs: int) -> np.ndarray:
        """Initial acceleration values."""
        sd = self.mdg.subdomains()[0]
        t = self.time_manager.time

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        vals = np.zeros((self.nd, sd.num_cells))

        if self.nd == 2:
            acceleration_function = u_v_a_wrap(self, return_ddt=True)

            vals[0] = acceleration_function[0](x, y, t)
            vals[1] = acceleration_function[1](x, y, t)

        elif self.nd == 3:
            z = sd.cell_centers[2, :]

            acceleration_function = u_v_a_wrap(self, is_2D=False, return_ddt=True)

            vals[0] = acceleration_function[0](x, y, z, t)
            vals[1] = acceleration_function[1](x, y, z, t)
            vals[2] = acceleration_function[2](x, y, z, t)

        return vals.ravel("F")


class MyEquations:
    def momentum_balance_equation(self, subdomains: list[pp.Grid]):
        inertia_mass = self.inertia(subdomains)
        stress = pp.ad.Scalar(-1) * self.stress(subdomains)
        body_force = self.body_force(subdomains)

        equation = self.balance_equation(
            subdomains, inertia_mass, stress, body_force, dim=self.nd
        )
        equation.set_name("momentum_balance_equation")
        return equation

    def inertia(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        mass_density = self.solid_density(subdomains)
        mass = self.volume_integral(mass_density, subdomains, dim=self.nd)
        mass.set_name("inertia_mass")
        return mass

    def balance_equation(
        self,
        subdomains: list[pp.Grid],
        inertia_mass: pp.ad.Operator,
        surface_term: pp.ad.Operator,
        source: pp.ad.Operator,
        dim: int,
    ) -> pp.ad.Operator:
        div = pp.ad.Divergence(subdomains, dim=dim)

        op = self.displacement(subdomains)
        dt_op = self.velocity_time_dep_array(subdomains)
        ddt_op = self.acceleration_time_dep_array(subdomains)

        inertia_term = time_derivatives.inertia_term(
            model=self,
            op=op,
            dt_op=dt_op,
            ddt_op=ddt_op,
            time_step=pp.ad.Scalar(self.time_manager.dt),
        )

        return inertia_mass * inertia_term + div @ surface_term - source


class MySolutionStrategy:
    def prepare_simulation(self) -> None:
        """Run at the start of simulation. Used for initialization etc."""
        # Set the material and geometry of the problem. The geometry method must be
        # implemented in a ModelGeometry class.
        self.set_materials()
        self.set_geometry()

        # This should be moved elsewhere. Maybe to prepare_simulation or something.
        sd = self.mdg.subdomains(dim=self.nd)[0]
        boundary_faces = sd.get_boundary_faces()
        self.boundary_cells_of_grid = sd.signs_and_cells_of_boundary_faces(
            faces=boundary_faces
        )[1]

        # Exporter initialization must be done after grid creation,
        # but prior to data initialization.
        self.initialize_data_saving()

        # Set variables, constitutive relations, discretizations and equations.
        # Order of operations is important here.
        self.set_equation_system_manager()
        self.create_variables()
        self.initial_condition()
        self.reset_state_from_file()
        self.set_equations()

        self.set_discretization_parameters()
        self.discretize()
        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()

        # Export initial condition
        self.save_data_time_step()

    def set_discretization_parameters(self) -> None:
        """Set discretization parameters for the simulation.

        Sets eta = 1/3 on all faces if it is a simplex grid. Default is to have 0 on the
        boundaries, but this causes divergence of the solution. 1/3 all over fixes this
        issue.

        """

        super().set_discretization_parameters()
        if self.params["grid_type"] == "simplex":
            num_subfaces = 0
            for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
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

    def velocity_time_dep_array(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.TimeDependentDenseArray:
        """Velocity.

        !!!

        Parameters:
            subdomains: List of subdomains on which to define the velocity.

        Returns:
            Array of the velocity values.

        """
        if not all([sd.dim == self.nd for sd in subdomains]):
            raise ValueError("Subdomains must be of dimension nd.")
        return pp.ad.TimeDependentDenseArray(self.velocity_key, subdomains)

    def acceleration_time_dep_array(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.TimeDependentDenseArray:
        """Acceleration.

        !!!

        Parameters:
            subdomains: List of subdomains on which to define the acceleration.

        Returns:
            Array of the acceleration values.

        """
        if not all([sd.dim == self.nd for sd in subdomains]):
            raise ValueError("Subdomains must be of dimension nd.")
        return pp.ad.TimeDependentDenseArray(self.acceleration_key, subdomains)

    def initial_displacement(self, dofs: int) -> np.ndarray:
        """Initial displacement values."""
        return np.zeros(dofs * self.nd)

    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial velocity values."""
        return np.zeros(dofs * self.nd)

    def initial_acceleration(self, dofs: int) -> np.ndarray:
        """Initial acceleration values."""
        return np.zeros(dofs * self.nd)

    def initial_condition(self):
        """Assigning initial displacement, velocity and acceleration values."""
        super().initial_condition()

        # # This should be moved elsewhere. Maybe to prepare_simulation or something.
        # sd = self.mdg.subdomains(dim=self.nd)[0]
        # boundary_faces = sd.get_boundary_faces()
        # self.boundary_cells_of_grid = sd.signs_and_cells_of_boundary_faces(
        #     faces=boundary_faces
        # )[1]

        self.vector_valued_mu_lambda()

        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            dofs = sd.num_cells

            initial_displacement = self.initial_displacement(dofs=dofs)
            initial_velocity = self.initial_velocity(dofs=dofs)
            initial_acceleration = self.initial_acceleration(dofs=dofs)

            pp.set_solution_values(
                name=self.displacement_variable,
                values=initial_displacement,
                data=data,
                time_step_index=0,
                iterate_index=0,
            )

            pp.set_solution_values(
                name=self.velocity_key,
                values=initial_velocity,
                data=data,
                time_step_index=0,
                iterate_index=0,
            )

            pp.set_solution_values(
                name=self.acceleration_key,
                values=initial_acceleration,
                data=data,
                time_step_index=0,
                iterate_index=0,
            )

    def velocity_values(self, subdomain: list[pp.Grid]) -> np.ndarray:
        """Update of velocity values to be done after linear system solve.

        Newmark time discretization formula for the current velocity, depending on the
        current displacement value and the previous acceleration value.

        Parameters:
            subdomain: The subdomain the velocity is defined on.

        """
        data = self.mdg.subdomain_data(subdomain[0])
        dt = self.time_manager.dt

        beta = self.beta
        gamma = self.gamma

        (
            a_previous,
            v_previous,
            u_previous,
            u_current,
        ) = acceleration_velocity_displacement(model=self, data=data)

        v = (
            v_previous * (1 - gamma / beta)
            + a_previous * (1 - gamma - (gamma * (1 - 2 * beta)) / (2 * beta)) * dt
            + (u_current - u_previous) * gamma / (beta * dt)
        )
        return v

    def acceleration_values(self, subdomain: pp.Grid) -> np.ndarray:
        """Update of acceleration values to be done after linear system solve.

        Newmark time discretization formula for the current acceleration, depending on
        the current displacement value and the previous velocity value.

        Parameters:
            subdomain: The subdomain the acceleration is defined on.

        """
        data = self.mdg.subdomain_data(subdomain[0])
        dt = self.time_manager.dt

        beta = self.beta

        (
            a_previous,
            v_previous,
            u_previous,
            u_current,
        ) = acceleration_velocity_displacement(model=self, data=data)

        a = (
            (u_current - u_previous) / (dt**2 * beta)
            - v_previous / (dt * beta)
            - a_previous * (1 - 2 * beta) / (2 * beta)
        )
        return a

    def update_time_dependent_ad_arrays_loc(self, initial: bool) -> None:
        """Update the time dependent arrays for the velocity and acceleration.

        Parameters:
            initial: If True, the array generating method is called for both the stored
                time steps and the stored iterates. If False, the array generating
                method is called only for the iterate, and the time step solution is
                updated by copying the iterate.

        """
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            vals_acceleration = self.acceleration_values([sd])
            vals_velocity = self.velocity_values([sd])
            if initial:
                pp.set_solution_values(
                    name=self.velocity_key,
                    values=vals_velocity,
                    data=data,
                    time_step_index=0,
                )
                pp.set_solution_values(
                    name=self.acceleration_key,
                    values=vals_acceleration,
                    data=data,
                    time_step_index=0,
                )
            else:
                # Copy old values from iterate to the solution.
                vals_velocity_it = pp.get_solution_values(
                    name=self.velocity_key, data=data, iterate_index=0
                )
                vals_acceleration_it = pp.get_solution_values(
                    name=self.acceleration_key, data=data, iterate_index=0
                )
                pp.set_solution_values(
                    name=self.velocity_key,
                    values=vals_velocity_it,
                    data=data,
                    time_step_index=0,
                )
                pp.set_solution_values(
                    name=self.acceleration_key,
                    values=vals_acceleration_it,
                    data=data,
                    time_step_index=0,
                )

            pp.set_solution_values(
                name=self.velocity_key,
                values=vals_velocity,
                data=data,
                iterate_index=0,
            )
            pp.set_solution_values(
                name=self.acceleration_key,
                values=vals_acceleration,
                data=data,
                iterate_index=0,
            )

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the solution, visualization, etc.

        Parameters:
            solution: The new solution, as computed by the non-linear solver.
            errors: The error in the solution, as computed by the non-linear solver.
            iteration_counter: The number of iterations performed by the non-linear
                solver.

        """
        solution = self.equation_system.get_variable_values(iterate_index=0)

        self.update_time_dependent_ad_arrays_loc(initial=True)

        self.equation_system.shift_time_step_values()
        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )
        self.convergence_status = True
        self.save_data_time_step()


class ConstitutiveLawsDynamicMomentumBalance:
    def vector_valued_mu_lambda(self) -> None:
        """Vector representation of mu and lambda.

        Cell-wise representation of the mu and lambda quantities in the rock matrix.

        """
        subdomain = self.mdg.subdomains(dim=self.nd)[0]

        self.lambda_vector = self.solid.lame_lambda() * np.ones(subdomain.num_cells)
        self.mu_vector = self.solid.shear_modulus() * np.ones(subdomain.num_cells)

    def stiffness_tensor(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa].

        Parameters:
            subdomain: Subdomain where the stiffness tensor is defined.

        Returns:
            Cell-wise stiffness tensor in SI units.

        """
        return pp.FourthOrderTensor(self.mu_vector, self.lambda_vector)


class SourceTermRelated:
    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()

        self.update_mechanics_source()

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Ad array represenation of the body force term."""

        external_sources = pp.ad.TimeDependentDenseArray(
            name="source_mechanics",
            domains=subdomains,
            previous_timestep=False,
        )
        return external_sources

    def update_mechanics_source(self) -> None:
        """Update values of external sources."""
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        if self.nd == 2:
            source_func = body_force_function(self)
        elif self.nd == 3:
            source_func = body_force_function(self, is_2D=False)

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
        cell_volume = sd.cell_volumes
        vals = np.zeros((self.nd, sd.num_cells))

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        if self.nd == 2:
            x_val = f[0](x, y, t)
            y_val = f[1](x, y, t)

        elif self.nd == 3:
            z = sd.cell_centers[2, :]

            x_val = f[0](x, y, z, t)
            y_val = f[1](x, y, z, t)
            z_val = f[2](x, y, z, t)

            vals[2] = z_val * cell_volume

        vals[0] = x_val * cell_volume
        vals[1] = y_val * cell_volume
        return vals.ravel("F")


class BoundaryGridRelated:
    def bc_values_robin(self, bg):
        return np.zeros(bg.num_cells)


class BoundaryGridStuff:
    """Mixin for adaptations related to Robin boundary conditions with boundary grids.

    This mixin contains everything I needed to adapt in the source code for making the
    Robin boundary conditions (and thus absorbing boundary conditions (ABCs)) work with
    the boundary grid setup. Methods from three separate files are adapted. Which chunk
    of methods belong to which files are mentioned by a comment above the first method
    in the chunk.

    It also contains some brief "documentation" of other adaptations that were needed.
    Methods herein include:
    * _combine_boundary_operators: Signature now contains a robin_operator. The code
        within is adapted such that all three boundary operators are combined, not only
        Neumann and Dirichlet.
    * _update_bc_type_filter: Included a function for Robin analogous to the Neumann and
        Dirichlet ones. Now the Robin filter values can also be fetched from the parent
        grid, projected onto the boundary grid, and then updated.
    * __bc_type_storage: Needed to have this "locally". Not entirely sure why.
    * mechanical_stress: Adapt signature in call to _combine_boundary_operators to also
        give the Robin operator.
    * displacement_divergence: Adapt signature in call to _combine_boundary_operators
        to also give the Robin operator.
    * update_all_boundary_conditions: Include a call to update Robin boundary
        conditions.

    In addition to this one needs to define the robin boundary condition key (for
    identifying the operator/values/etc.) and a method for setting Robin-related
    boundary values.
    * self.bc_robin_key: "bc_robin". Is a property defined in the base dynamic momentum
        balance model found within dynamic_momentum_balance.py.
    * self.bc_values_robin: Method for setting values to the Robin boundary conditions.
        That is, setting the right-hand side of sigma * n + alpha * u = G. Assigning the
        Robin weight has _not_ changed. This still happens in the bc_type_mechanics
        method.
        As of right now (while writing this docstring), the only occurence of this
        method is in runscripts utilizing Robin boundary conditions. This will be
        adapted soon.

    Specific change for the ABCs:
    * The right hand side of the ABCs is some coefficient multiplied by the previous
        face centered displacement value at the boundary. To obtain this, the utility
        method boundary_displacement (found in elastic_wave_equation_abc.py) was
        created to reconstruct the boundary displacements. Within here, a call to the
        (now deprecated) method bc_values_mechanics was found. Now we have to use the
        _combine_boundary_operators method instead.

    Changes related to non-trivial initial boundary values:
    * Certain simulations with ABCs include some non-trivial boundary values to be set.
        That is, a value that is dependent on the previous boundary displcement value.

        These values need to be initialized properly, and the way this was done before
        was to simply assign them before the simulation started. In the new set up,
        there is a check whether there are values present in the data dictionary the
        first time boundary conditions are to be updated. If initial values are set
        before this occurs, the method assumes the initial call has already been made
        and starts assigning new values and thus overriding the initial ones.

        This is solved within bc_values_robin by distinguishing what to return on the
        very first call to the method (self.time_manager.time_index == 0). This leads to
        the existence of an initial condition method specific for boundary values. The
        method itself is not too different from the previous method for setting initial
        bc values.

    """

    # From boundary_condition.py
    def _combine_boundary_operators(
        self,
        subdomains: Sequence[pp.Grid],
        dirichlet_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
        neumann_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
        bc_type: Callable[[pp.Grid], pp.BoundaryCondition],
        name: str,
        robin_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
        dim: int = 1,
    ) -> pp.ad.Operator:
        """Creates an operator representing Dirichlet and Neumann boundary conditions
        and projects it to the subdomains from boundary grids.

        Parameters:
            subdomains: List of subdomains.
            dirichlet_operator: Function that returns the Dirichlet boundary condition
                operator.
            neumann_operator: Function that returns the Neumann boundary condition
                operator.
            robin_operator: Function that returns the Robin boundary condition
                operator.
            dim: Dimension of the equation. Defaults to 1.
            name: Name of the resulting operator. Must be unique for an operator.

        Returns:
            Boundary condition representation operator.

        """
        boundary_grids = self.subdomains_to_boundary_grids(subdomains)

        # Creating the Dirichlet, Neumann and Robin AD expressions.
        dirichlet = dirichlet_operator(boundary_grids)
        neumann = neumann_operator(boundary_grids)
        robin = robin_operator(boundary_grids)

        # Adding bc_type function to local storage to evaluate it before every time step
        # in case if the type changes in the runtime.
        self.__bc_type_storage[name] = bc_type
        # Creating the filters to ensure that Dirichlet, Neumann and Robin arrays do not
        # intersect where we do not want it.
        dir_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_dir"), domains=boundary_grids
        )
        neu_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_neu"), domains=boundary_grids
        )
        rob_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_rob"), domains=boundary_grids
        )
        # Setting the values of the filters for the first time.
        self._update_bc_type_filter(name=name, bc_type_callable=bc_type)

        boundary_to_subdomain = pp.ad.BoundaryProjection(
            self.mdg, subdomains=subdomains, dim=dim
        ).boundary_to_subdomain

        # Ensure that the Dirichlet operator only assigns (non-zero)
        # values to faces that are marked as having Dirichlet conditions.
        dirichlet *= dir_filter
        # Same with Neumann conditions.
        neumann *= neu_filter
        # Same with Robin conditions
        robin *= rob_filter
        # Projecting from the boundary grid to the subdomain.
        result = boundary_to_subdomain @ (dirichlet + neumann + robin)
        result.set_name(name)
        return result

    def _update_bc_type_filter(
        self, name: str, bc_type_callable: Callable[[pp.Grid], pp.BoundaryCondition]
    ):
        """Update the filters for Dirichlet, Neumann and Robin values.

        This is done to discard the data related to Dirichlet boundary condition in
        cells where the ``bc_type`` is Neumann or Robin and vice versa.

        """

        # Note: transposition is unavoidable to treat vector values correctly.
        def dirichlet(bg: pp.BoundaryGrid):
            # Transpose to get a n_face x nd array with shape compatible with
            # the projection matrix.
            is_dir = bc_type_callable(bg.parent).is_dir.T
            is_dir = bg.projection() @ is_dir
            # Transpose back, then ravel (in that order).
            return is_dir.T.ravel("F")

        def neumann(bg: pp.BoundaryGrid):
            is_neu = bc_type_callable(bg.parent).is_neu.T
            is_neu = bg.projection() @ is_neu
            return is_neu.T.ravel("F")

        def robin(bg: pp.BoundaryGrid):
            is_rob = bc_type_callable(bg.parent).is_rob.T
            is_rob = bg.projection() @ is_rob
            return is_rob.T.ravel("F")

        self.update_boundary_condition(name=(name + "_filter_dir"), function=dirichlet)
        self.update_boundary_condition(name=(name + "_filter_neu"), function=neumann)
        self.update_boundary_condition(name=(name + "_filter_rob"), function=robin)

    @cached_property
    def __bc_type_storage(self) -> dict[str, Callable[[pp.Grid], pp.BoundaryCondition]]:
        """Storage of functions that determine the boundary condition type on the given
        grid.

        Used in :meth:`update_all_boundary_conditions` for Dirichlet and Neumann
        filters.

        Stores per operator name (key) a callable (value) returning an operator
        representing the BC type per subdomain.

        """
        return {}

    # From constitutive_laws.py
    def mechanical_stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Linear elastic mechanical stress.

        .. note::
            The below discretization assumes the stress is discretized with a Mpsa
            finite volume discretization. Other discretizations may be possible, but are
            not available in PorePy at the moment, and would likely require changes to
            this method.

        Parameters:
            grids: List of subdomains or boundary grids. If subdomains, should be of
                co-dimension 0.

        Raises:
            ValueError: If any grid is not of co-dimension 0.
            ValueError: If any the method is called with a mixture of subdomains and
                boundary grids.

        Returns:
            Ad operator representing the mechanical stress on the faces of the grids.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            return self.create_boundary_operator(
                name=self.stress_keyword, domains=domains  # type: ignore[call-arg]
            )

        # Check that the subdomains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                """Argument subdomains a mixture of grids and
                                boundary grids"""
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).
        domains = cast(list[pp.Grid], domains)

        for sd in domains:
            # The mechanical stress is only defined on subdomains of co-dimension 0.
            if sd.dim != self.nd:
                raise ValueError("Subdomain must be of co-dimension 0.")

        # No need to facilitate changing of stress discretization, only one is
        # available at the moment.
        discr = self.stress_discretization(domains)
        # Fractures in the domain
        interfaces = self.subdomains_to_interfaces(domains, [1])

        # Boundary conditions on external boundaries
        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=self.displacement,
            neumann_operator=self.mechanical_stress,
            bc_type=self.bc_type_mechanics,
            dim=self.nd,
            name="bc_values_mechanics",
            robin_operator=lambda bgs: self.create_boundary_operator(
                name=self.bc_robin_key, domains=bgs
            ),
        )

        proj = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=self.nd)
        # The stress in the subdomanis is the sum of the stress in the subdomain,
        # the stress on the external boundaries, and the stress on the interfaces.
        # The latter is found by projecting the displacement on the interfaces to the
        # subdomains, and let these act as Dirichlet boundary conditions on the
        # subdomains.
        stress = (
            discr.stress() @ self.displacement(domains)
            + discr.bound_stress() @ boundary_operator
            + discr.bound_stress()
            @ proj.mortar_to_primary_avg
            @ self.interface_displacement(interfaces)
        )
        stress.set_name("mechanical_stress")
        return stress

    def displacement_divergence(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Divergence of displacement [-].

        This is div(u). Note that opposed to old implementation, the temporal is not
        included here. Rather, it is handled by :meth:`pp.ad.dt`.

        Parameters:
            subdomains: List of subdomains where the divergence is defined.

        Returns:
            Divergence operator accounting from contributions from interior of the
            domain and from internal and external boundaries.

        """
        # Sanity check on dimension
        if not all(sd.dim == self.nd for sd in subdomains):
            raise ValueError("Displacement divergence only defined in nd.")

        # Obtain neighbouring interfaces
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        # Mock discretization (empty `discretize` method), used to access discretization
        # matrices computed by Biot discretization.
        discr = pp.ad.DivUAd(self.stress_keyword, subdomains, self.darcy_keyword)
        # Projections
        sd_projection = pp.ad.SubdomainProjections(subdomains, dim=self.nd)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=self.nd
        )

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=subdomains,
            dirichlet_operator=self.displacement,
            neumann_operator=self.mechanical_stress,
            bc_type=self.bc_type_mechanics,
            dim=self.nd,
            name="bc_values_mechanics",
            robin_operator=lambda bgs: self.create_boundary_operator(
                name=self.bc_robin_key, domains=bgs
            ),
        )

        # Compose operator.
        div_u_integrated = discr.div_u() @ self.displacement(
            subdomains
        ) + discr.bound_div_u() @ (
            boundary_operator
            + sd_projection.face_restriction(subdomains)
            @ mortar_projection.mortar_to_primary_avg
            @ self.interface_displacement(interfaces)
        )
        # Divide by cell volumes to counteract integration.
        # The div_u discretization contains a volume integral. Since div u is used here
        # together with intensive quantities, we need to divide by cell volumes.
        cell_volumes_inv = pp.ad.Scalar(1) / self.wrap_grid_attribute(
            subdomains, "cell_volumes", dim=1  # type: ignore[call-arg]
        )
        div_u = cell_volumes_inv * div_u_integrated
        div_u.set_name("div_u")
        return div_u

    # From momentum_balance
    def update_all_boundary_conditions(self) -> None:
        """Set values for the displacement and the stress on boundaries."""
        super().update_all_boundary_conditions()
        self.update_boundary_condition(
            self.displacement_variable, self.bc_values_displacement
        )
        self.update_boundary_condition(self.stress_keyword, self.bc_values_stress)
        self.update_boundary_condition(self.bc_robin_key, self.bc_values_robin)


class DynamicMomentumBalanceCommonParts(
    NamesAndConstants,
    BoundaryAndInitialConditions,
    MyEquations,
    ConstitutiveLawsDynamicMomentumBalance,
    SourceTermRelated,
    MySolutionStrategy,
    BoundaryGridRelated,
    BoundaryGridStuff,
    MomentumBalance,
):
    """Class of subclasses/methods that are common for ABC_1 and ABC_2."""

    ...


# From here and downwards: The classes BoundaryAndInitialConditionValuesX that are
# unique for ABC_X.
class BoundaryAndInitialConditionValues1:
    """Class with methods that are unique to ABC_1"""

    @property
    def discrete_robin_weight_coefficient(self) -> float:
        """Additional coefficient for discrete Robin boundary conditions.

        After discretizing the time derivative in the absorbing boundary condition
        expressions, there might appear coefficients additional to those within the
        coefficient matrix. This model property assigns that coefficient.

        Returns:
            The Robin weight coefficient.

        """
        return 1.0

    def bc_values_robin(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Method for assigning Robin boundary condition values.

        Parameters:
            boundary_grid: List of boundary grids on which to define boundary
                conditions.

        Returns:
            Array of boundary values.

        """
        if boundary_grid.dim != (self.nd - 1):
            return np.array([])

        if self.time_manager.time_index != 0:
            # After initialization, we need to fetch previous displacement values. This
            # is done by a helper function.
            displacement_values = self._previous_displacement_values(
                boundary_grid=boundary_grid
            )
        elif self.time_manager.time_index == 0:
            # The first time this method is called is on initialization of the boundary
            # values.
            return self.initial_condition_bc(boundary_grid)

        total_disp_vals = displacement_values

        sd = boundary_grid.parent
        boundary_faces = sd.get_boundary_faces()

        total_coefficient_matrix = self.total_coefficient_matrix(sd=sd)

        result = np.matmul(
            total_coefficient_matrix, total_disp_vals.T[..., None]
        ).squeeze(-1)
        result = result * sd.face_areas[boundary_faces][:, None]
        result = result.T
        return result.ravel("F")

    def _previous_displacement_values(
        self, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """Method for constructing/fetching previous boundary displacement values.

        Parameters:
            boundary_grid: The boundary grid whose displacement values we are
                interested in.

        Returns:
            An array with the displacement values on the boundary for the previous time
            step.

        """
        data = self.mdg.boundary_grid_data(boundary_grid)
        if self.time_manager.time_index > 1:
            # "Get face displacement"-strategy: Create them using
            # bound_displacement_face/-cell from second timestep and ongoing.
            sd = boundary_grid.parent
            displacement_boundary_operator = self.boundary_displacement([sd])
            displacement_values = displacement_boundary_operator.value(
                self.equation_system
            )

            displacement_values = (
                boundary_grid.projection(self.nd) @ displacement_values
            )

        elif self.time_manager.time_index == 1:
            # On first timestep, initial values are fetched from the data dictionary.
            # These initial values are assigned in the initial_condition function that
            # is called at the zeroth time step. The boundary displacement operator is
            # not available at this time.
            displacement_values = pp.get_solution_values(
                name="bc_robin", data=data, time_step_index=0
            )

        displacement_values = np.reshape(
            displacement_values, (self.nd, boundary_grid.num_cells), "F"
        )
        return displacement_values

    def initial_condition_bc(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Initial values for the boundary displacement."""
        return np.zeros((self.nd, bg.num_cells))


class BoundaryAndInitialConditionValues2:
    """Class with methods that are unique to ABC_2"""

    @property
    def discrete_robin_weight_coefficient(self) -> float:
        """Additional coefficient for discrete Robin boundary conditions.

        After discretizing the time derivative in the absorbing boundary condition
        expressions, there might appear coefficients additional to those within the
        coefficient matrix. This model property assigns that coefficient.

        Returns:
            The Robin weight coefficient.

        """
        return 3 / 2

    def bc_values_robin(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Method for assigning ABCs with second order backward difference in time.

        Parameters:
            boundary_grid: List of boundary grids on which to define boundary
                conditions.

        Returns:
            Array of boundary values.

        """
        if boundary_grid.dim != (self.nd - 1):
            return np.array([])

        if self.time_manager.time_index != 0:
            displacement_values_0, displacement_values_1 = (
                self._previous_displacement_values(boundary_grid=boundary_grid)
            )
        elif self.time_manager.time_index == 0:
            return self.initial_condition_bc(boundary_grid)

        total_disp_vals = displacement_values_0 + displacement_values_1

        sd = boundary_grid.parent
        boundary_faces = sd.get_boundary_faces()

        total_coefficient_matrix = self.total_coefficient_matrix(sd=sd)

        result = np.matmul(
            total_coefficient_matrix, total_disp_vals.T[..., None]
        ).squeeze(-1)
        result = result * sd.face_areas[boundary_faces][:, None]

        result = result.T
        return result.ravel("F")

    def _previous_displacement_values(
        self, boundary_grid: pp.BoundaryGrid
    ) -> tuple[np.ndarray, np.ndarray]:
        """Method for constructing/fetching previous boundary displacement values.

        It also makes sure to scale the displcement values with the appropriate
        coefficient. This coefficient is related to the choice of time derivative
        approximation. For ABC_2 the coefficients are 2 and -0.5.

        Parameters:
            boundary_grid: The boundary grid whose displacement values we are
                interested in.

        Returns:
            A tuple with the scaled displacement values one time step back in time and
            two time steps back in time.

        """
        data = self.mdg.boundary_grid_data(boundary_grid)
        if self.time_manager.time_index > 1:
            # The displacement value for the previous time step is constructed and the
            # one two time steps back in time is fetched from the dictionary.
            location = pp.TIME_STEP_SOLUTIONS
            name = "boundary_displacement_values"
            for i in range(1, 0, -1):
                data[location][name][i] = data[location][name][i - 1].copy()

            displacement_values_1 = pp.get_solution_values(
                name=name, data=data, time_step_index=1
            )

            sd = boundary_grid.parent
            displacement_boundary_operator = self.boundary_displacement([sd])
            displacement_values = displacement_boundary_operator.value(
                self.equation_system
            )

            displacement_values_0 = (
                boundary_grid.projection(self.nd) @ displacement_values
            )
            pp.set_solution_values(
                name="boundary_displacement_values",
                values=displacement_values_0,
                data=data,
                time_step_index=0,
            )
        elif self.time_manager.time_index == 1:
            # On first time step we need to fetch both initial values from the storage
            # location.
            displacement_values_0 = pp.get_solution_values(
                name="boundary_displacement_values", data=data, time_step_index=0
            )
            displacement_values_1 = pp.get_solution_values(
                name="boundary_displacement_values", data=data, time_step_index=1
            )

        # Reshaping the displacement value arrays to have a shape that is easier to work
        # with when assigning the robin bc values for the different domain sides.
        displacement_values_0 = np.reshape(
            displacement_values_0, (self.nd, boundary_grid.num_cells), "F"
        )
        displacement_values_1 = np.reshape(
            displacement_values_1, (self.nd, boundary_grid.num_cells), "F"
        )

        # According to the expression for ABC_2 we have a coefficient 2 in front of the
        # values u_(n-1) and -0.5 in front of u_(n-2):
        displacement_values_0 *= 2
        displacement_values_1 *= -0.5

        return displacement_values_0, displacement_values_1

    def initial_condition_bc(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Method for setting initial values for 0th and -1st time step in
        dictionary.

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the initial displacement boundary values.

        """
        vals_0 = self.initial_condition_bc_0(bg=bg)
        vals_1 = self.initial_condition_bc_1(bg=bg)

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

    def initial_condition_bc_0(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Initial boundary displacement values corresponding to time step 0."""
        return np.zeros((self.nd, bg.num_cells))

    def initial_condition_bc_1(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Initial boundary displacement values corresponding to time step -1."""
        return np.zeros((self.nd, bg.num_cells))


# Full model classes for the momentum balance with absorbing boundary conditions
class DynamicMomentumBalanceABC1(
    BoundaryAndInitialConditionValues1,
    DynamicMomentumBalanceCommonParts,
):
    """Full model class for momentum balance with absorbing boundary conditions, ABC_1.

    ABC_1 are absorbing boundary conditions where the time derivative of u (in the
    expression) is approximated by a first order backward difference.

    """

    ...


class DynamicMomentumBalanceABC2(
    BoundaryAndInitialConditionValues2,
    DynamicMomentumBalanceCommonParts,
):
    """Full model class for momentum balance with absorbing boundary conditions, ABC_2.

    ABC_2 are absorbing boundary conditions where the time derivative of u (in the
    expression) is approximated by a second order backward difference.

    """

    ...

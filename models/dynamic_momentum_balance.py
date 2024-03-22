from porepy.models.momentum_balance import MomentumBalance

import porepy as pp
import time_derivatives

from utils import acceleration_velocity_displacement

import numpy as np


class NamesAndConstants:
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

    @property
    def primary_wave_speed(self):
        """Primary wave speed (c_p).

        Speed of the compressive elastic waves.

        Returns:
            The value of the compressive elastic waves.

        """
        rho = self.solid.density()
        return np.sqrt((self.lambda_vector + 2 * self.mu_vector) / rho)

    @property
    def secondary_wave_speed(self):
        """Secondary wave speed (c_s).

        Speed of the shear elastic waves.

        Returns:
            The value of the shear elastic waves.

        """
        rho = self.solid.density()
        return np.sqrt(self.mu_vector / rho)


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        # 2D hardcoded
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.1 / self.units.m}
        return mesh_args


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

        # This should be moved elsewhere. Maybe to prepare_simulation or something.
        sd = self.mdg.subdomains(dim=self.nd)[0]
        boundary_faces = sd.get_boundary_faces()
        self.boundary_cells_of_grid = sd.signs_and_cells_of_boundary_faces(
            faces=boundary_faces
        )[1]

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

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()

        self.update_mechanics_source()

    def update_mechanics_source(self) -> None:
        """Update values of external sources."""
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        cell_volume = sd.cell_volumes

        vals[0] *= cell_volume
        vals[1] *= cell_volume
        if self.nd == 3:
            vals[2] *= cell_volume

        vals = vals.ravel("F")

        pp.set_solution_values(
            name="source_mechanics", values=vals, data=data, iterate_index=0
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
        # super().update_time_dependent_ad_arrays(initial)
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            vals_acceleration = self.acceleration_values([sd])
            vals_velocity = self.velocity_values([sd])
            if initial:
                # if self.time_manager.time_index != 0:
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


class TimeDependentSourceTerm:
    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force."""

        external_sources = pp.ad.TimeDependentDenseArray(
            name="source_mechanics",
            domains=subdomains,
            previous_timestep=False,
        )
        return external_sources


class BoundaryGridRelated:
    def bc_values_robin(self, bg):
        return np.zeros(bg.num_cells)


class DynamicMomentumBalance(
    NamesAndConstants,
    MyGeometry,
    MyEquations,
    TimeDependentSourceTerm,
    MySolutionStrategy,
    BoundaryGridRelated,
    MomentumBalance,
): ...

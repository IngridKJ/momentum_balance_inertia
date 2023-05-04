"""
Dynamic momentum balance equation - probably to inherit from momentum_balance_equation
"""
from porepy.models.momentum_balance import MomentumBalance

import porepy as pp
import time_derivatives as td
import utils as ut

import numpy as np


class NamesAndConstants:
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
    def time_step_indices(self) -> str:
        return np.array([0, 1])


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

        inertia_term = td.inertia_term(
            model=self,
            op=op,
            dt_op=dt_op,
            ddt_op=ddt_op,
            time_step=pp.ad.Scalar(self.time_manager.dt),
        )

        return inertia_mass * inertia_term + div @ surface_term - source


class MySolutionStrategy:
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

    def initial_velocity(self, dofs: int) -> np.ndarray:
        """Initial velocity values."""
        return np.zeros(dofs * self.nd)

    def initial_acceleration(self, dofs: int) -> np.ndarray:
        """Initial acceleration values."""
        return np.zeros(dofs * self.nd)

    def initial_condition(self):
        """Assigning initial velocity and acceleration values."""
        super().initial_condition()

        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            dofs = sd.num_cells

            initial_velocity = self.initial_velocity(dofs=dofs)
            initial_acceleration = self.initial_acceleration(dofs=dofs)

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

            self.update_time_dependent_ad_arrays(initial=True)

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
        ) = ut.acceleration_velocity_displacement(model=self, data=data)

        v = (
            (1 - gamma / beta) * v_previous
            + dt * (1 - gamma - (gamma * (1 - 2 * beta)) / (2 * beta)) * a_previous
            + gamma / (beta * dt) * (u_current - u_previous)
        )
        return v

    def acceleration_values(self, subdomain: pp.Grid) -> np.ndarray:
        """Update of acceleration values to be done after linear system solve (double
        check that this is actually where it happens).

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
        ) = ut.acceleration_velocity_displacement(model=self, data=data)

        a = (
            1
            / (beta * dt**2)
            * (
                u_current
                - u_previous
                - dt * v_previous
                - (1 - 2 * beta) * dt**2 / 2 * a_previous
            )
        )
        return a

    def update_time_dependent_ad_arrays(self, initial: bool) -> None:
        """Update the time dependent arrays for the velocity and acceleration.

        Parameters:
            initial: If True, the array generating method is called for both the stored
                time steps and the stored iterates. If False, the array generating
                method is called only for the iterate, and the time step solution is
                updated by copying the iterate.

        """
        super().update_time_dependent_ad_arrays(initial)
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            if initial:
                vals_velocity = self.velocity_values([sd])
                vals_acceleration = self.acceleration_values([sd])

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
                vals_velocity = ut.get_solution_values(
                    name=self.velocity_key, data=data, iterate_index=0
                )
                vals_acceleration = ut.get_solution_values(
                    name=self.acceleration_key, data=data, iterate_index=0
                )
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

            vals_velocity = self.velocity_values([sd])
            vals_acceleration = self.acceleration_values([sd])

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

    def reset_state_from_file(self) -> None:
        """Reset states but through a restart from file."""
        super().reset_state_from_file()

        self.update_time_dependent_ad_arrays(initial=True)

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        self.update_time_dependent_ad_arrays(initial=False)


class DynamicMomentumBalance(
    NamesAndConstants,
    MyEquations,
    MySolutionStrategy,
    MomentumBalance,
):
    ...

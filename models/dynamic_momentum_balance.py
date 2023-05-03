"""
Dynamic momentum balance equation - probably to inherit from momentum_balance_equation
"""
from porepy.models.momentum_balance import MomentumBalance

import porepy as pp
import time_derivatives as td
import utils as ut

import numpy as np


class MyEquations:
    def momentum_balance_equation(self, subdomains: list[pp.Grid]):
        inertia_mass = self.inertia_(subdomains)
        stress = pp.ad.Scalar(-1) * self.stress(subdomains)
        body_force = self.body_force(subdomains)

        equation = self.balance_equation(
            subdomains, inertia_mass, stress, body_force, dim=self.nd
        )
        equation.set_name("momentum_balance_equation")
        return equation

    def inertia_(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
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

    @property
    def iterate_indices(self) -> str:
        return np.array([0, 1])

    def reset_state_from_file(self) -> None:
        """Reset states but through a restart from file.

        Add treatment of boundary conditions to the standard reset of states from file.

        """
        super().reset_state_from_file()

        self.update_time_dependent_ad_arrays(initial=True)

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        # Update the mechanical boundary conditions to both the time step and iterate
        # solution.
        self.update_time_dependent_ad_arrays(initial=False)

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

    def initial_condition(self):
        super().initial_condition()

        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            dofs = sd.num_cells
            init_vals = np.zeros(dofs * self.nd) * 0.0000001
            init_vals_a = np.ones(dofs * self.nd) * 0.0000000001

            pp.set_solution_values(
                name=self.velocity_key,
                values=init_vals,
                data=data,
                time_step_index=0,
                iterate_index=0,
            )

            pp.set_solution_values(
                name=self.acceleration_key,
                values=init_vals_a,
                data=data,
                time_step_index=0,
                iterate_index=0,
            )

            self.update_time_dependent_ad_arrays(initial=True)

    def velocity_values(self, subdomain: list[pp.Grid]):
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
        super().update_time_dependent_ad_arrays(initial)
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            if initial:
                # I think this might be where update_X is needed
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
            a = 5


class DynamicMomentumBalance(
    MyEquations,
    MySolutionStrategy,
    MomentumBalance,
):
    ...

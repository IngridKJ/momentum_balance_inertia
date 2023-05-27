from porepy.models.momentum_balance import MomentumBalance

import porepy as pp
import time_derivatives

from utils import get_solution_values
from utils import acceleration_velocity_displacement

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
        """Assigning initial velocity and acceleration values."""
        super().initial_condition()

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

        # self.update_time_dependent_ad_arrays(initial=True)

    def before_nonlinear_loop(self) -> None:
        """Update values of external sources."""
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        vals = np.zeros((self.nd, sd.num_cells))

        cell_volume = sd.cell_volumes

        vals[0] *= cell_volume
        vals[1] *= cell_volume

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

        # Just for calculating rates.
        # v_e = data[pp.ITERATE_SOLUTIONS]["v_e"][0]
        # u_e = data[pp.ITERATE_SOLUTIONS]["u_e"][0]
        # u_h = data[pp.ITERATE_SOLUTIONS]["u"][0]
        # diff = v - v_e
        # nor_v = np.linalg.norm(diff)
        # diff = u_e - u_h
        # nor_u = np.linalg.norm(diff)
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
        ) = acceleration_velocity_displacement(model=self, data=data)

        a = (
            (u_current - u_previous) / (dt**2 * beta)
            - v_previous / (dt * beta)
            - a_previous * (1 - 2 * beta) / (2 * beta)
        )

        # Just for calculating rates.
        # a_e = data[pp.ITERATE_SOLUTIONS]["a_e"][0]
        # diff = a - a_e
        # nor = np.linalg.norm(diff)
        return a

    def update_time_dependent_ad_arrays(self, initial: bool) -> None:
        """Update the time dependent arrays for the velocity and acceleration.

        Parameters:
            initial: (Outdated because I messed with it to fix a bug) If True, the
                array generating method is called for both the stored time steps and the
                stored iterates. If False, the array generating method is called only
                for the iterate, and the time step solution is updated by copying the
                iterate.

        """
        super().update_time_dependent_ad_arrays(initial)
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            vals_acceleration = self.acceleration_values([sd])
            vals_velocity = self.velocity_values([sd])
            if initial:
                if self.time_manager.time_index != 0:
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
                vals_velocity_it = get_solution_values(
                    name=self.velocity_key, data=data, iterate_index=0
                )
                vals_acceleration_it = get_solution_values(
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

            if self.time_manager.time_index != 0:
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

        self.update_time_dependent_ad_arrays(initial=True)

        self.equation_system.shift_time_step_values()
        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )
        self.convergence_status = True
        self.save_data_time_step()


class TimeDependentSourceTerm:
    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force."""

        external_sources = pp.ad.TimeDependentDenseArray(
            name="source_mechanics",
            subdomains=subdomains,
            previous_timestep=False,
        )
        return external_sources


class DynamicMomentumBalance(
    NamesAndConstants,
    MyEquations,
    TimeDependentSourceTerm,
    MySolutionStrategy,
    MomentumBalance,
):
    ...

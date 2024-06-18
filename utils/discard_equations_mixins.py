from typing import cast

import numpy as np
import porepy as pp


class RemoveFractureRelatedEquationsMomentumBalance:
    """Remove equations for to fractures and fracture interfaces in momentum balance.

    To simulate simulate an internal Neumann boundary (that is, an open fracture) we
    have to remove some equations and variables. This mixin includes the necessary
    modifications to the code for the equations and variables related to fractures and
    fracture interfaces to be discarded.

    """

    def set_equations(self) -> None:
        """Set equations for the rock matrix.

        Modifications: Remove all equations but the momentum_balance_equation.

        """
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        matrix_eq = self.momentum_balance_equation(matrix_subdomains)
        self.equation_system.set_equation(
            matrix_eq, matrix_subdomains, {"cells": self.nd}
        )

    def mechanical_stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Linear elastic mechanical stress.

        Modifications: Remove the influence of interface displacements to the stress.

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
                """Argument subdomains a mixture of grids and boundary grids"""
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

        # Boundary conditions on external boundaries
        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=self.displacement,
            neumann_operator=self.mechanical_stress,
            robin_operator=self.mechanical_stress,
            bc_type=self.bc_type_mechanics,
            dim=self.nd,
            name=self.bc_values_mechanics_key,
        )

        stress = (
            discr.stress() @ self.displacement(domains)
            + discr.bound_stress() @ boundary_operator
        )
        stress.set_name("mechanical_stress")
        return stress

    def create_variables(self) -> None:
        """Set displacement variable in the matrix.

        Modifications: Do not create variables for traction and interface displacements.

        """
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.displacement_variable,
            subdomains=self.mdg.subdomains(dim=self.nd),
            tags={"si_units": "m"},
        )

    def data_to_export(self):
        """Return data to be exported.

        Modifications: Only export matrix displacements (as that is the only variable
        that exists).

        Return type should comply with pp.exporter.DataInput.

        Returns:
            List containing all (grid, name, scaled_values) tuples.

        """
        data = []
        variables = self.equation_system.variables
        for var in variables:
            scaled_values = self.equation_system.get_variable_values(
                variables=[var], time_step_index=0
            )
            units = var.tags["si_units"]
            values = self.fluid.convert_units(scaled_values, units, to_si=True)
            data.append((var.domain, var.name, values))
        return data  # type: ignore[return-value]

    def initial_condition(self) -> None:
        """Assigning initial initial conditions.

        Modifications: I have fetched all initial_condition methods from the parent
        models. One of them made a call to initiate traction values. Avoided this by
        just putting everything the initial_condition method does across models and
        inherited models.

        """
        val = np.zeros(self.equation_system.num_dofs())
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                time_step_index=time_step_index,
            )

        for iterate_index in self.iterate_indices:
            self.equation_system.set_variable_values(val, iterate_index=iterate_index)

        # Initialize time dependent ad arrays, including those for boundary values.
        self.update_time_dependent_ad_arrays()

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

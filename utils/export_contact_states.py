"""File for exporting contact states. The contents of this file is a combination of a
file I received from Marius, and methods used in Ivar's paper 'A line search
algorithm for multiphysics problems with fracture deformation'. The code for the latter one is found e.g. on GitHub.
"""

import numpy as np
import porepy as pp

Scalar = pp.ad.Scalar


class ExportContactStates:
    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        # Setting export_constants_separately to False facilitates operations such as
        # filtering by dimension in ParaView and is done here for illustrative purposes.
        self.iteration_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"] + "_iterations",
            folder_name=self.params["folder_name"],
            export_constants_separately=False,
        )

    def data_to_export(self):
        """Return data to be exported.

        Return type should comply with pp.exporter.DataInput.

        Returns:
            List containing all (grid, name, scaled_values) tuples.
        """
        data = super().data_to_export()

        for dim in range(self.nd + 1):
            for sd in self.mdg.subdomains(dim=dim):
                if dim == self.nd - 1:
                    names = ["displacement_jump", "aperture"]
                    for n in names:
                        data.append((sd, n, self._evaluate_and_scale(sd, n, "m")))
                    data.append(
                        (
                            sd,
                            "contact_states",
                            self.report_on_contact_states([sd]),
                        )
                    )
        return data

    def data_to_export_iteration(self):
        """Returns data for iteration exporting.

        Returns:
            Any type compatible with data argument of pp.Exporter().write_vtu().

        """
        # The following is a slightly modified copy of the method
        # data_to_export() from DataSavingMixin.
        data = []
        variables = self.equation_system.variables
        for var in variables:
            # Note that we use iterate_index=0 to get the current solution, whereas
            # the regular exporter uses time_step_index=0.
            scaled_values = self.equation_system.get_variable_values(
                variables=[var], iterate_index=0
            )
            units = var.tags["si_units"]
            values = self.units.convert_units(scaled_values, units, to_si=True)
            data.append((var.domain, var.name, values))

        for sd in self.mdg.subdomains(dim=self.nd - 1):
            vals = self.report_on_contact_states([sd])
            data.append((sd, "contact_states", vals))

        return data

    def save_data_iteration(self):
        """Export current solution to vtu files.

        This method is typically called by after_nonlinear_iteration.

        Having a separate exporter for iterations avoids distinguishing between iterations
        and time steps in the regular exporter's history (used for export_pvd).

        """
        # To make sure the nonlinear iteration index does not interfere with the
        # time part, we multiply the latter by the next power of ten above the
        # maximum number of nonlinear iterations. Default value set to 10 in
        # accordance with the default value used in NewtonSolver
        n = self.params.get("max_iterations", 10)
        p = round(np.log10(n))
        r = 10**p
        if r <= n:
            r = 10 ** (p + 1)
        self.iteration_exporter.write_vtu(
            self.data_to_export_iteration(),
            time_dependent=True,
            time_step=self.nonlinear_solver_statistics.num_iteration
            + r * self.time_manager.time_index,
        )

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution to
        iterate subdictionary.

        """
        super().after_nonlinear_iteration(solution_vector)
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()

    def report_on_contact_states(self, subdomains: list[pp.Grid] = None):
        """Report on the contact states of the fractures.

        Parameters:
            subdomains: List of subdomains to report on. If None, all fractures are
                considered.

        Returns:
            np.ndarray: Array of contact states, one for each fracture cell.

        """

        if subdomains is None:
            subdomains = self.mdg.subdomains(dim=self.nd - 1)

        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        contact_force_n = t_n.value(self.equation_system)
        opening = (u_n - self.fracture_gap(subdomains)).value(self.equation_system)

        c_num = self.contact_mechanics_numerical_constant(subdomains).value(
            self.equation_system
        )
        zerotol = 1e-12
        in_contact = (-contact_force_n - c_num * opening) > zerotol

        nd_vec_to_tangential = self.tangential_component(subdomains)

        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)

        # Combine the above into expressions that enter the equation
        ut_val = u_t.value(self.equation_system).reshape((self.nd - 1, -1), order="F")
        sliding = np.logical_and(np.linalg.norm(ut_val, axis=0) > zerotol, in_contact)
        # 0 sticking, 1 sliding, 2 opening
        return sliding + 2 * np.logical_not(in_contact)

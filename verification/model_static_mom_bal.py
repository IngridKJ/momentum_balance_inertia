import numpy as np
import porepy as pp
import sympy as sym
from porepy.models.momentum_balance import MomentumBalance


class Source:
    def body_force_func(self) -> list:
        """Function for calculating rhs corresponding to a manufactured solution, 2D."""
        lam = self.solid.lame_lambda()
        mu = self.solid.shear_modulus()

        x, y = sym.symbols("x y")
        manufactured_sol = self.params.get("manufactured_solution", "bubble")
        if manufactured_sol == "bubble":
            # Manufactured solution (bubble<3)
            u1 = u2 = x * (1 - x) * y * (1 - y)
            u = [u1, u2]
        elif manufactured_sol == "cub_cub":
            u1 = u2 = x**2 * y**2 * (1 - x) * (1 - y)
            u = [u1, u2]

        grad_u = [
            [sym.diff(u[0], x), sym.diff(u[0], y)],
            [sym.diff(u[1], x), sym.diff(u[1], y)],
        ]

        grad_u_T = [[grad_u[0][0], grad_u[1][0]], [grad_u[0][1], grad_u[1][1]]]

        div_u = sym.diff(u[0], x) + sym.diff(u[1], y)

        trace_grad_u = grad_u[0][0] + grad_u[1][1]

        strain = 0.5 * np.array(
            [
                [grad_u[0][0] + grad_u_T[0][0], grad_u[0][1] + grad_u_T[0][1]],
                [grad_u[1][0] + grad_u_T[1][0], grad_u[1][1] + grad_u_T[1][1]],
            ]
        )

        sigma = [
            [2 * mu * strain[0][0] + lam * trace_grad_u, 2 * mu * strain[0][1]],
            [2 * mu * strain[1][0], 2 * mu * strain[1][1] + lam * trace_grad_u],
        ]

        div_sigma = [
            sym.diff(sigma[0][0], x) + sym.diff(sigma[0][1], y),
            sym.diff(sigma[1][0], x) + sym.diff(sigma[1][1], y),
        ]

        return [
            sym.lambdify((x, y), div_sigma[0], "numpy"),
            sym.lambdify((x, y), div_sigma[1], "numpy"),
        ]

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()

        self.update_mechanics_source()

    def update_mechanics_source(self) -> None:
        """Update values of external sources."""
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        source_func = self.body_force_func()
        mech_source = self.source_values(source_func, sd, t)

        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, iterate_index=0
        )

    def source_values(self, f, sd, t) -> np.ndarray:
        """Function for computing the source values.

        Parameters:
            f: Function depending on time and space for the source term.
            sd: Subdomain where the source term is defined.
            t: Current time in the time-stepping.

        Returns:
            An array of source values.

        """
        vals = np.zeros((self.nd, sd.num_cells))

        x = sd.cell_centers[0, :]
        y = sd.cell_centers[1, :]

        x_val = f[0](x, y, t)
        y_val = f[1](x, y, t)

        cell_volume = sd.cell_volumes

        vals[0] = x_val * cell_volume
        vals[1] = y_val * cell_volume

        return vals.ravel("F")

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force."""

        external_sources = pp.ad.TimeDependentDenseArray(
            name="source_mechanics",
            subdomains=subdomains,
            previous_timestep=False,
        )
        return external_sources


class MomBalSourceAnalytical(
    Source,
    MomentumBalance,
):
    ...

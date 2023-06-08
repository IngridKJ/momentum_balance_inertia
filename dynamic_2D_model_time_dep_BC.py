import porepy as pp
import numpy as np

from dynamic_2D_model import MyMomentumBalance

from utils import body_force_function
from utils import get_solution_values


class MyGeometry:
    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.25 / 32.0 / self.units.m}
        return mesh_args


class BoundaryAndInitialCond:
    def bc_values_mechanics(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.TimeDependentDenseArray:
        """Boundary values for mechanics.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Time dependent dense array for boundary values.

        """
        # Use time dependent array to allow for time dependent boundary conditions in
        # the div(u) term.
        return pp.ad.TimeDependentDenseArray(self.bc_values_mechanics_key, subdomains)

    def time_dependent_bc_values_mechanics(
        self, subdomains: list[pp.Grid]
    ) -> np.ndarray:
        """Method for assigning the time dependent bc values.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        assert len(subdomains) == 1
        sd = subdomains[0]

        domain_sides = self.domain_boundary_sides(sd)
        values = np.zeros((self.nd, sd.num_faces))

        if self.time_manager.time_index <= 1:
            values[1, domain_sides.north] += self.solid.convert_units(
                self.params.get("uy_north", -1e-3),
                "m",
            )
        if self.time_manager.time_index <= 1:
            values[1, domain_sides.south] += self.solid.convert_units(
                self.params.get("uy_south", 1e-3),
                "m",
            )
        if self.time_manager.time_index <= 1:
            values[0, domain_sides.east] += self.solid.convert_units(
                self.params.get("ux_east", -1e-3),
                "m",
            )
        if self.time_manager.time_index <= 1:
            values[0, domain_sides.west] += self.solid.convert_units(
                self.params.get("ux_west", 1e-3),
                "m",
            )

        return values.ravel("F")

    def initial_condition(self):
        """Assigning initial bc values."""
        super().initial_condition()
        self.update_time_dependent_bc(initial=True)


class SolutionStrategySourceBC:
    def update_time_dependent_bc(self, initial: bool) -> None:
        """Method for updating the time dependent boundary conditions.

        Analogous to the method for updating other time dependent dense arrays (namely
        velocity and acceleration). But to avoid problems with the velocity and
        acceleration being updated when they should not, the updating of bc values is
        separated from them.

        Will revisit what is the most proper to do here.

        Parameters:
            initial: If True, the array generating method is called for both the stored
                time steps and the stored iterates. If False, the array generating
                method is called only for the iterate, and the time step solution is
                updated by copying the iterate.

        """
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            bc_vals = self.time_dependent_bc_values_mechanics([sd])
            if initial:
                pp.set_solution_values(
                    name=self.bc_values_mechanics_key,
                    values=bc_vals,
                    data=data,
                    time_step_index=0,
                )

            else:
                # Copy old values from iterate to the solution.
                bc_vals_it = get_solution_values(
                    name=self.bc_values_mechanics_key, data=data, iterate_index=0
                )

                pp.set_solution_values(
                    name=self.bc_values_mechanics_key,
                    values=bc_vals_it,
                    data=data,
                    time_step_index=0,
                )

            pp.set_solution_values(
                name=self.bc_values_mechanics_key,
                values=bc_vals,
                data=data,
                iterate_index=0,
            )

    def before_nonlinear_loop(self) -> None:
        """Update values of external sources and time dependent bc values."""
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        source_func = body_force_function(self)
        mech_source = self.source_values(source_func, sd, t)
        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, iterate_index=0
        )

        # Update time dependent bc
        # CHECK: Is this method called in the "same loop" as initial cond? Are they
        # updated twice?
        # Yes. It is initiated, and then updated again for iterate solutions in this
        # method. Revisit whether this is the intention or not.
        self.update_time_dependent_bc(initial=False)


class MyMomentumBalance(
    BoundaryAndInitialCond,
    MyGeometry,
    SolutionStrategySourceBC,
    MyMomentumBalance,
):
    ...


t_shift = 0.0
dt = 1.0 / 100.0
tf = 1.0

time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "density": 1,
        "lame_lambda": 1,
        "permeability": 1,
        "porosity": 1,
        "shear_modulus": 1,
    }
)

material_constants = {"solid": solid_constants}
params = {
    "time_manager": time_manager,
    "material_constants": material_constants,
    "folder_name": "delete",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
}
model = MyMomentumBalance(params)

pp.run_time_dependent_model(model, params)

import porepy as pp
import numpy as np

from dynamic_2D_model import MyMomentumBalance

from utils import body_force_function
from utils import get_solution_values


class MyGeometry:
    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.25 / 16.0 / self.units.m}
        return mesh_args


class BoundaryAndInitialCond:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        # Approximating the time derivative in the BCs and rewriting the BCs on "Robin
        # form" gives 1/(dt*c_(p/s)) as Robin weight. c_s/c_p depends on the boundary
        # side and displacement component. See handwritten notes/implementation
        # plan/reference paper (Higdon, 1991).
        dt = self.time_manager.dt
        cp = self.primary_wave_speed
        cs = self.secondary_wave_speed
        lam = self.solid.lame_lambda()
        mu = self.solid.shear_modulus()

        robin_weight_tensile = (lam + 2 * mu) / (dt * cp)
        robin_weight_shear = (mu) / (dt * cs)

        bounds = self.domain_boundary_sides(sd)

        # These two lines provides an array with 2 components. The first component is a
        # 2d array with ones in the first row and zeros in thesecond. The second
        # component is also a 2d array, but now the first row is zeros and the second
        # row is ones.
        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        value = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F")

        # value[0][0][bounds.west] *= -robin_weight_tensile
        # value[1][1][bounds.west] *= -robin_weight_shear

        # value[0][0][bounds.north] *= robin_weight_shear
        # value[1][1][bounds.north] *= robin_weight_tensile

        value[0][0][bounds.south] *= robin_weight_shear
        value[1][1][bounds.south] *= robin_weight_tensile

        # value[0][0][bounds.east] *= robin_weight_tensile
        # value[1][1][bounds.east] *= robin_weight_shear

        # Choosing type of boundary condition for the different domain sides.
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.south,
            "rob",
        )
        # bc = pp.BoundaryConditionVectorial(
        #     sd,
        #     bounds.north + bounds.south + bounds.east + bounds.west,
        #     "rob",
        # )

        bc.robin_weight = value
        return bc

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
        values = np.zeros((self.nd, sd.num_faces))
        displacement_values = np.zeros((self.nd, sd.num_faces))

        dt = self.time_manager.dt
        cp = self.primary_wave_speed
        cs = self.secondary_wave_speed
        lam = self.solid.lame_lambda()
        mu = self.solid.shear_modulus()

        robin_weight_tensile = (lam + 2 * mu) / (dt * cp)
        robin_weight_shear = (mu) / (dt * cs)

        bounds = self.domain_boundary_sides(sd)
        if self.time_manager.time_index > 1:
            displacement_boundary_operator = self.boundary_displacement([sd])
            displacement_values = displacement_boundary_operator.evaluate(
                self.equation_system
            ).val

            # Reshape for further use
            displacement_values = np.reshape(
                displacement_values, (self.nd, sd.num_faces)
            )

        # values[0][bounds.east] += (
        #     robin_weight_tensile * displacement_values[0][bounds.east]
        # )
        # values[1][bounds.east] += (
        #     robin_weight_shear * displacement_values[1][bounds.east]
        # )

        # values[0][bounds.west] += -(
        #     robin_weight_tensile * displacement_values[0][bounds.west]
        # )
        # values[1][bounds.west] += -(
        #     robin_weight_shear * displacement_values[1][bounds.west]
        # )

        # values[0][bounds.north] += (
        #     robin_weight_shear * displacement_values[0][bounds.north]
        # )
        # values[1][bounds.north] += (
        #     robin_weight_tensile * displacement_values[1][bounds.north]
        # )

        values[0][bounds.south] += (
            robin_weight_shear * displacement_values[0][bounds.south]
        )
        values[1][bounds.south] += (
            robin_weight_tensile * displacement_values[1][bounds.south]
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

    def source_values(self, f, sd, t) -> np.ndarray:
        """Computes the integrated source values by the source function.

        Parameters:
            f: Function depending on time and space for the source term.
            sd: Subdomain where the source term is defined.
            t: Current time in the time-stepping.

        Returns:
            An array of source values.

        """
        # 2D hardcoded
        vals = np.zeros((self.nd, sd.num_cells))

        vals[0][2527:2528] = 1
        vals[1][2527:2528] = 1

        if self.time_manager.time_index <= 1:
            return vals.ravel("F") * 1e-8
        else:
            return vals.ravel("F") * 0

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


class MyConstitutiveLaws:
    def boundary_displacement(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Method for reconstructing the boundary displacement.

        Note: This is for the pure mechanical problem - even without faults.
            Modifications are needed when a coupling to fluid flow is introduced at some
            later point.

        Parameters:
            subdomains: List of subdomains. Should be of co-dimension 0.

        Returns:
            Ad operator representing the displacement on grid faces of the subdomains.

        """
        # No need to facilitate changing of stress discretization, only one is
        # available at the moment.
        discr = self.stress_discretization(subdomains)

        # Boundary conditions on external boundaries
        bc = self.bc_values_mechanics(subdomains)

        # Displacement
        displacement = self.displacement(subdomains)

        # The boundary displacement is the sum of the boundary cell displacement and the
        # boundary face displacement?
        boundary_displacement = (
            discr.bound_displacement_cell @ displacement
            + discr.bound_displacement_face @ bc
        )

        boundary_displacement.set_name("boundary_displacement")
        return boundary_displacement


class MyMomentumBalance(
    BoundaryAndInitialCond,
    MyGeometry,
    SolutionStrategySourceBC,
    MyConstitutiveLaws,
    MyMomentumBalance,
):
    ...


t_shift = 0.0
tf = 0.1
dt = tf / 100.0

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
    "folder_name": "rob",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
}
model = MyMomentumBalance(params)


pp.run_time_dependent_model(model, params)

import porepy as pp
import numpy as np

from dynamic_2D_model import MyMomentumBalance

from utils import body_force_function
from utils import get_solution_values


class NewmarkConstants:
    @property
    def gamma(self) -> float:
        return 0.5

    @property
    def beta(self) -> float:
        return 0.25


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1000 / self.units.m
        y = 1000 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 20 / 2 / self.units.m}
        return mesh_args


class BoundaryAndInitialCond:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        # Approximating the time derivative in the BCs and rewriting the BCs on "Robin
        # form" gives the Robin weights.

        bounds = self.domain_boundary_sides(sd)

        # These two lines provide an array with 2 components. The first component is a
        # 2d array with ones in the first row and zeros in the second. The second
        # component is also a 2d array, but now the first row is zeros and the second
        # row is ones.
        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        value = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F")

        value[0][0][bounds.north] *= self.robin_weight_value(
            direction="shear", side="north"
        )
        value[1][1][bounds.north] *= self.robin_weight_value(
            direction="tensile", side="north"
        )

        value[0][0][bounds.south] *= self.robin_weight_value(
            direction="shear", side="south"
        )
        value[1][1][bounds.south] *= self.robin_weight_value(
            direction="tensile", side="south"
        )

        value[1][1][bounds.east] *= self.robin_weight_value(
            direction="shear", side="east"
        )
        value[0][0][bounds.east] *= self.robin_weight_value(
            direction="tensile", side="east"
        )

        value[1][1][bounds.west] *= self.robin_weight_value(
            direction="shear", side="west"
        )
        value[0][0][bounds.west] *= self.robin_weight_value(
            direction="tensile", side="west"
        )

        # Choosing type of boundary condition for the different domain sides.
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "rob",
        )

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
        # Equidimensional hard code
        assert len(subdomains) == 1
        sd = subdomains[0]
        data = self.mdg.subdomain_data(sd)

        values = np.zeros((self.nd, sd.num_faces))
        bounds = self.domain_boundary_sides(sd)

        if self.time_manager.time_index > 1:
            # Most recent "get face displacement"-idea: Create them using
            # bound_displacement_face/cell from second timestep and ongoing.
            displacement_boundary_operator = self.boundary_displacement([sd])
            displacement_values = displacement_boundary_operator.evaluate(
                self.equation_system
            ).val

        else:
            # On first timestep, initial values are fetched from the data dictionary.
            # These initial values are assigned in the initial_condition function.
            # Currently just set to zero...
            displacement_values = get_solution_values(
                name=self.bc_values_mechanics_key, data=data, time_step_index=0
            )

        displacement_values = np.reshape(displacement_values, (self.nd, sd.num_faces))

        values[0][bounds.north] += (
            self.robin_weight_value(direction="shear", side="north")
            * displacement_values[0][bounds.north]
        )
        values[1][bounds.north] += (
            self.robin_weight_value(direction="tensile", side="north")
            * displacement_values[1][bounds.north]
        )

        values[0][bounds.south] -= (
            self.robin_weight_value(direction="shear", side="south")
            * displacement_values[0][bounds.south]
        )
        values[1][bounds.south] -= (
            self.robin_weight_value(direction="tensile", side="south")
            * displacement_values[1][bounds.south]
        )

        values[1][bounds.east] += (
            self.robin_weight_value(direction="shear", side="east")
            * displacement_values[1][bounds.east]
        )
        values[0][bounds.east] += (
            self.robin_weight_value(direction="tensile", side="east")
            * displacement_values[0][bounds.east]
        )

        values[1][bounds.west] -= (
            self.robin_weight_value(direction="shear", side="west")
            * displacement_values[1][bounds.west]
        )
        values[0][bounds.west] -= (
            self.robin_weight_value(direction="tensile", side="west")
            * displacement_values[0][bounds.west]
        )

        return values.ravel("F")

    def initial_condition(self):
        """Assigning initial bc values."""
        super().initial_condition()

        sd = self.mdg.subdomains(dim=2)[0]
        data = self.mdg.subdomain_data(sd)

        bc_vals = np.zeros((sd.dim, sd.num_faces))
        bc_vals = bc_vals.flatten()

        pp.set_solution_values(
            name=self.bc_values_mechanics_key,
            values=bc_vals,
            data=data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name=self.bc_values_mechanics_key,
            values=bc_vals,
            data=data,
            iterate_index=0,
        )
        # self.update_time_dependent_bc(initial=True)


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
        vals = np.zeros((self.nd, sd.num_cells))

        # Assigning a one-cell source term in the middle of the domain
        x_mid = self.domain.bounding_box["xmax"] / 2
        y_mid = self.domain.bounding_box["ymax"] / 2
        closest_cell = sd.closest_cell(np.array([[x_mid], [y_mid], [0.0]]))[0]
        vals[0][closest_cell] = 1
        vals[1][closest_cell] = 1

        if self.time_manager.time_index <= 200:
            return vals.ravel("F") * self.time_manager.time
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

        # Update time dependent bc before next solve.
        self.update_time_dependent_bc(initial=True)


class MyConstitutiveLaws:
    def robin_weight_value(self, direction: str, side: str) -> float:
        """Shear Robin weight for Robin boundary conditions.

        Parameters:
            direction: Whether the boundary condition that uses the weight is the shear
                or tensile component of the displacement.
            side: Which boundary side is considered. This alters the sign of the weight.

        Returns:
            The weight/coefficient for use in the Robin boundary conditions.

        """
        dt = self.time_manager.dt

        cs = self.secondary_wave_speed
        cp = self.primary_wave_speed

        lam = self.solid.lame_lambda()
        mu = self.solid.shear_modulus()

        if direction == "shear":
            value = mu / (cs * dt)
            if side == "west" or side == "south":
                value = (-1) * value
        elif direction == "tensile":
            value = (lam + 2 * mu) / (cp * dt)
            if side == "west" or side == "south":
                value = -1 * value
        return value

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
        # boundary face displacement.

        # Example usage within porepy:
        # porepy\utils\examples_utils.py

        # Other note, from the file referenced above:
        # "Compute the pseudo-trace of the displacement
        # Note that this is not the real trace, as this only holds for particular
        # choices of boundary condtions."

        # Possible reason things are incorrect: Are the boundary conditions updated at
        # the correct time-step?
        boundary_displacement = (
            discr.bound_displacement_cell @ displacement
            + discr.bound_displacement_face @ bc
        )

        boundary_displacement.set_name("boundary_displacement")
        return boundary_displacement


class MyMomentumBalance(
    NewmarkConstants,
    BoundaryAndInitialCond,
    MyGeometry,
    SolutionStrategySourceBC,
    MyConstitutiveLaws,
    MyMomentumBalance,
):
    ...


t_shift = 0.0
tf = 0.8
dt = tf / 800.0

time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "density": 2425.0,
        "lame_lambda": 12.0e9,
        "permeability": 1.0,
        "shear_modulus": 4.5e9,
    }
)

material_constants = {"solid": solid_constants}
params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "material_constants": material_constants,
    "folder_name": "testing",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
}
model = MyMomentumBalance(params)


pp.run_time_dependent_model(model, params)

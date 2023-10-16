import porepy as pp
import numpy as np

from models import MomentumBalanceTimeDepSource

Scalar = pp.ad.Scalar


class BoundaryAndInitialCond:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        # Approximating the time derivative in the BCs and rewriting the BCs on "Robin
        # form" gives need for Robin weights.
        # The two following lines provide an array with 2 components. The first
        # component is a 2d array with ones in the first row and zeros in the second.
        # The second component is also a 2d array, but now the first row is zeros and
        # the second row is ones.
        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        value = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F")

        bounds = self.domain_boundary_sides(sd)

        robin_weight_shear = self.robin_weight_value(direction="shear")
        robin_weight_tensile = self.robin_weight_value(direction="tensile")

        # Assigning shear weight to the boundaries who have x-direction as shear
        # direction.
        value[0][0][
            bounds.north + bounds.south + bounds.bottom + bounds.top
        ] *= robin_weight_shear

        # Assigning tensile weight to the boundaries who have x-direction as tensile
        # direction.
        value[0][0][bounds.east + bounds.west] *= robin_weight_tensile

        # Assigning shear weight to the boundaries who have y-direction as shear
        # direction.
        value[1][1][
            bounds.east + bounds.west + bounds.bottom + bounds.top
        ] *= robin_weight_shear

        # Assigning tensile weight to the boundaries who have y-direction as tensile
        # direction.
        value[1][1][bounds.north + bounds.south] *= robin_weight_tensile

        if self.nd == 3:
            # Assigning shear weight to the boundaries who have z-direction as shear
            # direction.
            value[2][2][
                bounds.north + bounds.south + bounds.east + bounds.west
            ] *= robin_weight_shear

            # Assigning tensile weight to the boundaries who have z-direction as tensile
            # direction.
            value[2][2][bounds.top + bounds.bottom] *= robin_weight_tensile

        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north
            + bounds.south
            + bounds.east
            + bounds.west
            + bounds.bottom
            + bounds.top,
            "rob",
        )

        bc.robin_weight = value
        return bc

    def bc_values_robin(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Method for assigning Robin boundary condition values.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        face_areas = boundary_grid.cell_volumes
        data = self.mdg.boundary_grid_data(boundary_grid)

        values = np.zeros((self.nd, boundary_grid.num_cells))
        bounds = self.domain_boundary_sides(boundary_grid)

        if self.time_manager.time_index > 1:
            # "Get face displacement"-strategy: Create them using
            # bound_displacement_face/-cell from second timestep and ongoing.
            sd = boundary_grid.parent
            displacement_boundary_operator = self.boundary_displacement([sd])
            displacement_values = displacement_boundary_operator.evaluate(
                self.equation_system
            ).val

            displacement_values = (
                boundary_grid.projection(self.nd) @ displacement_values
            )

        elif self.time_manager.time_index == 1:
            # On first timestep, initial values are fetched from the data dictionary.
            # These initial values are assigned in the initial_condition function that
            # is called at the zeroth time step. The boundary displacement operator is
            # not available at this time.
            displacement_values = pp.get_solution_values(
                name="bc_robin", data=data, time_step_index=0
            )

        elif self.time_manager.time_index == 0:
            # The first time this method is called is on initialization of the boundary
            # values.
            return self.initial_condition_bc(boundary_grid)

        displacement_values = np.reshape(
            displacement_values, (self.nd, boundary_grid.num_cells), "F"
        )

        # Assigning the values like this is a very brute force way. A
        # tensor-normal-vector-product is considered as an altenative (but not
        # implemented yet)
        robin_weight_shear = self.robin_weight_value(direction="shear")
        robin_weight_tensile = self.robin_weight_value(direction="tensile")

        values[0][bounds.north] += (
            robin_weight_shear * displacement_values[0][bounds.north]
        ) * face_areas[bounds.north]
        values[1][bounds.north] += (
            robin_weight_tensile * displacement_values[1][bounds.north]
        ) * face_areas[bounds.north]

        values[0][bounds.south] += (
            robin_weight_shear * displacement_values[0][bounds.south]
        ) * face_areas[bounds.south]
        values[1][bounds.south] += (
            robin_weight_tensile * displacement_values[1][bounds.south]
        ) * face_areas[bounds.south]

        values[0][bounds.east] += (
            robin_weight_tensile * displacement_values[0][bounds.east]
        ) * face_areas[bounds.east]
        values[1][bounds.east] += (
            robin_weight_shear * displacement_values[1][bounds.east]
        ) * face_areas[bounds.east]

        values[0][bounds.west] += (
            robin_weight_tensile * displacement_values[0][bounds.west]
        ) * face_areas[bounds.west]
        values[1][bounds.west] += (
            robin_weight_shear * displacement_values[1][bounds.west]
        ) * face_areas[bounds.west]

        if self.nd == 3:
            values[2][bounds.north] += (
                robin_weight_shear * displacement_values[2][bounds.north]
            ) * face_areas[bounds.north]

            values[2][bounds.south] += (
                robin_weight_shear * displacement_values[2][bounds.south]
            ) * face_areas[bounds.south]

            values[2][bounds.east] += (
                robin_weight_shear * displacement_values[2][bounds.east]
            ) * face_areas[bounds.east]

            values[2][bounds.west] += (
                robin_weight_shear * displacement_values[2][bounds.west]
            ) * face_areas[bounds.west]

            values[0][bounds.top] += (
                robin_weight_shear * displacement_values[0][bounds.top]
            ) * face_areas[bounds.top]
            values[1][bounds.top] += (
                robin_weight_shear * displacement_values[1][bounds.top]
            ) * face_areas[bounds.top]
            values[2][bounds.top] += (
                robin_weight_tensile * displacement_values[2][bounds.top]
            ) * face_areas[bounds.top]

            values[0][bounds.bottom] += (
                robin_weight_shear * displacement_values[0][bounds.bottom]
            ) * face_areas[bounds.bottom]
            values[1][bounds.bottom] += (
                robin_weight_shear * displacement_values[1][bounds.bottom]
            ) * face_areas[bounds.bottom]
            values[2][bounds.bottom] += (
                robin_weight_tensile * displacement_values[2][bounds.bottom]
            ) * face_areas[bounds.bottom]

        return values.ravel("F")

    def initial_condition_bc(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros((self.nd, bg.num_cells))


class SolutionStrategyABC:
    def _is_nonlinear_problem(self) -> bool:
        return True


class ConstitutiveLawsABC:
    def robin_weight_value(self, direction: str, side: str = None) -> float:
        """Shear Robin weight for Robin boundary conditions.

        Parameters:
            direction: Whether the boundary condition that uses the weight is the shear
                or tensile component of the displacement.
            side: Which boundary side is considered. This alters the sign of the
                weight. To be deprecated.

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
        elif direction == "tensile":
            value = (lam + 2 * mu) / (cp * dt)
        return value

    def boundary_displacement(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Method for reconstructing the boundary displacement.

        Usage within the "realm" of absorbing boundary conditions: we need the
        displacement values on the boundary at the previous time step.

        Note: This is for the pure mechanical problem - even without faults.
            Modifications are needed when a coupling to fluid flow is introduced at some
            later point.

        Parameters:
            subdomains: List of subdomains. Should be of co-dimension 0.

        Returns:
            Ad operator representing the displacement on grid faces of the subdomains.

        """
        # Discretization
        discr = self.stress_discretization(subdomains)

        # Boundary conditions on external boundaries
        # bc = self.bc_values_mechanics(subdomains)
        bc = self._combine_boundary_operators(  # type: ignore [call-arg]
            subdomains=subdomains,
            dirichlet_operator=self.displacement,
            neumann_operator=self.mechanical_stress,
            bc_type=self.bc_type_mechanics,
            robin_operator=lambda bgs: self.create_boundary_operator(
                name=self.bc_robin_key, domains=bgs
            ),
            dim=self.nd,
            name="bc_values_mechanics",
        )
        # Displacement
        displacement = self.displacement(subdomains)

        boundary_displacement = (
            discr.bound_displacement_cell @ displacement
            + discr.bound_displacement_face @ bc
        )

        boundary_displacement.set_name("boundary_displacement")
        return boundary_displacement


class MomentumBalanceABC(
    BoundaryAndInitialCond,
    SolutionStrategyABC,
    ConstitutiveLawsABC,
    MomentumBalanceTimeDepSource,
):
    ...

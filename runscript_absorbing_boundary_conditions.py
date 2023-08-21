import porepy as pp
import numpy as np

from models import MomentumBalanceABC

from utils import get_solution_values


class DifferentBC:
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

        value[1][1][bounds.east] *= self.robin_weight_value(
            direction="shear", side="east"
        )
        value[0][0][bounds.east] *= self.robin_weight_value(
            direction="tensile", side="east"
        )

        # Choosing type of boundary condition for the different domain sides.
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "rob",
        )
        bc.is_rob[:, bounds.west] = False
        bc.is_rob[:, bounds.north] = False
        bc.is_rob[:, bounds.south] = False
        # bc.is_rob[:, bounds.east] = False

        bc.is_dir[:, bounds.west] = True
        bc.is_neu[:, bounds.north] = True
        bc.is_neu[:, bounds.south] = True
        # bc.is_dir[:, bounds.east] = True

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
        t = self.time_manager.time

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

        # Zero Neumann on top - Waves are just allowed to slide alongside the
        # boundaries. Strictly not necessary to have it here, as the values are zero by
        # default. But easier to modify the value later then.
        values[0][bounds.north] = np.zeros(len(displacement_values[0][bounds.north]))
        values[1][bounds.north] = np.zeros(len(displacement_values[1][bounds.north]))

        values[0][bounds.south] = np.zeros(len(displacement_values[0][bounds.south]))
        values[1][bounds.south] = np.zeros(len(displacement_values[1][bounds.south]))

        # Value for the absorbing boundary
        values[1][bounds.east] += (
            self.robin_weight_value(direction="shear", side="east")
            * displacement_values[1][bounds.east]
        )
        values[0][bounds.east] += (
            self.robin_weight_value(direction="tensile", side="east")
            * displacement_values[0][bounds.east]
        )
        values[0][bounds.west] += np.ones(
            len(displacement_values[0][bounds.west])
        ) * np.sin(t)

        return values.ravel("F")


class MyGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 40.0 / self.units.m
        y = 30.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 1.0 / self.units.m}
        return mesh_args


class MomentumBalanceABCModifiedGeometry(DifferentBC, MyGeometry, MomentumBalanceABC):
    ...


t_shift = 0.0
tf = 12.0
dt = tf / 1200.0


time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "density": 1.0,
        "lame_lambda": 1.0e1,
        "shear_modulus": 1.0e1,
    }
)

material_constants = {"solid": solid_constants}

params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "material_constants": material_constants,
    "folder_name": "testing_4",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
}

model = MomentumBalanceABCModifiedGeometry(params)
pp.run_time_dependent_model(model, params)

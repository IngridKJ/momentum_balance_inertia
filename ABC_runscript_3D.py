import porepy as pp

import numpy as np

from models import MomentumBalanceABC
from utils import get_solution_values


class BoundaryCondUnit3D:
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

        value[0][0][bounds.east] *= self.robin_weight_value(
            direction="tensile", side="east"
        )

        bc = pp.BoundaryConditionVectorial(sd, bounds.east, "rob")

        bc.is_neu[:, bounds.west + bounds.east] = False

        bc.is_dir[:, bounds.west] = True
        bc.is_neu[:, bounds.north + bounds.south + bounds.top + bounds.bottom] = True

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
        face_areas = sd.face_areas
        data = self.mdg.subdomain_data(sd)

        values = np.zeros((self.nd, sd.num_faces))
        bounds = self.domain_boundary_sides(sd)

        if self.time_manager.time_index > 1:
            # "Get face displacement"-strategy: Create them using
            # bound_displacement_face/-cell from second timestep and ongoing.
            displacement_boundary_operator = self.boundary_displacement([sd])
            displacement_values = displacement_boundary_operator.evaluate(
                self.equation_system
            ).val

        else:
            # On first timestep, initial values are fetched from the data dictionary.
            # These initial values are assigned in the initial_condition function.
            displacement_values = get_solution_values(
                name=self.bc_values_mechanics_key, data=data, time_step_index=0
            )

        displacement_values = np.reshape(
            displacement_values, (self.nd, sd.num_faces), "F"
        )

        values[0][bounds.west] += np.ones(
            len(displacement_values[0][bounds.west])
        ) * np.sin(self.time_manager.time)

        values[0][bounds.east] += (
            self.robin_weight_value(direction="tensile", side="east")
            * displacement_values[0][bounds.east]
        ) * face_areas[bounds.east]
        return values.ravel("F")


class MyGeometry:
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 30.0 / self.units.m
        y = 1.0 / self.units.m
        z = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.25 / self.units.m}
        return mesh_args


class MomentumBalanceABCModifiedGeometry(
    BoundaryCondUnit3D,
    MyGeometry,
    MomentumBalanceABC,
):
    ...


t_shift = 0.0
tf = 10.0
dt = tf / 1000.0


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
    "folder_name": "meh",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
}

model = MomentumBalanceABCModifiedGeometry(params)
pp.run_time_dependent_model(model, params)

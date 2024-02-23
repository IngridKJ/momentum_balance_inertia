import porepy as pp
import numpy as np

from models import MomentumBalanceABC1

from utils import InnerDomainVTIStiffnessTensorMixin

with open("energy_vals.txt", "w") as file:
    pass


class ModifiedBoundaryConditions:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        value = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F")

        bounds = self.domain_boundary_sides(sd)

        robin_weight_shear = self.robin_weight_value(direction="shear")
        robin_weight_tensile = self.robin_weight_value(direction="tensile")

        # Assigning shear weight to the boundaries who have x-direction as shear
        # direction.
        value[0][0][bounds.north + bounds.south + bounds.bottom] *= robin_weight_shear

        # Assigning tensile weight to the boundaries who have x-direction as tensile
        # direction.
        value[0][0][bounds.east + bounds.west] *= robin_weight_tensile

        # Assigning shear weight to the boundaries who have y-direction as shear
        # direction.
        value[1][1][
            bounds.east + bounds.bottom + bounds.top + bounds.west
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
            value[2][2][bounds.bottom] *= robin_weight_tensile

        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.bottom + bounds.west,
            "rob",
        )

        bc.is_neu[:, bounds.top] = False

        # Top boundary is Dirichlet
        bc.is_dir[:, bounds.top] = True
        bc.robin_weight = value
        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        t = self.time_manager.time

        displacement_values = np.zeros((self.nd, bg.num_cells))

        # Time dependent sine Dirichlet condition
        values[2][bounds.top] += np.ones(
            len(displacement_values[0][bounds.top])
        ) * np.sin(t)

        return values.ravel("F")


class ModifiedGeometry:
    def nd_rect_domain(self, x, y, z) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y, "zmin": 0, "zmax": z})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 50.0 / self.units.m
        y = 50.0 / self.units.m
        z = 30.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y, z)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 5.0 / self.units.m}
        return mesh_args


class EntireAnisotropy3DModel(
    ModifiedBoundaryConditions,
    ModifiedGeometry,
    InnerDomainVTIStiffnessTensorMixin,
    MomentumBalanceABC1,
):
    def data_to_export(self):
        """Define the data to export to vtu.

        Returns:
            list: List of tuples containing the subdomain, variable name,
            and values to export.

        """
        data = super().data_to_export()
        for sd in self.mdg.subdomains(dim=self.nd):
            vel_op = self.velocity_time_dep_array([sd]) * self.velocity_time_dep_array(
                [sd]
            )
            vel_op_int = self.volume_integral(integrand=vel_op, grids=[sd], dim=3)
            vel_op_int_val = vel_op_int.value(self.equation_system)

            vel = self.velocity_time_dep_array([sd]).value(self.equation_system)

            data.append((sd, "energy", vel_op_int_val))
            data.append((sd, "velocity", vel))

            with open("energy_vals.txt", "a") as file:
                file.write(f"{np.sum(vel_op_int_val)},")

        return data


t_shift = 0.0
tf = 15.0
time_steps = 10.0
dt = tf / time_steps


time_manager = pp.TimeManager(
    schedule=[0.0 + t_shift, tf + t_shift],
    dt_init=dt,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

anisotropy_constants = {
    "mu_parallel": 1,
    "mu_orthogonal": 1,
    "lambda_parallel": 5,
    "lambda_orthogonal": 5,
    "volumetric_compr_lambda": 5,
}


params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "folder_name": "visualization",
    "manufactured_solution": "simply_zero",
    "inner_domain_width": 5,
    "progressbars": True,
    "anisotropy_constants": anisotropy_constants,
}

model = EntireAnisotropy3DModel(params)

pp.run_time_dependent_model(model=model, params=params)

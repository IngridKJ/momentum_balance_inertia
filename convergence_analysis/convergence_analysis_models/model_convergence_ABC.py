"""Model class setup for the convergence analysis of MPSA-Newmark with absorbing
boundaries.

All off-diagonal/shear components of the stiffness tensor are discarded.

"""

import sys

import numpy as np
import porepy as pp
import sympy as sym

sys.path.append("../../")

from models import DynamicMomentumBalanceABCLinear
from utils import u_v_a_wrap
from utils.anisotropy_mixins import TransverselyIsotropicTensorMixin
import utils


from porepy.applications.convergence_analysis import ConvergenceAnalysis


class RandomProperties:
    @property
    def heterogeneity_location(self):
        return self.params.get("heterogeneity_location", 0.5)

    @property
    def heterogeneity_factor(self):
        return self.params.get("heterogeneity_factor", 1.0)


class Geometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}
        box.update({"ymin": 0, "ymax": y})
        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def set_polygons(self):
        if type(self.heterogeneity_location) is list:
            L = self.heterogeneity_location[0]
            W = self.heterogeneity_location[1]
        else:
            L = self.heterogeneity_location
            W = self.domain.bounding_box["xmax"]

        H = self.domain.bounding_box["ymax"]
        west = np.array([[L, L], [0.0, H]])
        north = np.array([[L, W], [H, H]])
        east = np.array([[W, W], [H, 0.0]])
        south = np.array([[W, L], [0.0, 0.0]])
        return west, north, east, south

    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        west, north, east, south = self.set_polygons()

        self._fractures = [
            pp.LineFracture(west),
            pp.LineFracture(north),
            pp.LineFracture(east),
            pp.LineFracture(south),
        ]


class BoundaryConditions:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Method for assigning boundary condition type.

        Assigns the following boundary condition types:
            * North and south: Roller boundary (Neumann in x-direction and Dirichlet in
              y-direction)
            * West: Dirichlet
            * East: Robin

        Parameters:
            sd: The subdomain whose bc type is assigned.

        Return:
            The boundary condition object.

        """
        # Fetch boundary sides and assign type of boundary condition for the different
        # sides
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.all_bf, "dir")

        # East side: Absorbing
        bc.is_dir[:, bounds.east] = False
        bc.is_rob[:, bounds.east] = True

        # North side: Roller
        bc.is_dir[0, bounds.north + bounds.south] = False
        bc.is_neu[0, bounds.north + bounds.south] = True

        # Calling helper function for assigning the Robin weight
        self.assign_robin_weight(sd=sd, bc=bc)
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Method for assigning Neumann and Robin boundary condition values.

        Specifically, for the Robin values, this method assigns the values corresponding
        to absorbing boundary conditions with a second order approximation of u_t in:

            sigma * n + alpha * u_t = G

        Robin/Absorbing boundaries are employed for the east and west boundary. Zero
        Neumann conditions are assigned for the north and south boundary.

        Parameters:
            boundary_grid: The boundary grids on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        if boundary_grid.dim != (self.nd - 1):
            return np.array([])

        data = self.mdg.boundary_grid_data(boundary_grid)
        sd = boundary_grid.parent
        boundary_faces = sd.get_boundary_faces()
        name = "boundary_displacement_values"

        if self.time_manager.time_index == 0:
            return self.initial_condition_bc(boundary_grid)

        displacement_values_0 = pp.get_solution_values(
            name=name, data=data, time_step_index=0
        )
        displacement_values_1 = pp.get_solution_values(
            name=name, data=data, time_step_index=1
        )

        # According to the expression for the absorbing boundaries we have a coefficient
        # 2 in front of the values u_(n-1) and -0.5 in front of u_(n-2):
        displacement_values = 2 * displacement_values_0 - 0.5 * displacement_values_1

        # Transposing and reshaping displacement values to prepare for broadcasting
        displacement_values = displacement_values.reshape(
            boundary_grid.num_cells, self.nd, 1
        )

        # Assembling the vector representing the RHS of the Robin conditions
        total_coefficient_matrix = self.total_coefficient_matrix(sd=sd)
        robin_rhs = np.matmul(total_coefficient_matrix, displacement_values).squeeze(-1)
        robin_rhs *= sd.face_areas[boundary_faces][:, None]
        robin_rhs = robin_rhs.T

        boundary_sides = self.domain_boundary_sides(sd)
        for direction in ["north", "south"]:
            inds = np.where(getattr(boundary_sides, direction))[0]
            inds = np.where(np.isin(boundary_faces, inds))[0]
            robin_rhs[:, inds] *= 0

        return robin_rhs.ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Method for setting Dirichlet boundary values.

        Sets a time dependent sine condition in the x-direction of the western boundary.
        Zero elsewhere.

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the displacement boundary values at time t.

        """
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)
        t = self.time_manager.time

        # Time dependent sine Dirichlet condition
        values[0][bounds.west] += np.ones(len(values[0][bounds.west])) * np.sin(t)

        return values.ravel("F")

    def initial_condition_bc(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Method for setting initial boundary values for 0th and -1st time step.

        Parameters:
            bg: Boundary grid whose boundary displacement value is to be set.

        Returns:
            An array with the initial displacement boundary values.

        """
        dt = self.time_manager.dt
        vals_0 = self.initial_condition_value_function(bg=bg, t=0)
        vals_1 = self.initial_condition_value_function(bg=bg, t=0 - dt)

        data = self.mdg.boundary_grid_data(bg)

        # The values for the 0th and -1th time step are to be stored
        pp.set_solution_values(
            name="boundary_displacement_values",
            values=vals_0,
            data=data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="boundary_displacement_values",
            values=vals_1,
            data=data,
            time_step_index=1,
        )
        return vals_0

    def initial_condition_value_function(
        self, bg: pp.BoundaryGrid, t: float
    ) -> np.ndarray:
        """Initial values for the absorbing boundaries.

        In the quasi-1d test we have to assign initial values to the east and west
        boundaries (absorbing boundaries).

        """
        sd = bg.parent

        x = sd.face_centers[0, :]
        y = sd.face_centers[1, :]

        boundary_sides = self.domain_boundary_sides(sd)
        inds_east = np.where(boundary_sides.east)[0]
        inds_west = np.where(boundary_sides.west)[0]

        bc_vals = np.zeros((sd.dim, sd.num_faces))

        displacement_function = u_v_a_wrap(model=self)

        # East
        bc_vals[0, :][inds_east] = displacement_function[0](
            x[inds_east], y[inds_east], t
        )

        # West
        bc_vals[0, :][inds_west] = displacement_function[0](
            x[inds_west], y[inds_west], t
        )

        bc_vals = bg.projection(self.nd) @ bc_vals.ravel("F")

        return bc_vals


class ConstitutiveLawsAndSource:
    def elastic_force(self, sd, sigma_total, time: float) -> np.ndarray:
        """Evaluate exact elastic force [N] at the face centers for a quasi-1D setting.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_faces, )`` containing the exact elastic force
            at the face centers for the given ``time``.

        """
        # Symbolic variables
        x, y, t = sym.symbols("x y t")

        # Get cell centers and face normals
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression
        sigma_total_fun = [
            [
                sym.lambdify((x, y, t), sigma_total[0][0], "numpy"),
                sym.lambdify((x, y, t), sigma_total[0][1], "numpy"),
            ],
            [
                sym.lambdify((x, y, t), sigma_total[1][0], "numpy"),
                sym.lambdify((x, y, t), sigma_total[1][1], "numpy"),
            ],
        ]

        # Face-centered elastic force, but in a quasi 1d setting (y-components are zero)
        force_total_fc: list[np.ndarray] = [
            # (sigma_xx * n_x + sigma_xy * n_y) * face_area
            sigma_total_fun[0][0](fc[0], fc[1], time) * fn[0]
            + sigma_total_fun[0][1](fc[0], fc[1], time) * fn[1],
            # (sigma_yx * n_x + sigma_yy * n_y) * face_area
            sigma_total_fun[1][0](fc[0], fc[1], time) * fn[0]
            + sigma_total_fun[1][1](fc[0], fc[1], time) * fn[1],
        ]

        # Flatten array
        force_total_flat: np.ndarray = np.asarray(force_total_fc).ravel("F")
        return force_total_flat

    def evaluate_mechanics_source(self, f: list, sd: pp.Grid, t: float) -> np.ndarray:
        vals = np.zeros((self.nd, sd.num_cells))
        return vals.ravel("F")


class ExportData:
    def data_to_export(self):
        data = super().data_to_export()
        if self.time_manager.final_time_reached():
            self.compute_and_save_errors(filename=self.filename_path)
        return data

    def compute_and_save_errors(self, filename: str) -> None:
        sd = self.mdg.subdomains(dim=self.nd)[0]
        x_cc = sd.cell_centers[0, :]
        time = self.time_manager.time
        cp = self.primary_wave_speed(is_scalar=True)

        # Exact displacement and traction
        u_exact = np.array([np.sin(time - x_cc / cp), np.zeros(len(x_cc))]).ravel("F")

        u, x, y, t = utils.symbolic_representation(model=self)
        _, sigma, _ = utils.symbolic_equation_terms(model=self, u=u, x=x, y=y, t=t)
        T_exact = self.elastic_force(
            sd=sd, sigma_total=sigma, time=self.time_manager.time
        )

        # Approximated displacement and traction
        displacement_ad = self.displacement([sd])
        u_approximate = self.equation_system.evaluate(displacement_ad)
        traction_ad = self.stress([sd])
        T_approximate = self.equation_system.evaluate(traction_ad)

        # Compute error for displacement and traction
        error_displacement = ConvergenceAnalysis.lp_error(
            grid=sd,
            true_array=u_exact,
            approx_array=u_approximate,
            is_scalar=False,
            is_cc=True,
            relative=True,
        )
        error_traction = ConvergenceAnalysis.lp_error(
            grid=sd,
            true_array=T_exact,
            approx_array=T_approximate,
            is_scalar=False,
            is_cc=False,
            relative=True,
        )

        with open(filename, "a") as file:
            file.write(
                f"{sd.num_cells}, {self.time_manager.time_index}, {error_displacement}, {error_traction}\n"
            )


class ABCModel(
    RandomProperties,
    Geometry,
    BoundaryConditions,
    ConstitutiveLawsAndSource,
    TransverselyIsotropicTensorMixin,
    ExportData,
    DynamicMomentumBalanceABCLinear,
): ...

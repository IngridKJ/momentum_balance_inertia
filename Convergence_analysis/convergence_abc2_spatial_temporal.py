"""This file is a model setup and runscript for the unit test. Relevant commits:
* Porepy:  develop
* Momentum balance inertia d33a1a9d

Figure out what has happened in between those commits and now (mid June)
"""

import sys

import numpy as np
import porepy as pp
import sympy as sym

sys.path.append("../")

from models import DynamicMomentumBalanceABC2
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from utils import get_boundary_cells, u_v_a_wrap
import utils


class BoundaryConditionsUnitTest:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for the absorbing boundary condition model class.

        Assigns Robin boundaries to all subdomain boundaries. This includes setting the
        Robin weight.

        """
        # Fetch boundary sides and assign type of boundary condition for the different
        # sides
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.east + bounds.west, "rob")

        # Only the eastern boundary will be Robin (absorbing)
        bc.is_rob[:, bounds.west] = False

        # Western boundary is Dirichlet
        bc.is_dir[:, bounds.west] = True

        # Calling helper function for assigning the Robin weight
        self.assign_robin_weight(sd=sd, bc=bc)
        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        t = self.time_manager.time

        displacement_values = np.zeros((self.nd, bg.num_cells))

        # Time dependent sine Dirichlet condition
        values[0][bounds.west] += np.ones(
            len(displacement_values[0][bounds.west])
        ) * np.sin(t)

        return values.ravel("F")

    def initial_condition_bc(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Method for setting initial values for 0th and -1st time step in
        dictionary.

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

    def initial_condition_value_function(self, bg, t):
        sd = bg.parent

        x = sd.face_centers[0, :]
        y = sd.face_centers[1, :]

        inds_east = get_boundary_cells(self=self, sd=sd, side="east", return_faces=True)
        inds_west = get_boundary_cells(self=self, sd=sd, side="west", return_faces=True)

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


class ConstitutiveLawUnitTest:
    def stiffness_tensor(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa].

        Parameters:
            subdomain: Subdomain where the stiffness tensor is defined.

        Returns:
            Cell-wise stiffness tensor in SI units.

        """
        lmbda = self.solid.lame_lambda() * np.ones(subdomain.num_cells)
        mu = self.solid.shear_modulus() * np.ones(subdomain.num_cells)
        return FourthOrderTensorUnitTest(mu, lmbda)

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
        return vals.ravel("F")


class FourthOrderTensorUnitTest(object):
    """Cell-wise representation of fourth order tensor, represented by (3^2, 3^2 ,Nc)-array, where Nc denotes the number of cells, i.e. the tensor values are stored discretely.

    For each cell, there are dim^4 degrees of freedom, stored in a 3^2 * 3^2 matrix.

    The only constructor available for the moment is based on the Lame parameters, e.g.
    using two degrees of freedom.

    Attributes:
        values: dimensions (3^2, 3^2, nc), cell-wise representation of
            the stiffness matrix.
        lmbda (np.ndarray): Nc array of first Lame parameter
        mu (np.ndarray): Nc array of second Lame parameter

    """

    def __init__(self, mu: np.ndarray, lmbda: np.ndarray):
        """Constructor for fourth order tensor on Lame-parameter form

        Parameters:
            mu: Nc array of shear modulus (second lame parameter).
            lmbda: Nc array of first lame parameter.

        Raises:
            ValueError if mu or lmbda are not 1d arrays.
            ValueError if the lengths of mu and lmbda are not matching.
        """

        if not isinstance(mu, np.ndarray):
            raise ValueError("Input mu should be a numpy array")
        if not isinstance(lmbda, np.ndarray):
            raise ValueError("Input lmbda should be a numpy array")
        if not mu.ndim == 1:
            raise ValueError("mu should be 1-D")
        if not lmbda.ndim == 1:
            raise ValueError("Lmbda should be 1-D")
        if mu.size != lmbda.size:
            raise ValueError("Mu and lmbda should have the same length")

        # Save lmbda and mu, can be useful to have in some cases
        self.lmbda = lmbda
        self.mu = mu

        # Basis for the contributions of mu, lmbda and phi is hard-coded
        mu_mat = np.array(
            [
                [2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2],
            ]
        )
        lmbda_mat = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Expand dimensions to prepare for cell-wise representation
        mu_mat = mu_mat[:, :, np.newaxis]
        lmbda_mat = lmbda_mat[:, :, np.newaxis]

        c = mu_mat * mu + lmbda_mat * lmbda
        self.values = c

    def copy(self) -> "FourthOrderTensorUnitTest":
        """`
        Define a deep copy of the tensor.

        Returns:
            FourthOrderTensor: New tensor with identical fields, but separate
                arrays (in the memory sense).
        """
        C = FourthOrderTensorUnitTest(mu=self.mu, lmbda=self.lmbda)
        C.values = self.values.copy()
        return C


class ABC2Model(
    BoundaryConditionsUnitTest,
    ConstitutiveLawUnitTest,
    DynamicMomentumBalanceABC2,
):
    def elastic_force(self, sd, sigma_total, time: float) -> np.ndarray:
        """Evaluate exact elastic force [N] at the face centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_faces, )`` containing the exact ealstic
            force at the face centers for the given ``time``.

        Notes:
            - The returned elastic force is given in PorePy's flattened vector
              format.
            - Recall that force = (stress dot_prod unit_normal) * face_area.

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

        # Face-centered elastic force
        force_total_fc: list[np.ndarray] = [
            # (sigma_xx * n_x + sigma_xy * n_y) * face_area
            sigma_total_fun[0][0](fc[0], fc[1], time) * fn[0]
            + 0 * sigma_total_fun[0][1](fc[0], fc[1], time) * fn[1],
            # (sigma_yx * n_x + sigma_yy * n_y) * face_area
            sigma_total_fun[1][0](fc[0], fc[1], time) * fn[0]
            + 0 * sigma_total_fun[1][1](fc[0], fc[1], time) * fn[1],
        ]

        # Flatten array
        force_total_flat: np.ndarray = np.asarray(force_total_fc).ravel("F")

        return force_total_flat


class MyUnitGeometry:
    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {
            "cell_size": 0.25 / 2 ** (self.refinement) / self.units.m
        }
        return mesh_args


class SpatialRefinementModel(MyUnitGeometry, ABC2Model):
    def data_to_export(self):
        """Define the data to export to vtu.

        Returns:
            list: List of tuples containing the subdomain, variable name,
            and values to export.

        """
        data = super().data_to_export()
        for sd in self.mdg.subdomains(dim=self.nd):
            x = sd.cell_centers[0, :]
            t = self.time_manager.time
            cp = self.primary_wave_speed(is_scalar=True)

            d = self.mdg.subdomain_data(sd)

            u_e = np.sin(t - x / cp)
            u_h = pp.get_solution_values(name="u", data=d, time_step_index=0)

            u_h = np.reshape(u_h, (self.nd, sd.num_cells), "F")[0]

            error = ConvergenceAnalysis.l2_error(
                grid=sd,
                true_array=u_e,
                approx_array=u_h,
                is_scalar=True,
                is_cc=True,
                relative=True,
            )

            data.append((sd, "analytical", u_e))
            data.append((sd, "diff", u_h - u_e))
            with open(f"error_{self.refinement}_spatial_temporal.txt", "a") as file:
                file.write(f"{error},")

            u, x, y, t = utils.symbolic_representation(model=self)
            _, sigma, _ = utils.symbolic_equation_terms(model=self, u=u, x=x, y=y, t=t)

            exact_elastic_force = self.elastic_force(
                sd=sd, sigma_total=sigma, time=self.time_manager.time
            )

            force_ad = self.stress([sd])
            approx_force = force_ad.value(self.equation_system)

            error_force = ConvergenceAnalysis.l2_error(
                grid=sd,
                true_array=exact_elastic_force,
                approx_array=approx_force,
                is_scalar=False,
                is_cc=False,
                relative=True,
            )
            with open(f"force_errors_{self.refinement}.txt", "a") as file:
                file.write(f"{error_force},")
        print("u:", error)
        print("T:", error_force, sd.num_cells)
        return data


refinements = np.array([0, 1, 2, 3, 4])


def read_float_values(filename) -> np.ndarray:
    with open(filename, "r") as file:
        content = file.read()
        numbers = np.array([float(num) for num in content.split(",")[:-1]])
        return numbers


with open(f"spatial_temporal_refinement.txt", "w") as file:
    pass

for refinement_coefficient in refinements:
    tf = 15.0
    time_steps = 15 * (2**refinement_coefficient)
    dt = tf / time_steps

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
    )
    # Unit square:
    solid_constants = pp.SolidConstants({"lame_lambda": 0.01, "shear_modulus": 0.01})
    material_constants = {"solid": solid_constants}

    params = {
        "time_manager": time_manager,
        "grid_type": "simplex",
        "folder_name": "testing_diag_wave",
        "manufactured_solution": "unit_test",
        "progressbars": True,
        "material_constants": material_constants,
    }
    model = SpatialRefinementModel(params)
    model.refinement = refinement_coefficient
    with open(f"error_{model.refinement}_spatial_temporal.txt", "w") as file:
        pass

    pp.run_time_dependent_model(model, params)
    error_value = read_float_values(
        filename=f"error_{model.refinement}_spatial_temporal.txt"
    )[-1]
    error_value_force = read_float_values(
        filename=f"force_errors_{model.refinement}.txt"
    )[-1]
    with open(f"spatial_temporal_refinement.txt", "a") as file:
        file.write(
            f"\nRefinement coefficient: {refinement_coefficient}. Error value: {error_value}\nForce error value: {error_value_force}\n"
        )

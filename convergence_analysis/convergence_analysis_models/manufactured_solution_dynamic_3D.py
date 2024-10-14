"""Manufactured solution setup for a 3D elastic wave equation.

The setup is an adaptation/extension of a similar setup for poromechanics found within
PorePy.

"""

from __future__ import annotations

import sys

sys.path.append("../../")
from dataclasses import dataclass
from typing import Callable

import numpy as np
import porepy as pp
import sympy as sym
from models import DynamicMomentumBalanceABC2Linear
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.utils.examples_utils import VerificationUtils
from porepy.viz.data_saving_model_mixin import VerificationDataSaving
from utils import symbolic_equation_terms, symbolic_representation

# PorePy typings
number = pp.number
grid = pp.GridLike


# -----> Data-saving
@dataclass
class ManuMechSaveData:
    """Data class to save relevant results from the verification setup."""

    approx_displacement: np.ndarray
    """Numerical displacement."""

    approx_force: np.ndarray
    """Numerical elastic force."""

    error_displacement: number
    """L2-discrete relative error for the displacement."""

    error_force: number
    """L2-discrete relative error for the elastic force."""

    exact_displacement: np.ndarray
    """Exact displacement."""

    exact_force: np.ndarray
    """Exact elastic force."""

    time: number
    """Current simulation time."""


class ManuMechDataSaving(VerificationDataSaving):
    """Mixin class to save relevant data."""

    displacement: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Displacement variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """

    exact_sol: ManuMechExactSolution3d
    """Exact solution object."""

    stress: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that returns the (integrated) elastic stress in the form of an Ad
    operator.

    """

    relative_l2_error: Callable
    """Method for computing the discrete relative L2-error. Normally provided by a
    mixin instance of :class:`~porepy.applications.building_blocks.
    verification_utils.VerificationUtils`.

    """

    def collect_data(self) -> ManuMechSaveData:
        """Collect data from the verification setup.

        Returns:
            ManuMechSaveData object containing the results of the verification for
            the current time.

        """

        mdg: pp.MixedDimensionalGrid = self.mdg
        sd: pp.Grid = mdg.subdomains()[0]
        t: number = self.time_manager.time

        # Collect data
        exact_displacement = self.exact_sol.displacement(sd=sd, time=t)
        displacement_ad = self.displacement([sd])
        approx_displacement = displacement_ad.value(self.equation_system)
        error_displacement = ConvergenceAnalysis.l2_error(
            grid=sd,
            true_array=exact_displacement,
            approx_array=approx_displacement,
            is_scalar=False,
            is_cc=True,
            relative=True,
        )

        exact_force = self.exact_sol.elastic_force(sd=sd, time=t)
        force_ad = self.stress([sd])
        approx_force = force_ad.value(self.equation_system)
        error_force = ConvergenceAnalysis.l2_error(
            grid=sd,
            true_array=exact_force,
            approx_array=approx_force,
            is_scalar=False,
            is_cc=False,
            relative=True,
        )

        # Store collected data in data class
        collected_data = ManuMechSaveData(
            approx_displacement=approx_displacement,
            approx_force=approx_force,
            error_displacement=error_displacement,
            error_force=error_force,
            exact_displacement=exact_displacement,
            exact_force=exact_force,
            time=t,
        )

        return collected_data


# -----> Exact solution
class ManuMechExactSolution3d:
    """Class containing the exact manufactured solution for the verification setup."""

    def __init__(self, setup):
        """Constructor of the class."""
        # Symbolic variables
        u, x, y, z, t = symbolic_representation(model=setup, is_2D=False)
        (
            source_mech,
            sigma_total,
            acceleration_term,
        ) = symbolic_equation_terms(model=setup, u=u, x=x, y=y, z=z, t=t, is_2D=False)

        # Public attributes
        self.u = u  # Displacement
        self.sigma_total = sigma_total  # Elastic stress
        self.source_mech = source_mech  # Source term entering the momentum balance
        self.acceleration_term = acceleration_term  # Acceleration term entering mom bal

    # -----> Primary and secondary variables
    def displacement(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Evaluate exact displacement [m] at the cell centers.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Array of ``shape=(2 * sd.num_cells, )`` containing the exact displacements
            at the cell centers for the given ``time``.

        Notes:
            The returned displacement is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers
        cc = sd.cell_centers

        # Lambdify expression
        u_fun: list[Callable] = [
            sym.lambdify((x, y, z, t), self.u[0], "numpy"),
            sym.lambdify((x, y, z, t), self.u[1], "numpy"),
            sym.lambdify((x, y, z, t), self.u[2], "numpy"),
        ]

        # Cell-centered displacements
        u_cc: list[np.ndarray] = [
            u_fun[0](cc[0], cc[1], cc[2], time),
            u_fun[1](cc[0], cc[1], cc[2], time),
            u_fun[2](cc[0], cc[1], cc[2], time),
        ]

        # Flatten array
        u_flat: np.ndarray = np.asarray(u_cc).ravel("F")

        return u_flat

    def elastic_force(self, sd: pp.Grid, time: float) -> np.ndarray:
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
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers and face normals
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression
        sigma_total_fun: list[list[Callable]] = [
            [
                sym.lambdify((x, y, z, t), self.sigma_total[0][0], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[0][1], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[0][2], "numpy"),
            ],
            [
                sym.lambdify((x, y, z, t), self.sigma_total[1][0], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[1][1], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[1][2], "numpy"),
            ],
            [
                sym.lambdify((x, y, z, t), self.sigma_total[2][0], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[2][1], "numpy"),
                sym.lambdify((x, y, z, t), self.sigma_total[2][2], "numpy"),
            ],
        ]

        # Face-centered elastic force
        force_total_fc: list[np.ndarray] = [
            # (sigma_xx * n_x + sigma_xy * n_y + sigma_xz * n_z) * face_area
            sigma_total_fun[0][0](fc[0], fc[1], fc[2], time) * fn[0]
            + sigma_total_fun[0][1](fc[0], fc[1], fc[2], time) * fn[1]
            + sigma_total_fun[0][2](fc[0], fc[1], fc[2], time) * fn[2],
            # (sigma_yx * n_x + sigma_yy * n_y + sigma_yz * n_z) * face_area
            sigma_total_fun[1][0](fc[0], fc[1], fc[2], time) * fn[0]
            + sigma_total_fun[1][1](fc[0], fc[1], fc[2], time) * fn[1]
            + sigma_total_fun[1][2](fc[0], fc[1], fc[2], time) * fn[2],
            # (sigma_zx * n_x + sigma_zy * n_y + sigma_zz * n_z) * face_area
            sigma_total_fun[2][0](fc[0], fc[1], fc[2], time) * fn[0]
            + sigma_total_fun[2][1](fc[0], fc[1], fc[2], time) * fn[1]
            + sigma_total_fun[2][2](fc[0], fc[1], fc[2], time) * fn[2],
        ]

        # Flatten array
        force_total_flat: np.ndarray = np.asarray(force_total_fc).ravel("F")

        return force_total_flat

    # -----> Sources
    def mechanics_source(self, sd: pp.Grid, time: float) -> np.ndarray:
        """Compute exact source term for the momentum balance equation.

        Parameters:
            sd: Subdomain grid.
            time: Time in seconds.

        Returns:
            Exact right hand side of the momentum balance equation with ``shape=(
            2 * sd.num_cells, )``.

        Notes:
            The returned array is given in PorePy's flattened vector format.

        """
        # Symbolic variables
        x, y, z, t = sym.symbols("x y z t")

        # Get cell centers and cell volumes
        cc = sd.cell_centers
        vol = sd.cell_volumes

        # Lambdify expression
        source_mech_fun: list[Callable] = [
            sym.lambdify((x, y, z, t), self.source_mech[0], "numpy"),
            sym.lambdify((x, y, z, t), self.source_mech[1], "numpy"),
            sym.lambdify((x, y, z, t), self.source_mech[2], "numpy"),
        ]

        # Evaluate and integrate source
        source_mech: list[np.ndarray] = [
            source_mech_fun[0](cc[0], cc[1], cc[2], time) * vol,
            source_mech_fun[1](cc[0], cc[1], cc[2], time) * vol,
            source_mech_fun[2](cc[0], cc[1], cc[2], time) * vol,
        ]

        # Flatten array
        source_mech_flat: np.ndarray = np.asarray(source_mech).ravel("F")

        return source_mech_flat


# -----> Utilities
class ManuMechUtils(VerificationUtils):
    """Mixin class containing useful utility methods for the setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    results: list[ManuMechSaveData]
    """List of ManuMechSaveData objects."""

    def plot_results(self) -> None:
        """Plotting results."""
        self._plot_displacement()

    def _plot_displacement(self):
        """Plot exact and numerical displacements."""

        sd = self.mdg.subdomains()[0]
        u_ex = self.results[-1].exact_displacement
        u_num = self.results[-1].approx_displacement

        # Horizontal displacement
        pp.plot_grid(sd, u_ex[::2], plot_2d=True, linewidth=0, title="u_x (Exact)")
        pp.plot_grid(sd, u_num[::2], plot_2d=True, linewidth=0, title="u_x (Numerical)")

        # Vertical displacement
        pp.plot_grid(sd, u_ex[1::2], plot_2d=True, linewidth=0, title="u_y (Exact)")
        pp.plot_grid(
            sd, u_num[1::2], plot_2d=True, linewidth=0, title="u_y (Numerical)"
        )

        # z displacement
        pp.plot_grid(sd, u_ex[2::3], plot_2d=True, linewidth=0, title="u_z (Exact)")
        pp.plot_grid(
            sd, u_num[2::3], plot_2d=True, linewidth=0, title="u_z (Numerical)"
        )


# -----> Geometry
class UnitSquareGrid:
    """Class for setting up the geometry of the unit square domain."""

    params: dict
    """Simulation model parameters."""

    def set_domain(self) -> None:
        """Set domain."""
        self._domain = nd_cube_domain(3, 1.0)

    def meshing_arguments(self) -> dict[str, float]:
        """Set meshing arguments."""
        default_mesh_arguments = {"cell_size": 0.1}
        return self.params.get("meshing_arguments", default_mesh_arguments)


# -----> Solution strategy
class ManuMechSolutionStrategy3d:
    """Solution strategy for the verification setup."""

    exact_sol: ManuMechExactSolution3d
    """Exact solution object."""

    plot_results: Callable
    """Method for plotting results. Usually provided by the mixin class
    :class:`SetupUtilities`.

    """

    results: list[ManuMechSaveData]
    """List of SaveData objects."""

    def __init__(self, params: dict):
        """Constructor for the class."""
        super().__init__(params)

        self.exact_sol: ManuMechExactSolution3d
        """Exact solution object."""

        self.results: list[ManuMechSaveData] = []
        """Results object that stores exact and approximated solutions and errors."""

        self.stress_variable: str = "elastic_force"
        """Keyword to access the elastic force."""

    def _is_time_dependent(self):
        return True

    def set_materials(self):
        """Set material parameters."""
        super().set_materials()

        # Instantiate exact solution object after materials have been set
        self.exact_sol = ManuMechExactSolution3d(self)

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()

        self.update_mechanics_source()

    def update_mechanics_source(self) -> None:
        """Update values of external sources."""
        sd = self.mdg.subdomains()[0]
        data = self.mdg.subdomain_data(sd)
        t = self.time_manager.time

        # Mechanics source
        mech_source = self.exact_sol.mechanics_source(sd=sd, time=t)

        pp.set_solution_values(
            name="source_mechanics", values=mech_source, data=data, iterate_index=0
        )

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params.get("plot_results", False):
            self.plot_results()


class ManuMechBoundaryConditions:
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north
            + bounds.south
            + bounds.east
            + bounds.west
            + bounds.top
            + bounds.bottom,
            "dir",
        )

        return bc


# -----> Mixer class
class ManuMechSetup3d(  # type: ignore[misc]
    UnitSquareGrid,
    ManuMechSolutionStrategy3d,
    ManuMechUtils,
    ManuMechDataSaving,
    ManuMechBoundaryConditions,
    DynamicMomentumBalanceABC2Linear,
):
    """
    Mixer class for the two-dimensional mechanics verification setup.

    """

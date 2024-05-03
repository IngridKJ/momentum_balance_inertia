"""This file contains a run_time_dependent_model function that should be used only for
linear problems where the Jacobian is not changed between time steps."""

from __future__ import annotations

import logging


from typing import Optional

from tqdm.autonotebook import trange

from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)

import numpy as np
import scipy.sparse as sps

import porepy as pp

logger = logging.getLogger(__name__)


class CustomLinearSolver:
    """Wrapper around models that solves linear problems, and calls the methods in the
    model class before and after solving the problem.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        """Define linear solver.

        Parameters:
            params (dict): Parameters for the linear solver. Will be passed on to the
                model class. Thus the contect should be adapted to whatever needed for
                the problem at hand.

        """
        if params is None:
            params = {}
        self.params = params

    def solve(self, setup: pp.SolutionStrategy) -> tuple[float, bool]:
        """Solve a linear problem defined by the current state of the model.

        Parameters:
            setup (subclass of pp.SolutionStrategy): Model to be solved.

        Returns:
            float: Norm of the error.
            boolean: True if the linear solver converged.

        """

        setup.before_nonlinear_loop()
        prev_sol = setup.equation_system.get_variable_values(time_step_index=0)

        # Only assemble the linear system once:
        if setup.time_manager.time_index == 1:
            setup.assemble_linear_system()

        sol = setup.solve_linear_system()

        error_norm, is_converged, _ = setup.check_convergence(
            sol, prev_sol, prev_sol, self.params
        )

        if is_converged:
            # IMPLEMENTATION NOTE: The following is a bit awkward, and really shows there is
            # something wrong with how the linear and non-linear solvers interact with the
            # models (and it illustrates that the model convention for the before_nonlinear_*
            # and after_nonlinear_* methods is not ideal).
            # Since the setup's after_nonlinear_convergence may expect that the converged
            # solution is already stored as an iterate (this may happen if a model is
            # implemented to be valid for both linear and non-linear problems, as is
            # the case for ContactMechanics and possibly others). Thus, we first call
            # after_nonlinear_iteration(), and then after_nonlinear_convergence()
            setup.after_nonlinear_iteration(sol)
            setup.after_nonlinear_convergence(sol, error_norm, iteration_counter=1)
        else:
            setup.after_nonlinear_failure(sol, error_norm, iteration_counter=1)
        return error_norm, is_converged


def _choose_solver(params: dict) -> CustomLinearSolver:
    solver = CustomLinearSolver(params)
    return solver


def run_time_dependent_model(model, params: dict) -> None:
    """Run a time dependent model."""
    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # When multiple nested ``tqdm`` bars are used, their position needs to be specified
    # such that they are displayed in the correct order. The orders are increasing, i.e.
    # 0 specifies the lowest level, 1 the next-lowest etc.
    # When the ``NewtonSolver`` is called inside ``run_time_dependent_model``, the
    # ``progress_bar_position`` parameter with the updated position of the progress bar
    # for the ``NewtonSolver`` is passed.
    params.update({"progress_bar_position": 1})

    # Assign a solver
    solver = _choose_solver(params)

    # Define a function that does all the work during one time step, except
    # for everything ``tqdm`` related.
    def time_step() -> None:
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        logger.info(
            f"\nTime step {model.time_manager.time_index} at time"
            + f" {model.time_manager.time:.1e}"
            + f" of {model.time_manager.time_final:.1e}"
            + f" with time step {model.time_manager.dt:.1e}"
        )
        solver.solve(model)

    # Redirect the root logger, s.t. no logger interferes with with the
    # progressbars.
    with logging_redirect_tqdm([logging.root]):
        # Time loop
        # Create a time bar. The length is estimated as the timesteps predetermined
        # by the schedule and initial time step size.
        # NOTE: If e.g. adaptive time stepping results in more time steps, the time
        # bar will increase with partial steps corresponding to the ratio of the
        # modified time step size to the initial time step size.
        expected_timesteps: int = int(
            np.round(
                (model.time_manager.schedule[-1] - model.time_manager.schedule[0])
                / model.time_manager.dt
            )
        )

        initial_time_step: float = model.time_manager.dt
        time_progressbar = trange(
            expected_timesteps,
            desc="time loop",
            position=0,
        )

        while not model.time_manager.final_time_reached():
            time_progressbar.set_description_str(
                f"Time step {model.time_manager.time_index} + 1"
            )
            time_step()
            # Update time progressbar length by the time step size divided by the
            # initial time step size.
            time_progressbar.update(n=model.time_manager.dt / initial_time_step)

    model.after_simulation()

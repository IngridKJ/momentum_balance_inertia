"""This file contains the necessary classes and functions to run the model class which
only assembles the Jacobian once.

Contents:
* The class defining the linear solver. Now the residual is fetched from the model
attribute linear_system_residual instead of constructed. It is constructed already in the "custom" solution strategy.
* The function choosing the solver now ghooses the "custom" linear solver defined in this file.
* A function for running the linear model. The same as pp.run_time_dependent_model(),
with the only difference being its name. Just needed a new function for it as it
utilizes _choose_solver(), and that method had changes to it.

"""

import logging

from typing import Optional, Union

import numpy as np
import porepy as pp

try:
    from porepy.utils.ui_and_logging import (
        logging_redirect_tqdm_with_level as logging_redirect_tqdm,
    )
    from tqdm.autonotebook import trange  # type: ignore

except ImportError:
    _IS_TQDM_AVAILABLE: bool = False
else:
    _IS_TQDM_AVAILABLE = True

logger = logging.getLogger(__name__)
from porepy.models.solution_strategy import SolutionStrategy


class LinearSolverModifiedResidualInstantiation:
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

    def solve(self, setup: SolutionStrategy) -> bool:
        """Solve a linear problem defined by the current state of the model.

        Parameters:
            setup (subclass of pp.SolutionStrategy): Model to be solved.

        Returns:
            boolean: True if the linear solver converged.

        """

        setup.before_nonlinear_loop()

        setup.assemble_linear_system()
        residual = setup.linear_system_residual
        nonlinear_increment = setup.solve_linear_system()

        _, _, is_converged, _ = setup.check_convergence(
            nonlinear_increment, residual, residual.copy(), self.params
        )

        if is_converged:
            setup.after_nonlinear_iteration(nonlinear_increment)
            setup.after_nonlinear_convergence()
        else:
            setup.after_nonlinear_failure()
        return is_converged


def _choose_solver(model, params: dict) -> Union[pp.LinearSolver, pp.NewtonSolver]:
    return LinearSolverModifiedResidualInstantiation(params)


def run_linear_model(model, params: dict) -> None:
    """"""
    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    params.update({"progress_bar_position": 1})

    # Assign a solver
    solver = _choose_solver(model, params)

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

    # Progressbars turned off or tqdm not installed:
    if not params.get("progressbars", False) or not _IS_TQDM_AVAILABLE:
        while not model.time_manager.final_time_reached():
            time_step()

    # Progressbars turned on:
    else:
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

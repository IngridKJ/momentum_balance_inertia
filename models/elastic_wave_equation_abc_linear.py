"""This file contains a model setup which only assembles the Jacobian once.

The default behavior when running porepy models is that the Jacobian is assembled at
every time step. This is not necessary for linear problems, where the Jacobian doesn't
change between time steps. Hence, this is a model class setup for the purpose of running
linear models:

The model inherits from the dynamic momentum balance with ABC2. A "custom" solution
strategy mixin is defined. This solution strategy mixin checks the time step to see if
only the residual, or both the residual and Jacobian should be assembled. In the case of
time_index > 1, the residual is constructed and the Jacobian is kept the same.
    
"""

from __future__ import annotations
import logging
import time

import numpy as np
import scipy.sparse as sps

from . import DynamicMomentumBalanceABC2

logger = logging.getLogger(__name__)


class SolutionStrategyAssembleLinearSystemOnce:
    def assemble_linear_system(self) -> None:
        """Assemble the linearized system and store it in :attr:`linear_system`.

        The linear system is defined by the current state of the model.

        """
        t_0 = time.time()
        if self.time_manager.time_index <= 1:
            ba = time.time()
            self.linear_system = self.equation_system.assemble()
            self.linear_system_jacobian = self.linear_system[0]
            self.linear_system_residual = self.linear_system[1]
            aa = time.time()
            print("Time assemble linear system:", aa - ba)
        else:
            ba = time.time()
            self.linear_system_residual = self.equation_system.assemble(
                evaluate_jacobian=False
            )
            aa = time.time()
            print("\nTime assemble residual:", aa - ba)
        logger.debug(f"Assembled linear system in {time.time() - t_0:.2e} seconds.")


class DynamicMomentumBalanceABC2Linear(
    SolutionStrategyAssembleLinearSystemOnce,
    DynamicMomentumBalanceABC2,
): ...

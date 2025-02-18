"""This file contains a model setup which facilitates only assembling the Jacobian once.

The default behavior when running PorePy models is that the Jacobian is assembled at
every time step. This is not necessary for linear problems, where the Jacobian doesn't
change between time steps. Hence, this is a model class setup with the purpose of
running linear models.
    
Note that the Jacobian must be constant throughout the simulation (linear problem) for
this simplification to be valid.

"""

import logging
import time

from . import DynamicMomentumBalanceABC

logger = logging.getLogger(__name__)


class SolutionStrategyAssembleLinearSystemOnce:
    def assemble_linear_system(self) -> None:
        """Assemble the linearized system and store it in :attr:`linear_system`.

        The linear system is defined by the current state of the model.

        """
        t_0 = time.time()
        if self.time_manager.time_index <= 1:
            self.linear_system = self.equation_system.assemble()
            self.linear_system_jacobian = self.linear_system[0]
            self.linear_system_residual = self.linear_system[1]
        else:
            self.linear_system_residual = self.equation_system.assemble(
                evaluate_jacobian=False
            )
        logger.debug(f"\nAssembled linear system in {time.time() - t_0:.2e} seconds.")


class DynamicMomentumBalanceABCLinear(
    SolutionStrategyAssembleLinearSystemOnce,
    DynamicMomentumBalanceABC,
): ...

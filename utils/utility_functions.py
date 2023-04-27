import numpy as np
import porepy as pp
from typing import Optional


def get_solution_values(
    name: str,
    data: dict,
    time_step_index: Optional[int] = None,
    iterate_index: Optional[int] = None,
) -> np.ndarray:
    """Function for fetching values stored in data dictionary.

    It looks very ugly to fetch values (that are not connected to a variable) from the
    data dictionary. Therefore this function will come in handy such that the keys
    `pp.TIME_STEP_SOLUTIONS` or `pp.ITERATE_SOLUTIONS` are not scattered all over the
    code.

    Parameters:
        name: Name of the parameter whose values we are interested in.
        data: The data dictionary.
        time_step_index: Which time step we want to get values for. 0 is current, 1 is
            one time step back in time. Only limited by how many time steps are stored
           from before.
        iterate_index: Which iterate we want to get values for. 0 is current, 1 is one
            iterate back in time. Only limited by how many iterates are stored from
            before.

    """
    if (time_step_index is None and iterate_index is None) or (
        time_step_index is not None and iterate_index is not None
    ):
        raise ValueError(
            "Both time_step_index and iterate_index cannot be None/assigned a value."
            "Only one at a time."
        )

    if time_step_index is not None:
        return data[pp.TIME_STEP_SOLUTIONS][name][time_step_index]
    else:
        return data[pp.ITERATE_SOLUTIONS][name][iterate_index]

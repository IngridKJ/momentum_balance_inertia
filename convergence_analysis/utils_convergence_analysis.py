from copy import deepcopy

import porepy as pp
import numpy as np
from porepy.utils.txt_io import TxtData, export_data_to_txt


def run_analysis(self) -> list:
    """Run convergence analysis.

    See the original method (found in porepy.applications.convergence_analysis) for more
    detailed documentation. This function is a copy of the PorePy-one, with the
    extension to also append the number of cells to the convergence results.

    Returns:
        List of results (i.e., data classes containing the errors) for each
        refinement level.

    """
    convergence_results: list = []
    for level in range(self.levels):
        setup = self.model_class(deepcopy(self.model_params[level]))
        pp.run_time_dependent_model(setup, deepcopy(self.model_params[level]))

        setattr(setup.results[-1], "cell_diameter", setup.mdg.diameter())
        setattr(setup.results[-1], "dt", setup.time_manager.dt)
        setattr(
            setup.results[-1],
            "num_cells",
            setup.mdg.subdomains(dim=setup.nd)[0].num_cells,
        )

        convergence_results.append(setup.results[-1])
    return convergence_results


def export_errors_to_txt(
    self,
    list_of_results: list,
    variables_to_export=None,
    file_name="error_analysis.txt",
) -> None:
    """Write errors into a ``txt`` file.

    See the original method (found in porepy.applications.convergence_analysis) for more
    detailed documentation. This function is a copy of the PorePy-one, with the
    extension to also export number of cells in the domain.

    """
    # Filter variables from the list of results
    var_names: list[str] = self._filter_variables_from_list_of_results(
        list_of_results=list_of_results,
        variables=variables_to_export,
    )

    # Filter errors to be exported
    errors_to_export: dict[str, np.ndarray] = {}
    for name in var_names:
        # Loop over lists of results
        var_error: list[float] = []
        for result in list_of_results:
            var_error.append(getattr(result, name))
        # Append to the dictionary
        errors_to_export[name] = np.array(var_error)

    # Prepare to export
    list_of_txt_data: list[TxtData] = []
    # Append cell diameters
    cell_diameters = np.array([result.cell_diameter for result in list_of_results])
    list_of_txt_data.append(
        TxtData(
            header="cell_diameter",
            array=cell_diameters,
            format=self._set_column_data_format("cell_diameter"),
        )
    )

    # Append cell number
    cell_diameters = np.array([result.num_cells for result in list_of_results])
    list_of_txt_data.append(
        TxtData(
            header="num_cells",
            array=cell_diameters,
            # Want integer value for cell number
            format="%d",
        )
    )

    time_steps = np.array([result.dt for result in list_of_results])
    list_of_txt_data.append(
        TxtData(
            header="time_step",
            array=time_steps,
            format=self._set_column_data_format("time_step"),
        )
    )

    for key in errors_to_export.keys():
        list_of_txt_data.append(
            TxtData(
                header=key,
                array=errors_to_export[key],
                format=self._set_column_data_format(key),
            )
        )

    # Finally, call the function to write into the txt
    export_data_to_txt(list_of_txt_data, file_name)

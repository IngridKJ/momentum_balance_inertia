import porepy as pp

from base_script_fixed_grid import FixedGridTestModel7Meter
from base_script_fixed_time import time_manager_tf50_ts1000


params = {
    "time_manager": time_manager_tf50_ts1000,
    "grid_type": "cartesian",
    "folder_name": "testing_visualization",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
}

model = FixedGridTestModel7Meter(params)

import os
import sys

runscript_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
model.filename = f"error_ts{time_manager_tf50_ts1000.time_steps}_{runscript_name}.txt"

pp.run_time_dependent_model(model, params)

from plot_l2_error import plot_the_error

plot_the_error(model.filename)

import porepy as pp

import os
import sys

from base_script_fixed_grid import FixedGridTestModel7Meter
from base_script_fixed_time import time_manager_tf50_ts500

time_manager = time_manager_tf50_ts500

params = {
    "time_manager": time_manager,
    "grid_type": "cartesian",
    "folder_name": "testing_visualization",
    "manufactured_solution": "simply_zero",
    "progressbars": True,
    "write_errors": True,
}

model = FixedGridTestModel7Meter(params)

runscript_name = os.path.splitext(os.path.basename(sys.argv[0]))[0][10:]
model.filename = f"ts{time_manager.time_steps}_{runscript_name}.txt"

pp.run_time_dependent_model(model, params)

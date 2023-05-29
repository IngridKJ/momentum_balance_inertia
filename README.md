# momentum_balance_inertia
Within PorePy there is a model class for solving the static momentum balance equation.
This repo is dedicated to including inertia term to this model class, such that the dynamic momentum balance equation (elastic wave equation) can be solved.

Specifically, this includes implementation of the Newmark time discretization scheme. 
The implementation is verified by convergence analysis using manufactured analytical solution.

Models are found within the [models](./models/) directory:
* [dynamic momentum balance](./models/dynamic_momentum_balance.py).
* [momentum balance](./models/no_inertia_momentum_balance.py) (i.e. just a call to and slight modification of the built-in PorePy model).

Runscripts:
* [run.py](./runscripts/run.py) is for running the momentum balance as is in PorePy. Only the geometry is slightly modified.
* [run_dynamic_1D.py](./runscripts/run_dynamic_1D.py) is for running the dynamic momentum balance in quasi 1D, meaning 3D but only one cell in two of the three directions.
* [run_dynamic_2D.py](./runscripts/run_dynamic_2D.py) is for running the dynamic momentum balance in 2D.
* [run_dynamic_3D.py](./runscripts/run_dynamic_3D.py) is for running the dynamic momentum balance in 3D.

Note that these scripts are mostly used as example runs. 
The initial velocity/acceleration are as of now only set arbitrary values.

Runscripts first made for "visual" analytical comparison:
* [2D_static_analytical_comparison](./2D_static_analytical_comparison.py) is a successfull attempt at MMS with static momentum balance equation.
* [dynamic_2D_model](./dynamic_2D_model.py) is for dynamic momentum balance (2D).
* [3D_dynamic_time_dep_source](./3D_dynamic_time_dep_source.py) is for dynamic momentum balance (3D).

Actual verification is only done in 2D, but 3D should be no different. 
The setup is as follows:
* [convergence_runscript_dynamic](./convergence_runscript_dynamic.py) is the runscript for running the convergence analysis. Function "bubble" is used for investigating convergence in space. Function "sine_bubble" is used for investigating convergence in time.
* [manufactured_solution_dynamic](./manufactured_solution_dynamic.py) is implementation of a verification setup for the dynamic momentum balance.

This verification setup takes use of the model defined within [dynamic_2D_model](./dynamic_2D_model.py).

Sanity check: Checking convergence rates for static momentum balance
* [convergence_runscript_static](./verification/convergence_runscript_static.py) is the runscript for running the convergence analysis.
* [manufactured_solution_static](./verification/manufactured_solution_static.py) is implementation of a verification setup for the static momentum balance. 
* [model_static_mom_bal.py](./verification/model_static_mom_bal.py) is the model setup for the above.

This might be deleted, but for now it's just hid within [verification](./verification/).

[ParaView](./ParaView/) directory is mainly for .py scripts or .pvsm files for visualization in ParaView.
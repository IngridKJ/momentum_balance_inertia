# momentum_balance_inertia
Within PorePy there is a model class for solving the static momentum balance equation.
This repo is dedicated to including inertia term to this model class, such that the dynamic momentum balance equation (elastic wave equation) can be solved.

Specifically, this includes implementation of the Newmark time discretization scheme. 
The implementation is verified by convergence analysis using manufactured analytical solution.

## Models
Models are found within the [models](./models/) directory:
* [dynamic momentum balance](./models/dynamic_momentum_balance.py).
* [momentum balance](./models/no_inertia_momentum_balance.py) (i.e. just a call to and slight modification of the built-in PorePy model).


## Runscripts
* [2D_static_analytical_comparison](./2D_static_analytical_comparison.py) is a successfull attempt at MMS with static momentum balance equation.
* [dynamic_2D_model](./dynamic_2D_model.py) is for dynamic momentum balance (2D).
* [dynamic_3D_model](./dynamic_3D_model.py) is for dynamic momentum balance (3D).


## Actual verification setup
Convergence analysis is only done in 2D, but 3D should be no different. 
The setup is as follows:
* [runscript_convergence_analysis](./runscript_convergence_analysis.py) is the runscript for running the convergence analysis. Function "bubble" is used for investigating convergence in space. Function "sine_bubble" is used for investigating convergence in time.
* [manufactured_solution_dynamic](./manufactured_solution_dynamic.py) is implementation of a verification setup for the dynamic momentum balance.

This verification setup takes use of the model defined within [dynamic_2D_model](./dynamic_2D_model.py).


## Sanity check
Before the convergence rates for the dynamic momentum balance were as expected, a "practice run" of checking the convergence of the static momentum balance was performed.
* [convergence_runscript_static](./verification/convergence_runscript_static.py) is the runscript for running the convergence analysis.
* [manufactured_solution_static](./verification/manufactured_solution_static.py) is implementation of a verification setup for the static momentum balance. 
* [model_static_mom_bal.py](./verification/model_static_mom_bal.py) is the model setup for the above.

This might be deleted, but for now it's just hid within [verification](./verification/).

## Other
[ParaView](./ParaView/) directory is mainly for .py scripts or .pvsm files for visualization in ParaView.

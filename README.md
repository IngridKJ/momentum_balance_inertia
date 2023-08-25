# momentum_balance_inertia
Within PorePy there is a model class for solving the static momentum balance equation.
This repo is dedicated to including inertia term to this model class, such that the
dynamic momentum balance equation (elastic wave equation) can be solved.

Specifically, this includes implementation of the Newmark time discretization scheme.
The implementation is verified by convergence analysis using manufactured analytical
solution. Every time the source code concerning the "core" implementation has been
refactored, a convergence analysis is run prior to pushing - just in case.

## Models
Models are found within the [models](./models/) directory:
* [momentum balance](./models/no_inertia_momentum_balance.py) (i.e. just a call to and
  slight modification of the built-in PorePy model).
* [dynamic momentum balance](./models/dynamic_momentum_balance.py) is the basic dynamic
  momentum balance model. It does not include things such as time dependent source terms
  or time dependent boundary conditions.
* [time_dependent_source_term](./models/time_dependent_source_term.py) is a model
  indlucing a time dependent source term. It inherits from [the basic
  model](./models/dynamic_momentum_balance.py).
* (WIP) [absorbing_boundary_conditions](./models/absorbing_boundary_conditions.py) is a
  model that has absorbing boundary conditions included. It inherits from the [time
  dependent source term](./models/time_dependent_source_term.py) model.


## Runscripts and modified models
* [unit_test_ABC](./unit_test_ABC.py) is for running the a quasi 1D model with one
absorbing boundary.
* [runscript_center_source_ABC](./runscript_center_source_ABC.py) is for running a 2D
model with absorbing boundary conditions on all domain boundaries. It has a centered
source term to see how the propagating wave is handled by the ABCs.
### "Old" runscripts
I am working with refactoring the models. The refactoring is done, but not all
runscripts are updated accordingly. The idea is to avoid having the "original" modified
models defined in runscripts, like in the files listed below. However, these scripts
(should) still work.
* [2D_static_analytical_comparison](./2D_static_analytical_comparison.py) is a
  successfull attempt at MMS with static momentum balance equation.
* [dynamic_2D_model](./dynamic_2D_model.py) is for dynamic momentum balance (2D).
* [dynamic_2D_model_time_dep_BC](./dynamic_2D_model_time_dep_BC.py) is for dynamic
  momentum balance with time dependent boundary conditions.
* [dynamic_2D_model_ABC](./dynamic_2D_model_ABC.py) is for dynamic momentum balance with
  absorbing boundary conditions (ABCs).
* [dynamic_3D_model](./dynamic_3D_model.py) is for dynamic momentum balance (3D).
## Actual verification setup
Convergence analysis is only done in 2D, but 3D should be no different. The setup is as
follows:
* [runscript_convergence_analysis](./runscript_convergence_analysis.py) is the runscript
  for running the convergence analysis. Function "bubble" is used for investigating
  convergence in space. Function "sine_bubble" is used for investigating convergence in
  time.
* [manufactured_solution_dynamic](./manufactured_solution_dynamic.py) is implementation
  of a verification setup for the dynamic momentum balance.

The verification setup takes use of [this](./models/time_dependent_source_term.py)
model.


## Sanity check
Before the convergence rates for the dynamic momentum balance were as expected, a
"practice run" of checking the convergence of the static momentum balance was performed.
* [convergence_runscript_static](./verification/convergence_runscript_static.py) is the
  runscript for running the convergence analysis.
* [manufactured_solution_static](./verification/manufactured_solution_static.py) is
  implementation of a verification setup for the static momentum balance. 
* [model_static_mom_bal.py](./verification/model_static_mom_bal.py) is the model setup
  for the above.

This might be deleted, but for now it's just hid within [verification](./verification/).

## Other
[ParaView](./ParaView/) directory is mainly for .py scripts or .pvsm files for
visualization in ParaView.

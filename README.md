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
  momentum balance model.
* [time_dependent_source_term](./models/time_dependent_source_term.py) is a model also
  assigning a time dependent source term. It inherits from [the basic
  model](./models/dynamic_momentum_balance.py).
* [absorbing_boundary_conditions](./models/absorbing_boundary_conditions.py) is a model
  that has absorbing boundary conditions included. It inherits from the [time dependent
  source term](./models/time_dependent_source_term.py) model.

## Runscripts
The two following runscripts have a centered source term to see how the propagating wave
is handled by the ABCs. All domain boundaries are absorbing.
* [center_source_2D_ABC](./center_source_2D_ABC.py) is for running the 2D model with
  absorbing boundaries.
* [center_source_3D_ABC](./center_source_3D_ABC.py) is for running the 3D model with
  absorbing boundaries.

Other runscripts:
* [runscript_3D](./runscript_3D.py) is a regular 3D runscript for the dynamic momentum
  balance. No absorbing boundaries, just zero Dirichlet.

## Test setups for the absorbing boundary conditions (ABCs)
* [Orthogonal wave](./verification/Verification%20and%20runscripts/Orthogonal%20wave/)
  contains a setup for propagation of an orthogonal wave. It is ready with a few
  different choices for time stepping, parameter value and grid.
* [unit_test_3D_ABC](./unit_test_3D_ABC.py) is for running the 3D quasi-1D model with
one absorbing boundary.

## Verification setup for the dynamic momentum balance
Convergence analysis is only done in 2D, but 3D should be no different. The setup is as
follows:
* [runscript_convergence_analysis](./runscript_convergence_analysis.py) is the runscript
  for running the convergence analysis. Function "bubble" is used for investigating
  convergence in space. Function "sine_bubble" is used for investigating convergence in
  time.
* [runscript_convergence_analysis_3D](./runscript_convergence_analysis_3D.py) is the
  same as above but for 3D.
* [manufactured_solution_dynamic](./manufactured_solution_dynamic.py) is implementation
  of a verification setup for the dynamic momentum balance.
* [manufactured_solution_dynamic_3D](./manufactured_solution_dynamic_3D.py) is the same
  as above but for 3D.

The verification setups takes use of [this](./models/time_dependent_source_term.py)
model.

### "Old" runscripts
* [2D_static_analytical_comparison](./2D_static_analytical_comparison.py) is a
  successfull attempt at MMS with static momentum balance equation.

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

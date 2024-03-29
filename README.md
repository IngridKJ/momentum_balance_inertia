# momentum_balance_inertia
Within PorePy there is a model class for solving the static momentum balance equation.
This repo is dedicated to including inertia term to this model class, such that the
dynamic momentum balance equation (elastic wave equation) can be solved.

Specifically, this includes implementation of the Newmark time discretization scheme.
The implementation is verified by:
* Convergence analysis using manufactured analytical
solution

In addition to this I have implemented some absorbing boundary conditions.

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
Runscripts in the repository include:
* Generic 3D runscript for time dependent source term
* Heterogeneous and anisotropic 3D runscripts with Ricker wavelet as source 

All runscripts use the model classes listed above.


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

## Utility material
A collection of utility material is found within the [utils](./utils/) directory:
* [anisotropy mixins](./utils/anisotropy_mixins.py) contains mixins for anisotropic stiffness tensors.
* [boundary_condition_setups](./utils/boundary_condition_setups.py) contains a mixin that is to be deleted. Might be filled with other generalised setups at a later point.
* [perturbed_geometry_mixins](./utils/perturbed_geometry_mixins.py) contains mixins for three types/configurations of perturbed geometry.
* [stiffness tensors](./utils/stiffness_tensors.py) contains a fourth order stiffness tensor object for a transversely isotropic material.
* [utility functions](./utils/utility_functions.py) contains mostly functions related to analytical solution expressions and fetching subdomain-related quantities (that I think are not already covered by PorePy functions, I might be mistaken)

Refer to the files within the directory for more details about the specific contents.

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

# momentum_balance_inertia
Within PorePy there is a model class for solving the static momentum balance equation.
This repo is dedicated to including inertia term to this model class, such that the
dynamic momentum balance equation (elastic wave equation) can be solved.

Specifically, this includes implementation of the Newmark time discretization scheme.
The implementation is verified by:
* Convergence analysis using manufactured analytical
solution

In addition to this I have implemented some absorbing boundary conditions. Ongoing verification is happening through:
* Convergence analysis
* Energy decay analysis

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
* [runscript_transverse_isotropy](./runscript_transverse_isotropy.py) is a 3D model with a vertical transverse isotropic inner domain. 

Other runscripts:
* [runscript_3D](./runscript_3D.py) is a regular 3D runscript for the dynamic momentum
  balance. No absorbing boundaries, just zero Dirichlet.

## Test setups for the absorbing boundary conditions (ABCs)
The domain starts out empty and a wave propagates from west to east:
* [runscript_dirichlet_wave.py](./run_simulations/runscript_dirichlet_wave.py) has a wave that is initiated by a time dependent Dirichlet condition on the west boundary

The domain is fully filled with the wave and no energy is added to the system. 
The only things driving the wave are initial values for displacement, velocity and acceleration:
* [runscript_orthogonal.py](./run_simulations/runscript_orthogonal.py) is for running the 2D quasi-1D model with absorbing boundary on the west and east side. 
* [runscript_diagonal.py](./run_simulations/runscript_diagonal.py) is for running a simulation with all absorbing boundaries. The wave is a rotation of the wave in [runscript_orthogonal.py](./run_simulations/runscript_orthogonal.py). 


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

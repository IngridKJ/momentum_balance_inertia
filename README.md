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
The model class for solving the elastic wave equation is found within
[elastic_wave_equation_abc](./models/elastic_wave_equation_abc.py). Absorbing boundary
conditions are by default applied to all domain boundaries. 
* A special case of this equation is found
  [here](./models/elastic_wave_equation_abc_linear.py). In the case of a linear problem,
  the Jacobian doesn't change. This model setup makes sure that the Jacobian is
  assembled only once.

A model class for the static problem is found within [momentum
  balance](./models/no_inertia_momentum_balance.py). This is just a call to (and slight
  modification of) the built-in PorePy model.

## Custom solvers and run-functions
[solvers](./solvers) contains mixins for custom solvers. Specifically, the mixin that is
there now will allow for PETSc usage whenever that is available. It also takes into
consideration whether the Jacobian should be assembled or fetched.

[run_models](./run_models) contains custom run-models functions. In the case of a the
Jacobian being assembled only once, adaptations were needed to the run-model-function
such that the residual was not assembled twice per time step.

TODO: Adapt all runscripts to match this. Currently some of them rely on a specific
modification to porepy source code files.

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


### "Old" runscripts
* [2D_static_analytical_comparison](./2D_static_analytical_comparison.py) is a
  successfull attempt at MMS with static momentum balance equation.

## Utility material
A collection of utility material is found within the [utils](./utils/) directory:
* [anisotropy mixins](./utils/anisotropy_mixins.py) contains mixins for anisotropic
  stiffness tensors.
* [perturbed_geometry_mixins](./utils/perturbed_geometry_mixins.py) contains mixins for
  three types/configurations of perturbed geometry.
* [stiffness tensors](./utils/stiffness_tensors.py) contains a fourth order stiffness
  tensor object for a transversely isotropic material.
* [utility functions](./utils/utility_functions.py) contains mostly functions related to
  analytical solution expressions and fetching subdomain-related quantities (that I
  think are not already covered by PorePy functions, I might be mistaken)

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

Note that this part of the repo is not maintained, so it might not run. 
Small changes to PorePy may have influenced it.

### Known "bugs" or potential problems
* Heterogeneous wave speed representation in cases where manufactured solutions are used.
* Boundary condition methods might start acting weird in the case of intersecting fractures. Will fix when/if it becomes a problem.
* Robin boundary conditions with the PorePy models are a bit ... yeah. Seek a permanent solution for that.
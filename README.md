# momentum_balance_inertia
This repository contains everything needed to run the simulation examples found in the
MPSA-Newmark paper.

## Models
The model classes are standardized setups for solving the elastic wave equation with absorbing boundary conditions with PorePy. There are currently two model class setups:
* [elastic_wave_equation_abc](./models/elastic_wave_equation_abc.py) considers a general
  setup for solving the elastic wave equation with absorbing boundaries on all domain
  sides.
* [elastic_wave_equation_abc_linear](./models/elastic_wave_equation_abc_linear.py)
  considers the case where all equations related to fractures in the domain are
  discarded. In practice this means that the model class is used for solving elastic
  wave propagation in rocks without fractures, or in rocks with **open** fractures
  (inner zero traction Neumann boundaries). In both cases the problem is linear, meaning
  that the Jacobian doesn't change between time steps. Hence, this model setup
  facilitates that the Jacobian is assembled only once. The linear model setup should be
  used together with the custom run-model function which is detailed in the section
  below.

## Custom solvers and run-functions
[solvers](./solvers) contains mixins for custom solvers. Specifically, the mixin that is
there now will allow for PETSc usage whenever that is available. It also takes into
consideration whether the Jacobian should be assembled or fetched, where the latter is
the case if
[elastic_wave_equation_abc_linear](./models/elastic_wave_equation_abc_linear.py) is the
model setup to be run.

[run_models](./run_models) contains custom run-models functions. In the case of a the
Jacobian being assembled only once, adaptations were needed to the run-model-function
which originally lies within PorePy such that the residual was not assembled twice per
time step.

## Verification: Convergence and energy decay analyses
### Convergence analysis of MPSA-Newmark
The convergence analyses presented in the article are performed with 
homogeneous Dirichlet conditions on a 3D simplex grid:
* Convergence in space and time:
  * [runscript_space_time_convergence_dirichlet_boundaries](./convergence_analysis/runscript_space_time_convergence_dirichlet_boundaries.py)
* Convergence in space:
  * [runscript_space_convergence_dirichlet_boundaries](./convergence_analysis/runscript_space_convergence_dirichlet_boundaries.py) 
* Convergence in time:
  * [runscript_time_convergence_dirichlet_boundaries](./convergence_analysis/runscript_time_convergence_dirichlet_boundaries.py) 

All the runscripts utilize
[manufactured_solution_dynamic_3D](./convergence_analysis/convergence_analysis_models/manufactured_solution_dynamic_3D.py)
as the manufactured solution setup.

### Convergence analysis of MPSA-Newmark with absorbing boundaries
Convergence of the solution is performed by sending an orthogonal wave from the left
towards the right boundary, where the right boundary is absorbing. The top and bottom
boundaries have roller conditions, and the left boundary is a time-dependent Dirichlet
condition which sends in the orthogonal wave.

Convergence in space and time, isotropic:
  * [runscript_space_time_convergence_absorbing_boundaries](./convergence_analysis/runscript_space_time_convergence_absorbing_boundaries.py) which uses the model class setup found in [model_convergence_ABC](./convergence_analysis/convergence_analysis_models/model_convergence_ABC.py)

Convergence in space and time, vertical strip inside the domain which is anisotropic in
directions orthogonal to the propagating wave:
  * [runscript_space_time_absorbing_boundaries_anisotropy](./convergence_analysis/runscript_space_time_absorbing_boundaries_anisotropy.py)
    which uses the model class setup found in
    [model_convergence_ABC_anisotropy](./convergence_analysis/convergence_analysis_models/model_convergence_ABC_anisotropy.py)


### Energy decay analysis of MPSA-Newmark with absorbing boundaries
The energy decay analysis is performed both for successive refinement 
of the grid, as well as for varying wave incidence angles. 

* Successive grid refinement is done by running the script
[runscript_energy_decay_space_refinement](./convergence_analysis/runscript_energy_decay_space_refinement.py).

* Varying the wave incidence angle, $\theta$, is done by running the script
  [runscript_energy_decay_vary_theta](./convergence_analysis/runscript_energy_decay_vary_theta.py).
  

## Simulation examples
Simulation example runscripts are found within [this](./example_runscripts/) folder.
* The simulation from Example 1.1, which considers a seismic source located inside an
  inner transversely isotropic domain, is run by
  [runscript_example_1_1_source_in_inner_domain](./example_runscripts/runscript_example_1_1_source_in_inner_domain.py).
* The simulation from Example 1.2, which considers a seismic source located outside an
  inner transversely isotropic domain, is run by
  [runscript_example_1_2_source_in_outer_domain](./example_runscripts/runscript_example_1_2_source_in_outer_domain.py).
* The simulation from Example 2, which considers a layered heterogeneous medium with an
  open fracture, is run by
  [runscript_example_2_heterogeneous_fractured_domain](./example_runscripts/runscript_example_2_heterogeneous_fractured_domain.py).

## Utility material
A collection of utility material is found within the [utils](./utils/) directory:
* [anisotropy mixins](./utils/anisotropy_mixins.py) contains mixins 
for anisotropic stiffness tensors.
* [perturbed_geometry_mixins](./utils/perturbed_geometry_mixins.py) contains mixins for
three types/configurations of perturbed geometry.
* [stiffness tensors](./utils/stiffness_tensors.py) contains a fourth order stiffness
tensor object for a transversely isotropic material.
* [utility functions](./utils/utility_functions.py) contains mostly functions related to
analytical solution expressions and fetching subdomain-related quantities.

I refer to the files within the directory for more details about the specific contents.

## Tests
Tests are covering:
* MPSA-Newmark convergence with homogeneous Dirichlet conditions in 2D and 3D.
* MPSA-Newmark convergence with absorbing boundary conditions.
* Construction of the transversely isotropic tensor.
* The utility function ``inner_domain_cells`` which is used in the construction of the
  transversely isotropic tensor. Note: Moving over to use
  ``use_constraints_for_inner_domain_cells``. Consistent update pending.
# momentum_balance_inertia
Code for including inertia term to momentum balance equation in PorePy.

Runscripts:
* [run.py](./run.py) is for running the momentum balance as is in PorePy. The geometry is only slightly changed.
* [run_dynamic_1D.py](./run_dynamic_1D.py) is for running the dynamic momentum balance in quasi 1D, meaning 3D but only one cell in two of the three directions.
* [run_dynamic_2D.py](./run_dynamic_2D.py) is for running the dynamic momentum balance in 2D.
* [run_dynamic_3D.py](./run_dynamic_3D.py) is for running the dynamic momentum balance in 3D.

Runscripts for analytical comparison:
* [1D_analytical_comparison.py](./1D_analytical_comparison.py) is a first (and failed) attempt at comparing the solution of the momentum balance equation (non-dynamic).
* [2D_static_analytical_comparison](./1D_static_analytical_comparison.py) is a successfull attempt at MMS with static momentum balance equation.
* [2D_dynamic_time_dep_source](./2D_dynamic_time_dep_source.py) is for dynamic momentum balance

models directory contains models (surprise surprise ...)
* [dynamic momentum balance](./models/dynamic_momentum_balance.py).
* [momentum balance](./models/no_inertia_momentum_balance.py) (i.e. just a call to and slight modification of the built-in PorePy model)

[ParaView](./ParaView/) directory is mainly for .py scripts to run for visualization in ParaView.

Folders wit some visualization files are also found in the main directory. For now this includes the visualization files for
* 2D static momentum balance analytical comparison

"""This file contains some utility functions used in various files throughout this repo.

In short, functions in this file are called for e.g. fetching subdomain-related
quantities and for utilizing symbolic representations of analytical solutions (e.g.
creating source terms, setting initial values, compute errors, etc.). These analytical
solutions are determined by the "manufactured_solution" key value in the params
dictionary. Creation of source terms uses symbolic differentiation provided by sympy.
Therefore, running other manufactured solutions than those already present is easily
done by adding the expression for it where the manufactured solution is defined. Additionally, there are utility functions for construction of the 9x9 representation of a stiffness tensor representing a transversely isotropic media.


"""

from typing import Optional, Union

import numpy as np
import porepy as pp
import sympy as sym

# -------- Fetching/Computing values


def acceleration_velocity_displacement(
    model,
    data: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Function for fetching acceleration, velocity and displacement values.

    Found it repetitive to do this in the methods for updating velocity and acceleration
    values in the dynamic momentum balance model.

    Parameters:
        model: The model.
        data: Data dictionary we want to fecth values from.

    Returns:
        A tuple with previous acceleration, velocity and displacement + the current
        displacement.

    """
    a_previous = pp.get_solution_values(
        name=model.acceleration_key, data=data, time_step_index=0
    )
    v_previous = pp.get_solution_values(
        name=model.velocity_key, data=data, time_step_index=0
    )

    u_current = pp.get_solution_values(
        name=model.displacement_variable, data=data, iterate_index=0
    )

    u_previous = pp.get_solution_values(
        name=model.displacement_variable, data=data, time_step_index=0
    )

    return a_previous, v_previous, u_previous, u_current


# -------- Wrap for symbolic representations of 2D and 3D functions/equation terms.


def symbolic_representation(
    model, is_2D: bool = True, return_dt=False, return_ddt=False
) -> tuple:
    """Wrapper for symbolic representation of functions.

    Parameters:
        model: The model class
        is_2D: Flag for whether the problem is 2D or not. Defaults to True.
        return_dt: Flag for whether the time derivative of the function should be
            returned. Defaults to False.
        return_ddt: Flag for whether the second time derivative of the function should
            be returned. Defaults to False.

    Returns:
        The symbolic representation of either the displacement, velocity or
        acceleration.

    """
    if is_2D:
        return _symbolic_representation_2D(
            model=model, return_dt=return_dt, return_ddt=return_ddt
        )
    elif not is_2D:
        return _symbolic_representation_3D(
            model=model, return_dt=return_dt, return_ddt=return_ddt
        )
    return None


def symbolic_equation_terms(model, u, x, y, t, is_2D: bool = True, z=None):
    if is_2D:
        return _symbolic_equation_terms_2D(model=model, u=u, x=x, y=y, t=t)
    elif not is_2D:
        return _symbolic_equation_terms_3D(model=model, u=u, x=x, y=y, z=z, t=t)
    return None


## -------- 2D: Symbolic representation of manufactured solution-related expressions


def _symbolic_representation_2D(model, return_dt=False, return_ddt=False):
    """Symbolic representation of displacement, velocity or acceleration.

    Use of this method is rather simple, as the default analytical solution is that with
    the name "simply_zero" which is just zero solution. For another analytical solution
    one must simply just assign a different value to the key "manufactured_solution" in
    the model's parameter dictionary. Look into the code for what solutions are
    accessible, or make new ones if the ones already existing do not suffice.

    Parameters:
        model: The model class.
        return_dt: Flag for wether velocity is returned. Return velocity if True.
        return_ddt: Flag for whether acceleration is returned. Return acceleration if
            True.

    Raises:
        ValueError if return_dt and return_ddt is True.

    Returns:
        Tuple of the function (either u, first time derivative of u or second time
        derivative of u), and the symbols x, y, t.

    """
    if return_dt and return_ddt:
        raise ValueError(
            "Both return_dt and return_ddt cannot be True. Only one or neither."
        )

    x, y, t = sym.symbols("x y t")
    cp = model.primary_wave_speed(is_scalar=True)

    manufactured_sol = model.params.get("manufactured_solution", "simply_zero")
    if manufactured_sol == "unit_test":
        u1 = sym.sin(t - x / cp)
        u2 = 0
        u = [u1, u2]
    elif manufactured_sol == "simply_zero":
        u = [0, 0]
    elif manufactured_sol == "diagonal_wave":
        alpha = model.rotation_angle
        u1 = u2 = sym.sin(t - (x * sym.cos(alpha) + y * sym.sin(alpha)) / (cp))
        u = [sym.cos(alpha) * u1, sym.sin(alpha) * u2]
    elif manufactured_sol == "sin_bubble":
        u1 = u2 = sym.sin(5.0 * np.pi * t / 2.0) * x * (1 - x) * y * (1 - y)
        u = [u1, u2]

    if return_dt:
        dt_u = [sym.diff(u[0], t), sym.diff(u[1], t)]
        return dt_u, x, y, t
    elif return_ddt:
        ddt_u = [sym.diff(sym.diff(u[0], t), t), sym.diff(sym.diff(u[1], t), t)]
        return ddt_u, x, y, t

    # Here symbols are returned to avoid possible issues with references of created
    # symbols
    return u, x, y, t


def _symbolic_equation_terms_2D(model, u, x, y, t):
    """Symbolic representation of the momentum balance eqn. terms in 3D.

    Parameters:
        model: The model class
        u: Analytical solution from which the source term is found. As of now, this is
            defined through a call to the method symbolic_representation.
        x: Symbol for x-coordinate.
        y: Symbol for y-coordinate.
        t: Symbol for time variable.

    Returns:
        A tuple with the full source term, sigma and the acceleration term.

    """
    lam = model.solid.lame_lambda
    mu = model.solid.shear_modulus
    rho = model.solid.density

    # Exact acceleration
    ddt_u = [
        sym.diff(sym.diff(u[0], t), t),
        sym.diff(sym.diff(u[1], t), t),
    ]

    # Exact gradient of u and transpose of gradient of u
    grad_u = [
        [sym.diff(u[0], x), sym.diff(u[0], y)],
        [sym.diff(u[1], x), sym.diff(u[1], y)],
    ]

    grad_u_T = [[grad_u[0][0], grad_u[1][0]], [grad_u[0][1], grad_u[1][1]]]

    # Trace of gradient of u, in the linear algebra sense
    trace_grad_u = grad_u[0][0] + grad_u[1][1]

    # Exact strain (\epsilon(u))
    strain = 0.5 * np.array(
        [
            [grad_u[0][0] + grad_u_T[0][0], grad_u[0][1] + grad_u_T[0][1]],
            [grad_u[1][0] + grad_u_T[1][0], grad_u[1][1] + grad_u_T[1][1]],
        ]
    )

    # Exact stress tensor (\sigma(\epsilon(u)))
    sigma = [
        [2 * mu * strain[0][0] + lam * trace_grad_u, 2 * mu * strain[0][1]],
        [2 * mu * strain[1][0], 2 * mu * strain[1][1] + lam * trace_grad_u],
    ]

    # Divergence of sigma
    div_sigma = [
        sym.diff(sigma[0][0], x) + sym.diff(sigma[0][1], y),
        sym.diff(sigma[1][0], x) + sym.diff(sigma[1][1], y),
    ]

    # Full acceleration term
    acceleration_term = [rho * ddt_u[0], rho * ddt_u[1]]

    # Finally, the source term
    source_term = [
        acceleration_term[0] - div_sigma[0],
        acceleration_term[1] - div_sigma[1],
    ]

    # The two "extra" things returned here are for use in the convergence analysis runs.
    return source_term, sigma, acceleration_term


## -------- 3D: Symbolic representation of manufactured solution-related expressions


def _symbolic_representation_3D(model, return_dt=False, return_ddt=False):
    """3D symbolic representation of displacement, velocity or acceleration.

    See documentation of _symbolic_representation_2D.

    Parameters:
        model: The model class.
        return_dt: Flag for wether velocity is returned. Return velocity if True.
        return_ddt: Flag for whether acceleration is returned. Return acceleration if
            True.

    Raises:
        ValueError if return_dt and return_ddt is True.

    Returns:
        Tuple of the function (either u, first time derivative of u or second time
        derivative of u), and the symbols x, y, z, t.

    """
    if return_dt and return_ddt:
        raise ValueError(
            "Both return_dt and return_ddt cannot be True. Only one or neither."
        )

    x, y, z, t = sym.symbols("x y z t")
    cp = model.primary_wave_speed(is_scalar=True)
    manufactured_sol = model.params.get("manufactured_solution", "simply_zero")
    if manufactured_sol == "bubble":
        u1 = u2 = u3 = t**2 * x * (1 - x) * y * (1 - y) * z * (1 - z)
        u = [u1, u2, u3]
    elif manufactured_sol == "simply_zero":
        u = [0, 0, 0]
    elif manufactured_sol == "drum_solution":
        u1 = u2 = u3 = sym.sin(sym.pi * t) * x * (1 - x) * y * (1 - y) * z * (1 - z)
        u = [u1, u2, u3]
    elif manufactured_sol == "sin_bubble":
        u1 = u2 = u3 = (
            sym.sin(5.0 * np.pi * t / 2.0) * x * (1 - x) * y * (1 - y) * z * (1 - z)
        )
        u = [u1, u2, u3]
    elif manufactured_sol == "different_x_y_z_components":
        u1 = (
            sym.sin(5 * np.pi / 2 * t)
            * sym.sin(np.pi * x)
            * sym.sin(np.pi * y)
            * sym.sin(np.pi * z)
        )
        u2 = sym.sin(5 * np.pi / 2 * t) * x * (1 - x) * y * (1 - y) * z * (1 - z)
        u3 = sym.sin(5 * np.pi / 2 * t) * x * (1 - x) * y * (1 - y) * sym.sin(np.pi * z)
        u = [u1, u2, u3]
    elif manufactured_sol == "unit_test":
        u1 = 0
        u2 = 0
        u3 = sym.sin(t + z / cp)
        u = [u1, u2, u3]
    if return_dt:
        dt_u = [sym.diff(u[0], t), sym.diff(u[1], t), sym.diff(u[2], t)]
        return dt_u, x, y, z, t

    elif return_ddt:
        ddt_u = [
            sym.diff(sym.diff(u[0], t), t),
            sym.diff(sym.diff(u[1], t), t),
            sym.diff(sym.diff(u[2], t), t),
        ]
        return ddt_u, x, y, z, t

    # Here symbols are returned to avoid possible issues with references of created
    # symbols
    return u, x, y, z, t


def _symbolic_equation_terms_3D(model, u, x, y, z, t) -> list:
    """Symbolic representation of the momentum balance eqn. terms in 3D.

    Parameters:
        model: The model class
        u: Analytical solution from which the source term is found. As of now, this is
            defined through a call to the method symbolic_representation.
        x: Symbol for x-coordinate.
        y: Symbol for y-coordinate.
        z: Symbol for z-coordinate.
        t: Symbol for time variable.

    Returns:
        A tuple with the full source term, sigma and the acceleration term.

    """
    lam = model.solid.lame_lambda
    mu = model.solid.shear_modulus
    rho = model.solid.density

    # Exact acceleration
    ddt_u = [
        sym.diff(sym.diff(u[0], t), t),
        sym.diff(sym.diff(u[1], t), t),
        sym.diff(sym.diff(u[2], t), t),
    ]

    # Exact gradient of u and transpose of gradient of u
    grad_u = [
        [sym.diff(u[0], x), sym.diff(u[0], y), sym.diff(u[0], z)],
        [sym.diff(u[1], x), sym.diff(u[1], y), sym.diff(u[1], z)],
        [sym.diff(u[2], x), sym.diff(u[2], y), sym.diff(u[2], z)],
    ]

    grad_u_T = [
        [grad_u[0][0], grad_u[1][0], grad_u[2][0]],
        [grad_u[0][1], grad_u[1][1], grad_u[2][1]],
        [grad_u[0][2], grad_u[1][2], grad_u[2][2]],
    ]

    # Trace of gradient of u, in the linear algebra sense
    trace_grad_u = grad_u[0][0] + grad_u[1][1] + grad_u[2][2]

    # Exact strain (\epsilon(u))
    strain = 0.5 * np.array(
        [
            [
                grad_u[0][0] + grad_u_T[0][0],
                grad_u[0][1] + grad_u_T[0][1],
                grad_u[0][2] + grad_u_T[0][2],
            ],
            [
                grad_u[1][0] + grad_u_T[1][0],
                grad_u[1][1] + grad_u_T[1][1],
                grad_u[1][2] + grad_u_T[1][2],
            ],
            [
                grad_u[2][0] + grad_u_T[2][0],
                grad_u[2][1] + grad_u_T[2][1],
                grad_u[2][2] + grad_u_T[2][2],
            ],
        ]
    )

    # Exact stress tensor (\sigma(\epsilon(u)))
    sigma = [
        [
            2 * mu * strain[0][0] + lam * trace_grad_u,
            2 * mu * strain[0][1],
            2 * mu * strain[0][2],
        ],
        [
            2 * mu * strain[1][0],
            2 * mu * strain[1][1] + lam * trace_grad_u,
            2 * mu * strain[1][2],
        ],
        [
            2 * mu * strain[2][0],
            2 * mu * strain[2][1],
            2 * mu * strain[2][2] + lam * trace_grad_u,
        ],
    ]

    # Divergence of sigma
    div_sigma = [
        sym.diff(sigma[0][0], x) + sym.diff(sigma[0][1], y) + sym.diff(sigma[0][2], z),
        sym.diff(sigma[1][0], x) + sym.diff(sigma[1][1], y) + sym.diff(sigma[1][2], z),
        sym.diff(sigma[2][0], x) + sym.diff(sigma[2][1], y) + sym.diff(sigma[2][2], z),
    ]

    # Full acceleration term
    acceleration_term = [rho * ddt_u[0], rho * ddt_u[1], rho * ddt_u[2]]

    # Finally, the source term
    source_term = [
        acceleration_term[0] - div_sigma[0],
        acceleration_term[1] - div_sigma[1],
        acceleration_term[2] - div_sigma[2],
    ]
    # The two "extra" things returned here are for use in the convergence analysis runs.
    return source_term, sigma, acceleration_term


# -------- Wrap for displacement, velocity and acceleration lambdified function.


def u_v_a_wrap(
    model, is_2D: bool = True, return_dt: bool = False, return_ddt=False
) -> list:
    """Wrapper function for fetching displacement, velocity and acceleration functions.

    When setting up the simulation, it is possible to set a value to the key
    "manufactured_solution" in the parameter dictionary. This value helps choosing from
    a pool of manufactured solutions, and this method fetches those solutions. Depending
    on the parameters return_dt and return_ddt it can return the first and second time
    derivative of the manufactured solution. This is mostly used in convergence analysis
    runs or to initialize the model.

    Parameters:
        is_2D: Whether the problem is in 2D or 3D. Defaults to True (is 2D).
        return_dt: True if velocity is to be returned instead of displacement. Defaults
            to False.
        return_ddt: True if acceleration is to be returned instead of displacement.
            Defaults to False.

    """
    if is_2D:
        if not return_dt and not return_ddt:
            return _displacement_function_2D(model)
        elif return_dt:
            return _velocity_function_2D(model)
        elif return_ddt:
            return _acceleration_function_2D(model)
    elif not is_2D:
        if not return_dt and not return_ddt:
            return _displacement_function_3D(model)
        elif return_dt:
            return _velocity_function_3D(model)
        elif return_ddt:
            return _acceleration_function_3D(model)


# --------- Displacement, velocity and acceleration lambdified functions in 2D and 3D.

## -------- 2D


def _displacement_function_2D(model) -> list:
    """Lambdified expression of displacement function.

    Sometimes the symbolic representation of the displacement function is needed.
    Therefore, the lambdification of it is kept as a separate method here, and the
    symbolic representation is fetched from the method symbolic_representation.

    """
    u, x, y, t = symbolic_representation(model=model)
    u = [
        sym.lambdify((x, y, t), u[0], "numpy"),
        sym.lambdify((x, y, t), u[1], "numpy"),
    ]
    return u


def _velocity_function_2D(model) -> list:
    """Lambdified expression of velocity function.

    Sometimes the symbolic representation of the velocity function is needed. Therefore,
    the lambdification of it is kept as a separate method here, and the symbolic
    representation is fetched from the method symbolic_representation.

    """
    v, x, y, t = symbolic_representation(model=model, return_dt=True)
    v = [
        sym.lambdify((x, y, t), v[0], "numpy"),
        sym.lambdify((x, y, t), v[1], "numpy"),
    ]
    return v


def _acceleration_function_2D(model) -> list:
    """Lambdified expression of acceleration function.

    Sometimes the symbolic representation of the acceleration function is needed.
    Therefore, the lambdification of it is kept as a separate method here, and the
    symbolic representation is fetched from the method symbolic_representation.

    """
    a, x, y, t = symbolic_representation(model=model, return_ddt=True)
    a = [
        sym.lambdify((x, y, t), a[0], "numpy"),
        sym.lambdify((x, y, t), a[1], "numpy"),
    ]
    return a


## -------- 3D


def _displacement_function_3D(model) -> list:
    """Lambdified expression of displacement function.

    Sometimes the symbolic representation of the displacement function is needed.
    Therefore, the lambdification of it is kept as a separate method here, and the
    symbolic representation is fetched from the method symbolic_representation.

    """
    u, x, y, z, t = symbolic_representation(model=model, is_2D=False)
    u = [
        sym.lambdify((x, y, z, t), u[0], "numpy"),
        sym.lambdify((x, y, z, t), u[1], "numpy"),
        sym.lambdify((x, y, z, t), u[2], "numpy"),
    ]
    return u


def _velocity_function_3D(model) -> list:
    """Lambdified expression of velocity function.

    Sometimes the symbolic representation of the velocity function is needed. Therefore,
    the lambdification of it is kept as a separate method here, and the symbolic
    representation is fetched from the method symbolic_representation.

    """
    v, x, y, z, t = symbolic_representation(model=model, is_2D=False, return_dt=True)
    v = [
        sym.lambdify((x, y, z, t), v[0], "numpy"),
        sym.lambdify((x, y, z, t), v[1], "numpy"),
        sym.lambdify((x, y, z, t), v[2], "numpy"),
    ]
    return v


def _acceleration_function_3D(model) -> list:
    """Lambdified expression of acceleration function.

    Sometimes the symbolic representation of the acceleration function is needed.
    Therefore, the lambdification of it is kept as a separate method here, and the
    symbolic representation is fetched from the method symbolic_representation.

    """
    a, x, y, z, t = symbolic_representation(model=model, is_2D=False, return_ddt=True)
    a = [
        sym.lambdify((x, y, z, t), a[0], "numpy"),
        sym.lambdify((x, y, z, t), a[1], "numpy"),
        sym.lambdify((x, y, z, t), a[2], "numpy"),
    ]
    return a


# -------- Wrap for body force functions.


def body_force_function(model, is_2D: bool = True) -> list:
    """Wrapper function for the body forces in 2D and 3D.

    See the sub-methods for documentation. For now only used for constructing the force
    from a known analytical solution.

    Parameters:
        model: model class
        is_2D: flag for whether model is for 2D or 3D domain.

    Returns:
        A (lambdified) function to be used as the source term function.

    """
    if is_2D:
        return _body_force_func_2D(model)
    else:
        return _body_force_func_3D(model)


## -------- Body force functions in 2D and 3D


def _body_force_func_2D(model) -> list:
    """Lambdify the source term corresponding to a manufactured solution, 2D.

    Uses the methods symbolic_representation and symbolic_equation_terms. The former is
    for fetching the symbolic representation of the analytical solution, u, and the
    latter fetches the source term corresponding to u.

    Parameters:
        model: The model class.

    Returns:
        A (lambdified) function to be used as the source term function.

    """

    u, x, y, t = symbolic_representation(model=model)
    source, _, _ = symbolic_equation_terms(model=model, u=u, x=x, y=y, t=t)

    return [
        sym.lambdify((x, y, t), source[0], "numpy"),
        sym.lambdify((x, y, t), source[1], "numpy"),
    ]


def _body_force_func_3D(model) -> list:
    """Lambdify the source term corresponding to a manufactured solution, 2D.

    Uses the methods symbolic_representation and symbolic_equation_terms. The former is
    for fetching the symbolic representation of the analytical solution, u, and the
    latter fetches the source term corresponding to u.

    Parameters:
        model: The model class.

    Returns:
        A (lambdified) function to be used as the source term function.

    """

    u, x, y, z, t = symbolic_representation(model=model, is_2D=False)
    source, _, _ = symbolic_equation_terms(
        model=model, is_2D=False, u=u, x=x, y=y, z=z, t=t
    )

    return [
        sym.lambdify((x, y, z, t), source[0], "numpy"),
        sym.lambdify((x, y, z, t), source[1], "numpy"),
        sym.lambdify((x, y, z, t), source[2], "numpy"),
    ]


# -------- Functions related to subdomains


def use_constraints_for_inner_domain_cells(self, sd) -> np.ndarray:
    """Finds cell indices of the cells laying within constraints in the grid.

    Assumes the existance of the method set_polygons() in the model class which is
    calling this function. This function takes the nodes of the lines/polygons set by
    set_polygons(). The nodes are subsequently fed to the PorePy-functions
    point_in_polygon() (for 2D) or point_in_polyhedron() (for 3D) to find the indices of
    the cells within the constraints.
    
    Parameters:
        self: The model class (TODO: Rename). 
        sd: The subdomain where we want to find the indices of the cells within the 
            constraints.
    
    Returns:
        An array of the cell indices of the cells within the constraints.

    """
    points = sd.cell_centers[: self.nd, :]

    def nodes_of_constraints(self):
        """Helper function to fetch the nodes of the meshing constraints in the grid."""
        return np.array(
            [
                self._fractures[i].pts
                for i in self.params["meshing_kwargs"]["constraints"]
            ]
        )

    if self.nd == 2:
        if self.params["grid_type"] == "simplex":
            all_nodes_of_constraints = nodes_of_constraints(self)
        else:
            c1, c2, c3, c4 = self.set_polygons()
            all_nodes_of_constraints = np.array([c1, c2, c3, c4])
        polygon_vertices = all_nodes_of_constraints.T[0]
        inside = pp.geometry_property_checks.point_in_polygon(polygon_vertices, points)
    elif self.nd == 3:
        if self.params["grid_type"] == "simplex":
            all_nodes_of_constraints = nodes_of_constraints(self)
        else:
            all_nodes_of_constraints = self.set_polygons()
        inside = pp.geometry_property_checks.point_in_polyhedron(
            polyhedron=all_nodes_of_constraints, test_points=points
        )
    return np.where(inside)


def inner_domain_cells(
    self,
    sd: pp.Grid,
    width: Optional[Union[int, float, tuple]],
    inner_domain_center: Optional[tuple] = None,
) -> np.ndarray:
    """Function for fetching cells a certain width from the domain center in 3D.

    Relevant for e.g. constructing an inner anisotropic domain within an outer isotropic
    one. I need the cell numbers of the inner cells. Cell indices of the cells in the
    internal domain is returned.

    Raises:
        ValueError if the inner domain width exceeds that of the outer domain.

    Parameters:
        self: Kind of wrong to call it self.. Anyways, it is the model class. Same
            holds for the other functions being passed a "self".
        sd: Subdomain where the inner cells are to be found.
        width: If you want a cubic inner domain, pass an integer or a float. For a
            rectangular inner domain, pass a tuple with sidelengths in x-, y-, and
            z-direction.
        inner_domain_center: x, y, and z coordinate of the center of the inner domain.
            Note that this center should not be placed in such a way that the inner
            domain exceeds the boundaries of the outer domain. No error is raised at
            this point, so caution is adviced #dramatic. The code will probably still
            run, but then the absorbing boundaries are not correct anymore.

    Returns:
        An array of the cell indices of the cells within the "specified" inner domain.

    """
    cell_indices = []
    domain_width = self.domain.bounding_box["xmax"]

    if isinstance(width, float) or isinstance(width, int):
        if domain_width <= width:
            raise ValueError(
                "The domain width must be larger than the inner domain width."
            )
        cell_centers = sd.cell_centers.T
        for i, _ in enumerate(cell_centers):
            cs = cell_centers[i]
            if inner_domain_center is None:
                if np.all(cs < (domain_width + width) / 2.0) and (
                    np.all(cs > (domain_width - width) / 2.0)
                ):
                    cell_indices.append(i)
            else:
                inner_x_min = inner_domain_center[0] - width / 2
                inner_x_max = inner_domain_center[0] + width / 2

                inner_y_min = inner_domain_center[1] - width / 2
                inner_y_max = inner_domain_center[1] + width / 2

                inner_z_min = inner_domain_center[2] - width / 2
                inner_z_max = inner_domain_center[2] + width / 2

                if (
                    cs[0] > inner_x_min
                    and cs[1] > inner_y_min
                    and cs[2] > inner_z_min
                    and cs[0] < inner_x_max
                    and cs[1] < inner_y_max
                    and cs[2] < inner_z_max
                ):
                    cell_indices.append(i)

    elif isinstance(width, tuple):
        domain_width_x = self.domain.bounding_box["xmax"]
        domain_width_y = self.domain.bounding_box["ymax"]
        domain_width_z = self.domain.bounding_box["zmax"]

        if (
            domain_width_x <= width[0]
            or domain_width_y <= width[1]
            or domain_width_z <= width[2]
        ):
            raise ValueError(
                "The domain width must be larger than the inner domain width."
            )

        cell_centers = sd.cell_centers.T
        for i, _ in enumerate(cell_centers):
            cs = cell_centers[i]

            inner_x_min = inner_domain_center[0] - width[0] / 2
            inner_x_max = inner_domain_center[0] + width[0] / 2

            inner_y_min = inner_domain_center[1] - width[1] / 2
            inner_y_max = inner_domain_center[1] + width[1] / 2

            inner_z_min = inner_domain_center[2] - width[2] / 2
            inner_z_max = inner_domain_center[2] + width[2] / 2

            if (
                cs[0] > inner_x_min
                and cs[1] > inner_y_min
                and cs[2] > inner_z_min
                and cs[0] < inner_x_max
                and cs[1] < inner_y_max
                and cs[2] < inner_z_max
            ):
                cell_indices.append(i)
    return cell_indices


def _get_boundary_cells(self, sd: pp.Grid, coord: str, extreme: str) -> np.ndarray:
    """Grab cell indices of a certain side of a subdomain.

    This might already exist within PorePy, but I couldn't find it ...

    Parameters:
        sd: The subdomain we are interested in grabbing cells for.
        coord: Either "x", "y" or "z" depending on what coordinate direction is
            relevant for choosing the boundary cells. E.g., for an east boundary it
            is "x", for a north boundary it is "y".
        extreme: Whether it is the lower or upper "extreme" of the coord-value.
            East corresponds to "xmax" and west corresponds to "xmin".

    Returns:
        An array with the indices of the boundary cells of a certain domain side.

    """
    if coord == "x":
        coord = 0
    elif coord == "y":
        coord = 1
    elif coord == "z":
        coord = 2

    faces = sd.get_all_boundary_faces()
    face_centers = sd.face_centers
    face_indices = [
        f for f in faces if face_centers[coord][f] == self.domain.bounding_box[extreme]
    ]

    boundary_cells = sd.signs_and_cells_of_boundary_faces(faces=np.array(face_indices))[
        1
    ]

    return boundary_cells, face_indices


def get_boundary_cells(
    self, sd: pp.Grid, side: str, return_faces: bool = False
) -> np.ndarray:
    """Grabs the cell indices of a certain subdomain side.

    Wrapper-like function for fetching boundary cell indices.

    Parameters:
        sd: The subdomain
        side: The side we want the cell indices for. Should take values "south",
            "north", "west", "east", "top" or "bottom".

    Returns:
        An array with the indices of the boundary cells of a certain domain side.

    """
    if side == "south":
        cells, faces = _get_boundary_cells(self=self, sd=sd, coord="y", extreme="ymin")
    elif side == "north":
        cells, faces = _get_boundary_cells(self=self, sd=sd, coord="y", extreme="ymax")
    elif side == "west":
        cells, faces = _get_boundary_cells(self=self, sd=sd, coord="x", extreme="xmin")
    elif side == "east":
        cells, faces = _get_boundary_cells(self=self, sd=sd, coord="x", extreme="xmax")
    elif side == "top":
        cells, faces = _get_boundary_cells(self=self, sd=sd, coord="z", extreme="zmax")
    elif side == "bottom":
        cells, faces = _get_boundary_cells(self=self, sd=sd, coord="z", extreme="zmin")
    if return_faces:
        return faces
    return cells


# -------- 9x9 representation of transversely isotropic stiffness tensor
"""The following functions are used to create a 9x9 matrix representation of a
transversely isotropic media with an arbitrary symmetry axis. The expression for the
stiffness tensor is taken from its tensor notation and split into five terms, one per
material parameter. """


def kronecker_delta(i_ind, j_ind):
    """Returns 1 if i_ind == j_ind, otherwise 0."""
    return 1 if i_ind == j_ind else 0


def _compute_term(index_map, n, tensor_func, *args):
    """Generalized function to compute stiffness terms."""
    term = np.zeros((9, 9), dtype=float)
    for i_ind in range(1, 4):  # i_ind, j_ind, k_ind, l_ind range from 1 to 3
        for j_ind in range(1, 4):
            for k_ind in range(1, 4):
                for l_ind in range(1, 4):
                    idx_ij = index_map[(i_ind, j_ind)]
                    idx_kl = index_map[(k_ind, l_ind)]
                    term[idx_ij, idx_kl] += tensor_func(
                        i_ind, j_ind, k_ind, l_ind, n, *args
                    )
    return term


def _term_1(i_ind, j_ind, k_ind, l_ind, n, lambda_val):
    return lambda_val * kronecker_delta(i_ind, j_ind) * kronecker_delta(k_ind, l_ind)


def _term_2(i_ind, j_ind, k_ind, l_ind, n, lambda_parallel):
    return lambda_parallel * (
        kronecker_delta(i_ind, j_ind) * kronecker_delta(k_ind, l_ind)
        - kronecker_delta(i_ind, j_ind) * n[k_ind - 1] * n[l_ind - 1]
        - kronecker_delta(k_ind, l_ind) * n[i_ind - 1] * n[j_ind - 1]
        + n[i_ind - 1] * n[j_ind - 1] * n[k_ind - 1] * n[l_ind - 1]
    )


def _term_3(i_ind, j_ind, k_ind, l_ind, n, lambda_perpendicular):
    return (
        lambda_perpendicular * n[i_ind - 1] * n[j_ind - 1] * n[k_ind - 1] * n[l_ind - 1]
    )


def _term_4(i_ind, j_ind, k_ind, l_ind, n, mu_parallel):
    return mu_parallel * (
        2 * n[i_ind - 1] * n[j_ind - 1] * n[k_ind - 1] * n[l_ind - 1]
        - kronecker_delta(i_ind, k_ind) * n[j_ind - 1] * n[l_ind - 1]
        - kronecker_delta(j_ind, k_ind) * n[i_ind - 1] * n[l_ind - 1]
        - kronecker_delta(i_ind, l_ind) * n[j_ind - 1] * n[k_ind - 1]
        - kronecker_delta(j_ind, l_ind) * n[i_ind - 1] * n[k_ind - 1]
        + kronecker_delta(i_ind, k_ind) * kronecker_delta(j_ind, l_ind)
        + kronecker_delta(i_ind, l_ind) * kronecker_delta(j_ind, k_ind)
    )


def _term_5(i_ind, j_ind, k_ind, l_ind, n, mu_perpendicular):
    return mu_perpendicular * (
        kronecker_delta(i_ind, k_ind) * n[j_ind - 1] * n[l_ind - 1]
        + kronecker_delta(j_ind, k_ind) * n[i_ind - 1] * n[l_ind - 1]
        + kronecker_delta(i_ind, l_ind) * n[j_ind - 1] * n[k_ind - 1]
        + kronecker_delta(j_ind, l_ind) * n[i_ind - 1] * n[k_ind - 1]
        - 2 * n[i_ind - 1] * n[j_ind - 1] * n[k_ind - 1] * n[l_ind - 1]
    )


def create_stiffness_tensor_basis(
    lambda_val, lambda_parallel, lambda_perpendicular, mu_parallel, mu_perpendicular, n
):
    """Creates 9x9 matrix representation of transversely isotropic tensor.
    
    Basis for each of the five material parameters used in a transversely isotropic
    media is generated here. The basis is matrices are constructed from helper functions
    term_1(), term_2(), term_3(), term_4() and term_5().

    All the values that are passed here (except for `n`) can be assigned in two ways:
        * Assign the value 1 to all of them. In that way, this function returns only the
          basis matrices for the stiffness tensor. This is useful in the case of
          constructing the stiffness tensor by passing the 9x9 matrix and the value of
          the material parameter (as it is used now). This is an easier choice if the
          goal is to later assign a heterogeneous tensor (e.g. transversely isotropic in
          one region and isotropic in another).
        * Assign values different form 1 to receive a dictionary of the 9x9 matrix
          representation with the set material values, and not only receive the "clean"
          basis matrices as described above.

    Parameters: 
        lambda_val: First Lamé parameter.
        lambda_parallel: Transverse compressive stress parameter.
        lambda_perpendicular: Perpendicular compressive stress parameter.
        mu_parallel: transverse shear parameter.
        mu_perpendicular: transverse-to-perpendicular shear parameter.
        n: The vector representing the symmetry axis of the transversely isotropic  
            medium.

    """
    # Define a mapping of indices to a 9x9 structure
    index_map = {
        (1, 1): 0,
        (1, 2): 1,
        (1, 3): 2,
        (2, 1): 3,
        (2, 2): 4,
        (2, 3): 5,
        (3, 1): 6,
        (3, 2): 7,
        (3, 3): 8,
    }

    # Compute each term using the generalized function
    stiffness_matrices = {
        "lambda": _compute_term(index_map, n, _term_1, lambda_val),
        "lambda_parallel": _compute_term(index_map, n, _term_2, lambda_parallel),
        "lambda_perpendicular": _compute_term(
            index_map, n, _term_3, lambda_perpendicular
        ),
        "mu_parallel": _compute_term(index_map, n, _term_4, mu_parallel),
        "mu_perpendicular": _compute_term(index_map, n, _term_5, mu_perpendicular),
    }

    return stiffness_matrices

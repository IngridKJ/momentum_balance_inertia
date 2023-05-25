"""
ParaView quad_time analytical solution:
(time)^2 * (sin(acos(-1) * coords[0]) * sin(acos(-1) * coords[1]) * iHat + sin(acos(-1) * coords[0]) * sin(acos(-1) * coords[1]) * jHat)

ParaView bubble analytical solution:
(time)^2 * (coords[0] * coords[1] * (1 - coords[0]) * (1 - coords[1]) * iHat + coords[0] * coords[1] * (1 - coords[0]) * (1 - coords[1]) * jHat)

ParaView quad_space analytical solution:
cos(2*acos(-1)*time) * (coords[0] * coords[1] * (1 - coords[0]) * (1 - coords[1]) * iHat + coords[0] * coords[1] * (1 - coords[0]) * (1 - coords[1]) * jHat)

cos(time) * (coords[0] * coords[1] * (1 - coords[0]) * (1 - coords[1]) * iHat + coords[0] * coords[1] * (1 - coords[0]) * (1 - coords[1]) * jHat)

"""

import numpy as np
import porepy as pp
from typing import Optional
import sympy as sym

# ------------------- Fetching values


def get_solution_values(
    name: str,
    data: dict,
    time_step_index: Optional[int] = None,
    iterate_index: Optional[int] = None,
) -> np.ndarray:
    """Function for fetching values stored in data dictionary.

    It looks very ugly to fetch values (that are not connected to a variable) from the
    data dictionary. Therefore this function will come in handy such that the keys
    `pp.TIME_STEP_SOLUTIONS` or `pp.ITERATE_SOLUTIONS` are not scattered all over the
    code.

    Parameters:
        name: Name of the parameter whose values we are interested in.
        data: The data dictionary.
        time_step_index: Which time step we want to get values for. 0 is current, 1 is
            one time step back in time. Only limited by how many time steps are stored
           from before.
        iterate_index: Which iterate we want to get values for. 0 is current, 1 is one
            iterate back in time. Only limited by how many iterates are stored from
            before.

    """
    if (time_step_index is None and iterate_index is None) or (
        time_step_index is not None and iterate_index is not None
    ):
        raise ValueError(
            "Both time_step_index and iterate_index cannot be None/assigned a value."
            "Only one at a time."
        )

    if time_step_index is not None:
        return data[pp.TIME_STEP_SOLUTIONS][name][time_step_index].copy()
    else:
        return data[pp.ITERATE_SOLUTIONS][name][iterate_index].copy()


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
    a_previous = get_solution_values(
        name=model.acceleration_key, data=data, time_step_index=0
    )
    v_previous = get_solution_values(
        name=model.velocity_key, data=data, time_step_index=0
    )

    u_current = get_solution_values(
        name=model.displacement_variable, data=data, iterate_index=0
    )

    u_previous = get_solution_values(
        name=model.displacement_variable, data=data, time_step_index=0
    )

    return a_previous, v_previous, u_previous, u_current


# ------------------- Manufactured solution functions


def symbolic_representation(model, return_dt=False, return_ddt=False):
    """Symbolic representation for displacements, velocities and acceleration.

    Parameters:
        model: The model class.
        return_dt: Return velocity if True.
        return_ddt: Return acceleration if True.

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

    manufactured_sol = model.params.get("manufactured_solution", "bubble")
    if manufactured_sol == "bubble":
        u1 = u2 = t**2 * x * (1 - x) * y * (1 - y)
        u = [u1, u2]
    elif manufactured_sol == "sin_bubble":
        u1 = u2 = sym.sin(sym.sqrt(2.0) * np.pi * t) * x * (1 - x) * y * (1 - y)
        u = [u1, u2]
    elif manufactured_sol == "cub_cub":
        u1 = u2 = t**3 * x**2 * y**2 * (1 - x) * (1 - y)
        u = [u1, u2]
    elif manufactured_sol == "quad_time":
        u1 = u2 = t**2 * sym.sin(np.pi * x) * sym.sin(np.pi * y)
        u = [u1, u2]
    elif manufactured_sol == "cub_time":
        u1 = u2 = t**3 * sym.sin(np.pi * x) * sym.sin(np.pi * y)
        u = [u1, u2]
    elif manufactured_sol == "quad_space":
        u1 = u2 = x * y * (1 - x) * (1 - y) * sym.cos(t)
        u = [u1, u2]

    if return_dt:
        dt_u = [sym.diff(u[0], t), sym.diff(u[0], t)]
        return dt_u, x, y, t
    elif return_ddt:
        ddt_u = [sym.diff(sym.diff(u[0], t), t), sym.diff(sym.diff(u[0], t), t)]
        return ddt_u, x, y, t

    # Here symbols are returned to avoid possible issues with references of created
    # symbols
    return u, x, y, t


def u_func_time(model) -> list:
    """Lambdified expression of displacement function."""
    u, x, y, t = symbolic_representation(model=model)
    u = [
        sym.lambdify((x, y, t), u[0], "numpy"),
        sym.lambdify((x, y, t), u[1], "numpy"),
    ]
    return u


def v_func_time(model) -> list:
    """Lambdified expression of velocity function."""
    v, x, y, t = symbolic_representation(model=model, return_dt=True)
    v = [
        sym.lambdify((x, y, t), v[0], "numpy"),
        sym.lambdify((x, y, t), v[1], "numpy"),
    ]
    return v


def a_func_time(model) -> list:
    """Lambdified expression of acceleration function."""
    a, x, y, t = symbolic_representation(model, return_ddt=True)
    a = [
        sym.lambdify((x, y, t), a[0], "numpy"),
        sym.lambdify((x, y, t), a[1], "numpy"),
    ]
    return a


# ------------------- Body force functions


def body_force_func(model) -> list:
    """Function for calculating rhs corresponding to a static manufactured solution, 2D."""
    lam = model.solid.lame_lambda()
    mu = model.solid.shear_modulus()

    x, y = sym.symbols("x y")

    # Manufactured solution (bubble<3)
    u1 = u2 = x * (1 - x) * y * (1 - y)
    u = [u1, u2]

    grad_u = [
        [sym.diff(u[0], x), sym.diff(u[0], y)],
        [sym.diff(u[1], x), sym.diff(u[1], y)],
    ]

    grad_u_T = [[grad_u[0][0], grad_u[1][0]], [grad_u[0][1], grad_u[1][1]]]

    div_u = sym.diff(u[0], x) + sym.diff(u[1], y)

    trace_grad_u = grad_u[0][0] + grad_u[1][1]

    strain = 0.5 * np.array(
        [
            [grad_u[0][0] + grad_u_T[0][0], grad_u[0][1] + grad_u_T[0][1]],
            [grad_u[1][0] + grad_u_T[1][0], grad_u[1][1] + grad_u_T[1][1]],
        ]
    )

    sigma = [
        [2 * mu * strain[0][0] + lam * trace_grad_u, 2 * mu * strain[0][1]],
        [2 * mu * strain[1][0], 2 * mu * strain[1][1] + lam * trace_grad_u],
    ]

    div_sigma = [
        sym.diff(sigma[0][0], x) + sym.diff(sigma[0][1], y),
        sym.diff(sigma[1][0], x) + sym.diff(sigma[1][1], y),
    ]

    return [
        sym.lambdify((x, y), div_sigma[0], "numpy"),
        sym.lambdify((x, y), div_sigma[1], "numpy"),
    ]


def body_force_func_time(model) -> list:
    """Function for calculating rhs corresponding to a manufactured solution, 2D."""
    lam = model.solid.lame_lambda()
    mu = model.solid.shear_modulus()
    rho = model.solid.density()

    u, x, y, t = symbolic_representation(model=model)

    ddt_u = [
        sym.diff(sym.diff(u[0], t), t),
        sym.diff(sym.diff(u[1], t), t),
    ]

    grad_u = [
        [sym.diff(u[0], x), sym.diff(u[0], y)],
        [sym.diff(u[1], x), sym.diff(u[1], y)],
    ]

    grad_u_T = [[grad_u[0][0], grad_u[1][0]], [grad_u[0][1], grad_u[1][1]]]

    div_u = sym.diff(u[0], x) + sym.diff(u[1], y)

    trace_grad_u = grad_u[0][0] + grad_u[1][1]

    strain = 0.5 * np.array(
        [
            [grad_u[0][0] + grad_u_T[0][0], grad_u[0][1] + grad_u_T[0][1]],
            [grad_u[1][0] + grad_u_T[1][0], grad_u[1][1] + grad_u_T[1][1]],
        ]
    )

    sigma = [
        [2 * mu * strain[0][0] + lam * trace_grad_u, 2 * mu * strain[0][1]],
        [2 * mu * strain[1][0], 2 * mu * strain[1][1] + lam * trace_grad_u],
    ]

    div_sigma = [
        sym.diff(sigma[0][0], x) + sym.diff(sigma[0][1], y),
        sym.diff(sigma[1][0], x) + sym.diff(sigma[1][1], y),
    ]

    acceleration_term = [rho * ddt_u[0], rho * ddt_u[1]]

    full_eqn = [
        acceleration_term[0] - div_sigma[0],
        acceleration_term[1] - div_sigma[1],
    ]

    return [
        sym.lambdify((x, y, t), full_eqn[0], "numpy"),
        sym.lambdify((x, y, t), full_eqn[1], "numpy"),
    ]


def body_force_func_time_3D(model) -> list:
    """Function for calculating rhs corresponding to a manufactured solution, 2D."""
    lam = model.solid.lame_lambda()
    mu = model.solid.shear_modulus()
    rho = model.solid.density()

    x, y, z, t = sym.symbols("x y z t")

    manufactured_sol = model.params.get("manufactured_solution", "bubble")
    if manufactured_sol == "bubble":
        # Manufactured solution (bubble<3)
        u1 = u2 = u3 = t**2 * x * (1 - x) * y * (1 - y) * z * (1 - z)
        u = [u1, u2, u3]
    elif manufactured_sol == "quad_time":
        u1 = u2 = u3 = (
            t**2 * sym.sin(np.pi * x) * sym.sin(np.pi * y) * sym.sin(np.pi * z)
        )
        u = [u1, u2, u3]
    elif manufactured_sol == "quad_space":
        raise NotImplementedError

    ddt_u = [
        sym.diff(sym.diff(u[0], t), t),
        sym.diff(sym.diff(u[1], t), t),
        sym.diff(sym.diff(u[2], t), t),
    ]

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

    div_u = sym.diff(u[0], x) + sym.diff(u[1], y) + sym.diff(u[2], z)

    trace_grad_u = grad_u[0][0] + grad_u[1][1] + grad_u[2][2]

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
                grad_u[2][1] + grad_u_T[2][1],
            ],
            [
                grad_u[2][0] + grad_u_T[2][0],
                grad_u[2][1] + grad_u_T[2][1],
                grad_u[2][2] + grad_u_T[2][2],
            ],
        ]
    )

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

    div_sigma = [
        sym.diff(sigma[0][0], x) + sym.diff(sigma[0][1], y),
        sym.diff(sigma[0][2], z),
        sym.diff(sigma[1][0], x) + sym.diff(sigma[1][1], y),
        sym.diff(sigma[1][2], z),
        sym.diff(sigma[2][0], x) + sym.diff(sigma[2][1], y),
        sym.diff(sigma[2][2], z),
    ]

    acceleration_term = [rho * ddt_u[0], rho * ddt_u[1], rho * ddt_u[2]]

    full_eqn = [
        acceleration_term[0] - div_sigma[0],
        acceleration_term[1] - div_sigma[1],
        acceleration_term[2] - div_sigma[2],
    ]

    return [
        sym.lambdify((x, y, z, t), full_eqn[0], "numpy"),
        sym.lambdify((x, y, z, t), full_eqn[1], "numpy"),
        sym.lambdify((x, y, z, t), full_eqn[2], "numpy"),
    ]

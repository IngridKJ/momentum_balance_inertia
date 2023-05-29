import porepy as pp


def inertia_term(
    model,
    op: pp.ad.Operator,
    dt_op: pp.ad.Operator,
    ddt_op: pp.ad.Operator,
    time_step: pp.ad.Scalar,
) -> pp.ad.Operator:
    """Operator for second derivative term, for now only used in momentum balance eqn.

    Discretized with Newmark method.

    Parameters:
        model: Model that is run. Needed for accessing the beta-constant.
        op: Variable operator for equation.
        dt_op: First time derivative of op operator.
        ddt_op: Second time derivative of op operator.

    """
    beta = model.beta

    acceleration = (
        (op - op.previous_timestep()) / (time_step**2 * beta)
        - dt_op.previous_timestep() / (time_step * beta)
        - ddt_op.previous_timestep() * (1 - 2 * beta) / (2 * beta)
    )
    return acceleration

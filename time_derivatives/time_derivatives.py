import porepy as pp


def inertia_term(
    model,
    op: pp.ad.Operator,
    dt_op: pp.ad.Operator,
    ddt_op: pp.ad.Operator,
    time_step: pp.ad.Scalar,
) -> pp.ad.Operator:
    """Operator for the inertial term in the dynamic balance equation.

    Parameters:
        model: Model that is run. Needed for accessing the beta-constant.
        op:
        dt_op: First time derivative of op operator.
        ddt_op: Second time derivative of op operator.
    """
    beta = model.beta

    acceleration = (
        (op - op.previous_timestep()) / (time_step**2)
        - dt_op.previous_timestep() / time_step
        - ddt_op.previous_timestep() * (1 - 2 * beta)
    )
    return acceleration

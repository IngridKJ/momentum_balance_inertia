import porepy as pp
from porepy.numerics.ad.time_derivatives import time_increment


def inertia_term(
    model,
    op: pp.ad.Operator,
    dt_op: pp.ad.Operator,
    ddt_op: pp.ad.Operator,
    time_step: pp.ad.Scalar,
) -> pp.ad.Operator:
    """This one is probably fine. Keep it as an operator"""
    beta = model.beta

    increment = (
        op
        - op.previous_timestep
        - time_step * dt_op.previous_timestep()
        - (1 - 2 * beta) * time_step * time_step / 2 * ddt_op.previous_timestep()
    )
    return increment / (time_step * time_step)
